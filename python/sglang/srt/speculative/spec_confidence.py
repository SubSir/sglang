"""Draft-side confidence for the ragged verify budget, without a confidence head.

The budget planner consumes a per-slot conditional confidence c_k and sorts by
the survival cumprod(c). The entropy source maps the draft's own per-slot
entropy to a calibrated survival chain through a quantile-bin table fitted
offline, then hands the planner the conditional ratios of that chain.
"""

from __future__ import annotations

import enum
from typing import Optional

import msgspec
import torch


class ConfidenceSource(str, enum.Enum):
    HEAD = "head"
    ENTROPY = "entropy"


class EntropyCalibrationTable(msgspec.Struct, frozen=True):
    """One table per (temperature, thinking mode); fitted by
    ``analyze_sink.calibrate`` on the survival chain, 20 quantile bins per slot.

    ``edges[k]`` are the unique bin edges of slot k's raw chain value and
    ``values[k][b]`` the mean true survival of bin b (len(edges[k]) - 1 bins).
    """

    edges: list[list[float]]
    values: list[list[float]]
    signal: str = "exp_neg_entropy_chain"
    temperature: float = 1.0
    thinking: str = ""

    @property
    def gamma(self) -> int:
        return len(self.edges)

    def validate(self, *, gamma: int) -> None:
        if len(self.edges) != gamma or len(self.values) != gamma:
            raise ValueError(
                f"entropy calibration table has {len(self.edges)} slots but the "
                f"draft proposes {gamma} tokens per block; refit it for gamma={gamma}."
            )
        for k, (e, v) in enumerate(zip(self.edges, self.values)):
            if len(e) < 2:
                if len(v) != 1:
                    raise ValueError(f"slot {k}: a constant slot needs one value")
            elif len(v) != len(e) - 1:
                raise ValueError(
                    f"slot {k}: {len(e)} edges need {len(e) - 1} values, got {len(v)}"
                )

    @classmethod
    def from_json(cls, data: str | bytes) -> "EntropyCalibrationTable":
        return msgspec.json.decode(data, type=cls)

    @classmethod
    def load(cls, path: str) -> "EntropyCalibrationTable":
        with open(path, "rb") as f:
            return cls.from_json(f.read())


class EntropyCalibrator:
    """Device-resident copy of a table; padded to a common bin count so the
    lookup is one bucketize per slot."""

    def __init__(self, *, table: EntropyCalibrationTable, gamma: int, device) -> None:
        table.validate(gamma=gamma)
        self.table = table
        max_edges = max(max(len(e), 2) for e in table.edges)
        # Pad interior edges with +inf so padded bins are never selected.
        interior = torch.full((gamma, max_edges - 2), float("inf"), dtype=torch.float32)
        values = torch.zeros((gamma, max_edges - 1), dtype=torch.float32)
        for k, (e, v) in enumerate(zip(table.edges, table.values)):
            if len(e) >= 2:
                interior[k, : len(e) - 2] = torch.tensor(e[1:-1], dtype=torch.float32)
            values[k, : len(v)] = torch.tensor(v, dtype=torch.float32)
        self._interior = interior.to(device)
        self._values = values.to(device)
        self._num_bins = torch.tensor(
            [max(len(e) - 1, 1) for e in table.edges], dtype=torch.int64
        ).to(device)

    def survival(self, raw_chain: torch.Tensor) -> torch.Tensor:
        # searchsorted(edges, x, "right") - 1 over all edges == count of interior
        # edges <= x, clipped to the last bin: matches analyze_sink.calibrate.
        x = raw_chain.to(torch.float32)
        bins = (x.unsqueeze(-1) >= self._interior.unsqueeze(0)).sum(-1)
        bins = torch.minimum(bins, self._num_bins.unsqueeze(0) - 1)
        surv = torch.gather(
            self._values.unsqueeze(0).expand(x.shape[0], -1, -1),
            2,
            bins.unsqueeze(-1),
        ).squeeze(-1)
        return torch.cummin(surv.clamp(0.0, 1.0), dim=1).values


def raw_survival_chain(exp_neg_entropy: torch.Tensor) -> torch.Tensor:
    """chain_of(x) of the offline scripts: running min of the clipped cumprod."""
    chain = torch.cumprod(exp_neg_entropy.to(torch.float32).clamp(0.0, 1.0), dim=1)
    return torch.cummin(chain, dim=1).values


def conditional_from_survival(survival: torch.Tensor) -> torch.Tensor:
    """Per-slot c_k = S_k / S_{k-1} (S_0 = 1), so cumprod(c) gives back S."""
    prev = torch.cat([torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1)
    ratio = torch.where(prev > 0, survival / prev.clamp_min(1e-12), 0.0)
    return ratio.clamp(0.0, 1.0)


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Entropy over the last dim of a (possibly sparse-support) distribution."""
    p = probs.to(torch.float32)
    return -(p * p.clamp_min(1e-30).log()).sum(-1)


def entropy_from_logits(
    logits: torch.Tensor, temperatures: Optional[torch.Tensor]
) -> torch.Tensor:
    """Entropy of softmax(logits / T) per row; logits [bs, gamma, vocab]."""
    x = logits.to(torch.float32)
    if temperatures is not None:
        x = x / temperatures.to(torch.float32).view(-1, 1, 1).clamp_min(1e-5)
    logp = torch.log_softmax(x, dim=-1)
    return -(logp.exp() * logp).sum(-1)


def entropy_confidence(
    *, entropy: torch.Tensor, calibrator: EntropyCalibrator
) -> torch.Tensor:
    """[bs, gamma] entropy -> [bs, gamma] conditional confidence in [0, 1]."""
    survival = calibrator.survival(raw_survival_chain(torch.exp(-entropy)))
    return conditional_from_survival(survival)


class DraftConfidenceSelector:
    """Which signal feeds the verify budget, switchable at runtime so every
    arm of a comparison can share one server."""

    def __init__(
        self, *, gamma: int, device, default_source: ConfidenceSource, has_head: bool
    ) -> None:
        from sglang.srt.environ import envs

        self.gamma = int(gamma)
        self.device = device
        self.has_head = has_head
        self.calibrator: Optional[EntropyCalibrator] = None
        table_path = envs.SGLANG_SPEC_ENTROPY_TABLE_PATH.get()
        if table_path:
            self.load_table(table_path)
        requested = envs.SGLANG_SPEC_CONFIDENCE_SOURCE.get()
        self.source = default_source
        self.set_source(requested or default_source.value)

    @property
    def entropy_ready(self) -> bool:
        return self.calibrator is not None

    def load_table(self, path: str) -> None:
        self.calibrator = EntropyCalibrator(
            table=EntropyCalibrationTable.load(path), gamma=self.gamma, device=self.device
        )

    def set_source(self, name: str) -> None:
        source = ConfidenceSource(name)
        if source is ConfidenceSource.HEAD and not self.has_head:
            raise ValueError("confidence source 'head' needs a draft confidence head")
        if source is ConfidenceSource.ENTROPY and self.calibrator is None:
            raise ValueError(
                "confidence source 'entropy' needs a calibration table "
                "(SGLANG_SPEC_ENTROPY_TABLE_PATH or the spec_entropy_table_path knob)"
            )
        self.source = source

    @property
    def uses_entropy(self) -> bool:
        return self.source is ConfidenceSource.ENTROPY
