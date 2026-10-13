"""Draft-side confidence for the ragged verify budget, without a confidence head.

The budget planner consumes a per-slot conditional confidence c_k and sorts by
the survival cumprod(c). The entropy source maps the draft's own per-slot
entropy to a calibrated survival chain through a quantile-bin table fitted
offline, then hands the planner the conditional ratios of that chain. The
draft_prob source is D-cut's uncalibrated prefix product of q(drafted token).
"""

from __future__ import annotations

import enum
from typing import Optional

import msgspec
import torch


class ConfidenceSource(str, enum.Enum):
    HEAD = "head"
    ENTROPY = "entropy"
    # D-cut (arXiv 2607.14647): raw draft probability of the drafted tokens, used
    # in its lossless shifted form (App. B) -- slot k scores prod_{t<k} q(z_t).
    DRAFT_PROB = "draft_prob"


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


def shifted_draft_prob(q_drawn: torch.Tensor) -> torch.Tensor:
    """D-cut's shifted score as per-slot conditionals: slot 1 gets 1, slot t gets
    q(z_{t-1}), so cumprod gives prod_{t<k} q(z_t). The admission of slot k never
    depends on the token drawn at k, which keeps rejection sampling exact."""
    q = q_drawn.to(torch.float32).clamp(0.0, 1.0)
    return torch.cat([torch.ones_like(q[:, :1]), q[:, :-1]], dim=1)


def drawn_prob_from_logits(
    logits: torch.Tensor, temperatures: Optional[torch.Tensor], tokens: torch.Tensor
) -> torch.Tensor:
    """q(z) of the drafted tokens under softmax(logits / T); logits [bs, gamma, V]."""
    x = logits.to(torch.float32)
    if temperatures is not None:
        x = x / temperatures.to(torch.float32).view(-1, 1, 1).clamp_min(1e-5)
    logp = torch.log_softmax(x, dim=-1)
    return torch.gather(logp, 2, tokens.to(torch.int64).unsqueeze(-1)).squeeze(-1).exp()


def drawn_prob_from_candidates(
    *, q_rows: torch.Tensor, candidate_ids: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    """q(z) of the drafted tokens from a sparse top-k q ([bs, gamma, K])."""
    match = candidate_ids == tokens.to(candidate_ids.dtype).unsqueeze(-1)
    return (q_rows.to(torch.float32) * match).sum(-1)


def entropy_confidence(
    *,
    entropy: torch.Tensor,
    calibrator: EntropyCalibrator,
    calibrator_nothink: Optional[EntropyCalibrator] = None,
    thinking: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[bs, gamma] entropy -> [bs, gamma] conditional confidence in [0, 1]. With a
    second table, rows whose request is not in thinking mode ([bs] bool) use it."""
    raw = raw_survival_chain(torch.exp(-entropy))
    survival = calibrator.survival(raw)
    if calibrator_nothink is not None and thinking is not None:
        survival = torch.where(
            thinking.view(-1, 1), survival, calibrator_nothink.survival(raw)
        )
    return conditional_from_survival(survival)


def prompt_opens_thinking(ids, *, think_start: int, think_end: int) -> bool:
    """True when the prompt leaves a reasoning segment open: its last think marker
    is <think>. The Qwen3.x templates end the generation prompt with "<think>\n"
    (thinking) or "<think>\n\n</think>\n\n" (not); DeepSeek-V4/V4.1 with
    "<think>" or "</think>". A prompt with neither marker counts as not thinking."""
    for token in reversed(ids):
        if token == think_end:
            return False
        if token == think_start:
            return True
    return False


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
        # Optional table for requests not in thinking mode; None = one table for all.
        self.calibrator_nothink: Optional[EntropyCalibrator] = None
        # (<think>, </think>) ids and the request-pool getter; None = one table for all.
        self._think_markers: Optional[tuple[int, int]] = None
        self._pool_getter = None
        table_path = envs.SGLANG_SPEC_ENTROPY_TABLE_PATH.get()
        if table_path:
            self.load_table(table_path)
        nothink_path = envs.SGLANG_SPEC_ENTROPY_TABLE_PATH_NOTHINK.get()
        if nothink_path:
            self.calibrator_nothink = EntropyCalibrator(
                table=EntropyCalibrationTable.load(nothink_path),
                gamma=self.gamma,
                device=self.device,
            )
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

    def configure_thinking(self, *, tokenizer, req_to_token_pool_getter) -> None:
        """Per-request table choice: the think markers come from the tokenizer, the
        per-row flags live on the request pool (built after the workers, hence the
        getter) and are written by mark_prompts on every prefill."""
        if self.calibrator_nothink is None:
            return
        if tokenizer is None:
            raise ValueError(
                "SGLANG_SPEC_ENTROPY_TABLE_PATH_NOTHINK reads the prompt's think markers "
                "and needs the tokenizer (not --skip-tokenizer-init)"
            )
        vocab = tokenizer.get_vocab()
        start, end = vocab.get("<think>"), vocab.get("</think>")
        if start is None or end is None:
            raise ValueError(
                "SGLANG_SPEC_ENTROPY_TABLE_PATH_NOTHINK needs <think> and </think> "
                "tokens in the tokenizer vocabulary"
            )
        self._think_markers = (start, end)
        self._pool_getter = req_to_token_pool_getter

    def mark_prompts(self, reqs) -> None:
        """Called on prefill batches. Every request is prefilled after it is bound to
        its pool row and before its first draft step, including re-prefill after a
        retract or preemption; the flag depends on the prompt alone."""
        if self._think_markers is None or not reqs:
            return
        start, end = self._think_markers
        table = self._pool_getter().enable_prompt_thinking()
        rows = torch.tensor([r.kv.req_pool_idx for r in reqs], dtype=torch.int64)
        flags = torch.tensor(
            [prompt_opens_thinking(r.origin_input_ids, think_start=start, think_end=end)
             for r in reqs],
            dtype=torch.bool,
        )
        table[rows.to(table.device)] = flags.to(table.device)

    def thinking_rows(self, req_pool_indices: torch.Tensor) -> Optional[torch.Tensor]:
        if self._think_markers is None:
            return None
        table = self._pool_getter().enable_prompt_thinking()
        return table[req_pool_indices.to(torch.int64)]

    def entropy_confidence(
        self, *, entropy: torch.Tensor, req_pool_indices: torch.Tensor
    ) -> torch.Tensor:
        return entropy_confidence(
            entropy=entropy,
            calibrator=self.calibrator,
            calibrator_nothink=self.calibrator_nothink,
            thinking=self.thinking_rows(req_pool_indices),
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

    @property
    def uses_draft_prob(self) -> bool:
        return self.source is ConfidenceSource.DRAFT_PROB
