"""Opt-in per-verify-step JSONL trace for dynamic-verify experiments.

Enabled by SGLANG_SPEC_STEP_TRACE_PATH (a path prefix; rank 0 writes
``<prefix>.jsonl``). Every record forces a device sync, so leave it off for
timing runs; it exists to collect calibration data and to replay budgets.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import torch

from sglang.srt.environ import envs


def selector_alpha(
    *,
    q_rows: torch.Tensor,
    candidate_ids: torch.Tensor,
    target_logits: torch.Tensor,
    temperatures: Optional[torch.Tensor],
) -> torch.Tensor:
    """Exact rejection-sampling acceptance per slot for a sparse draft q:
    alpha_j = sum_c min(q_j(c), p_j(c)) over the candidates of slot j.

    q_rows / candidate_ids are [bs, gamma, K]; target_logits [bs * (gamma+1), V],
    where row j of a request scores the token that follows verify input j."""
    bs, gamma, _ = q_rows.shape
    logits = target_logits.view(bs, gamma + 1, -1)[:, :gamma].to(torch.float32)
    if temperatures is not None:
        logits = logits / temperatures.to(torch.float32).view(-1, 1, 1).clamp_min(1e-5)
    p = torch.softmax(logits, dim=-1)
    p_cand = torch.gather(p, 2, candidate_ids.to(torch.int64))
    return torch.minimum(q_rows.to(torch.float32), p_cand).sum(-1)


class SpecStepTrace:
    def __init__(self, *, tp_rank: int, tag: str) -> None:
        prefix = envs.SGLANG_SPEC_STEP_TRACE_PATH.get()
        self._file = None
        if prefix and tp_rank == 0:
            self._file = open(f"{prefix}.jsonl", "a", buffering=1)
        self._tag = tag

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def write(
        self,
        *,
        rids: list[str],
        budget: Optional[int],
        verify_lens: Optional[torch.Tensor],
        commit_lens: torch.Tensor,
        cap_trim_lens: Optional[torch.Tensor],
        source: str,
        entropy: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
    ) -> None:
        if self._file is None:
            return

        def rows(x):
            return None if x is None else x.detach().float().cpu().tolist()

        rec = {
            "tag": self._tag,
            "t": time.time(),
            "bs": len(rids),
            "rids": rids,
            "source": source,
            "budget": budget,
            "verify_lens": None if verify_lens is None else verify_lens.tolist(),
            "commit_lens": commit_lens.tolist(),
            "cap_trim_lens": None if cap_trim_lens is None else cap_trim_lens.tolist(),
            "entropy": rows(entropy),
            "confidence": rows(confidence),
            "alpha": rows(alpha),
        }
        self._file.write(json.dumps(rec) + "\n")
