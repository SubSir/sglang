"""Fused draft-entropy confidence: one launch per step instead of the ~25 small torch kernels of
entropy_from_probs -> raw_survival_chain -> EntropyCalibrator.survival -> conditional_from_survival
(sglang.srt.speculative.spec_confidence), with the optional per-row second (no-thinking) table.

One program per request walks its gamma slots in order, carrying the raw chain's running min and
the calibrated survival's running min, so every per-slot op of the torch reference is a scalar
recurrence here.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _entropy_confidence_kernel(
    q_ptr,
    stride_qb,
    stride_qg,
    stride_qk,
    ent_ptr,
    conf_ptr,
    interior_ptr,  # [2, G, E] fp32, +inf padded
    values_ptr,  # [2, G, E + 1] fp32
    num_bins_ptr,  # [2, G] int32
    think_ptr,  # [bs] bool (int8); row uses table 0 if thinking else table 1
    HAS_THINK: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    E: tl.constexpr,
    BE: tl.constexpr,
):
    b = tl.program_id(0)
    ok = tl.arange(0, BK)
    km = ok < K
    oe = tl.arange(0, BE)
    em = oe < E
    t = 0
    if HAS_THINK:
        t = tl.where(tl.load(think_ptr + b) != 0, 0, 1)
    chain = 1.0
    raw = 1.0
    surv = 1.0
    prev = 1.0
    for g in tl.static_range(G):
        p = tl.load(q_ptr + b * stride_qb + g * stride_qg + ok * stride_qk, mask=km, other=0.0).to(tl.float32)
        h = -tl.sum(p * tl.log(tl.maximum(p, 1e-30)), axis=0)
        tl.store(ent_ptr + b * G + g, h)
        x = tl.minimum(tl.maximum(tl.exp(-h), 0.0), 1.0)
        chain = chain * x
        raw = tl.minimum(raw, chain)
        edges = tl.load(interior_ptr + (t * G + g) * E + oe, mask=em, other=float("inf"))
        nbin = tl.sum(tl.where(raw >= edges, 1, 0), axis=0)
        nmax = tl.load(num_bins_ptr + t * G + g)
        nbin = tl.minimum(nbin, nmax - 1)
        v = tl.load(values_ptr + (t * G + g) * (E + 1) + nbin)
        v = tl.minimum(tl.maximum(v, 0.0), 1.0)
        surv = tl.minimum(surv, v)
        c = tl.where(prev > 0, surv / tl.maximum(prev, 1e-12), 0.0)
        tl.store(conf_ptr + b * G + g, tl.minimum(tl.maximum(c, 0.0), 1.0))
        prev = surv


def stack_calibration_tables(calibrators) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad EntropyCalibrator tables to one interior width and stack them: ([T, G, E], [T, G, E+1], [T, G])."""
    E = max(c._interior.shape[1] for c in calibrators)
    G = calibrators[0]._interior.shape[0]
    dev = calibrators[0]._interior.device
    interior = torch.full((len(calibrators), G, max(E, 1)), float("inf"), dtype=torch.float32, device=dev)
    values = torch.zeros((len(calibrators), G, max(E, 1) + 1), dtype=torch.float32, device=dev)
    num_bins = torch.zeros((len(calibrators), G), dtype=torch.int32, device=dev)
    for i, c in enumerate(calibrators):
        interior[i, :, : c._interior.shape[1]] = c._interior
        values[i, :, : c._values.shape[1]] = c._values
        num_bins[i] = c._num_bins.to(torch.int32)
    return interior, values, num_bins


def fused_entropy_confidence(
    q_rows: torch.Tensor,
    tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    thinking: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q_rows [bs, gamma, K] sparse draft distributions -> (entropy, conditional confidence), [bs, gamma] fp32.
    ``tables`` from stack_calibration_tables([thinking table, no-thinking table]) (one table: stack of one);
    ``thinking`` [bs] bool selects table 0 (True) or 1 (False) per row."""
    bs, G, K = q_rows.shape
    interior, values, num_bins = tables
    if thinking is not None and interior.shape[0] < 2:
        raise ValueError("per-row thinking flags need two stacked tables")
    ent = torch.empty((bs, G), dtype=torch.float32, device=q_rows.device)
    conf = torch.empty((bs, G), dtype=torch.float32, device=q_rows.device)
    if bs == 0:
        return ent, conf
    E = interior.shape[2]
    _entropy_confidence_kernel[(bs,)](
        q_rows,
        q_rows.stride(0),
        q_rows.stride(1),
        q_rows.stride(2),
        ent,
        conf,
        interior,
        values,
        num_bins,
        thinking.to(torch.int8) if thinking is not None else ent,
        HAS_THINK=thinking is not None,
        G=G,
        K=K,
        BK=triton.next_power_of_2(K),
        E=E,
        BE=triton.next_power_of_2(E),
    )
    return ent, conf
