"""GPU-side helpers for DFlash ragged verify (adaptive lengths + CUDA graph padding).

These ops stay on device to avoid per-request Python loops and redundant CPU syncs.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def prepare_dflash_adaptive_verify(
    *,
    draft_tokens: torch.Tensor,
    positions: Optional[torch.Tensor],
    verify_len_actual: torch.Tensor,
    target_total_tokens: int,
    verify_len_padded: torch.Tensor,
    start_offsets: torch.Tensor,
    packed_tokens: torch.Tensor,
    packed_positions: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Fill padded per-request lengths, exclusive start offsets, and packed token/position buffers.

    Padding distribution matches the historical round-robin rule used in ``DFlashWorker``:
    each step assigns at most one extra verify slot to each request that still has room
    below ``block_size``, in increasing index order, until the deficit is zero.
    """
    verify_len_padded.copy_(verify_len_actual)
    bs = int(verify_len_actual.shape[0])
    if bs == 0:
        return verify_len_padded, start_offsets, packed_tokens, packed_positions

    block_size_i = int(draft_tokens.shape[1])
    current_sum = int(verify_len_padded.sum().item())
    deficit = int(target_total_tokens) - current_sum
    if deficit > 0:
        remaining = deficit
        while remaining > 0:
            can_take = verify_len_padded < block_size_i
            avail_idx = torch.nonzero(can_take, as_tuple=False).flatten()
            if avail_idx.numel() == 0:
                break
            take_n = min(remaining, int(avail_idx.numel()))
            sel = avail_idx[:take_n]
            verify_len_padded.index_add_(
                0,
                sel,
                torch.ones(
                    (take_n,),
                    dtype=verify_len_padded.dtype,
                    device=verify_len_padded.device,
                ),
            )
            remaining -= take_n

    lens_i64 = verify_len_padded.to(torch.int64)
    cum = lens_i64.cumsum(dim=0)
    starts = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=verify_len_padded.device),
            cum[:-1],
        ]
    ).to(dtype=start_offsets.dtype)
    start_offsets.copy_(starts)

    max_a = int(lens_i64.max().item())
    if max_a > 0:
        device = draft_tokens.device
        rows = torch.arange(max_a, device=device, dtype=torch.int64).unsqueeze(0).expand(
            bs, -1
        )
        batch_i = torch.arange(bs, device=device, dtype=torch.int64).unsqueeze(1).expand(
            -1, max_a
        )
        mask = rows < lens_i64.unsqueeze(1)
        dst_idx = start_offsets.to(torch.int64).unsqueeze(1) + rows
        packed_tokens[dst_idx[mask]] = draft_tokens[batch_i, rows][mask]
        if positions is not None and packed_positions is not None:
            packed_positions[dst_idx[mask]] = positions[batch_i, rows][mask]

    return verify_len_padded, start_offsets, packed_tokens, packed_positions


def accept_dflash_adaptive_verify(
    *,
    draft_token_flat: torch.Tensor,
    target_predict_flat: torch.Tensor,
    actual_verify_lens: torch.Tensor,
    start_offsets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched greedy DFlash accept length + bonus token (same rule as ``compute_dflash_accept_len_and_bonus``).

    Args:
        draft_token_flat: Packed draft token ids, shape ``[total_tokens]``.
        target_predict_flat: Per-position argmax ids aligned with the verify forward, shape ``[total_tokens]``.
        actual_verify_lens: Per-request verify lengths (unpadded), shape ``[bs]``, int32/int64.
        start_offsets: Exclusive start index per request into the flat buffers, shape ``[bs]``, int32/int64.

    Returns:
        ``(accept_len, bonus)`` int32 ``[bs]`` and int64 ``[bs]``.
    """
    device = draft_token_flat.device
    bs = int(actual_verify_lens.shape[0])
    if bs == 0:
        z = torch.empty((0,), dtype=torch.int32, device=device)
        zb = torch.empty((0,), dtype=torch.int64, device=device)
        return z, zb

    max_v = int(actual_verify_lens.max().item())
    if max_v <= 0:
        z = torch.zeros((bs,), dtype=torch.int32, device=device)
        zb = torch.zeros((bs,), dtype=torch.int64, device=device)
        return z, zb

    idx = torch.arange(max_v, device=device, dtype=torch.int64).unsqueeze(0).expand(bs, -1)
    starts = start_offsets.to(torch.int64).unsqueeze(1)
    flat_idx = starts + idx
    row_lens = actual_verify_lens.to(torch.int64).unsqueeze(1)
    valid = idx < row_lens

    cand = torch.zeros((bs, max_v), dtype=draft_token_flat.dtype, device=device)
    pred = torch.zeros((bs, max_v), dtype=target_predict_flat.dtype, device=device)
    cand[valid] = draft_token_flat[flat_idx[valid]]
    pred[valid] = target_predict_flat[flat_idx[valid]]

    if max_v == 1:
        accept_len = torch.zeros((bs,), dtype=torch.int32, device=device)
        bonus = pred[:, 0].to(torch.int64)
        return accept_len, bonus

    ar = torch.arange(max_v - 1, device=device, dtype=torch.int64)
    len_m1 = (actual_verify_lens.to(torch.int64) - 1).clamp(min=0).unsqueeze(1)
    match_mask = ar.unsqueeze(0) < len_m1
    matches = (cand[:, 1:] == pred[:, :-1]) & match_mask
    accept_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    bonus = pred[torch.arange(bs, device=device, dtype=torch.int64), accept_len.to(torch.int64)]
    return accept_len, bonus.to(torch.int64)
