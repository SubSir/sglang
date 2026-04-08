"""GPU-side helpers for DFlash ragged verify (adaptive lengths + CUDA graph padding).

These ops stay on device to avoid per-request Python loops and redundant CPU syncs.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _dflash_adaptive_verify_proposed_kernel(
    draft_ptr,
    pred_ptr,
    actual_lens_ptr,
    start_ptr,
    accept_out_ptr,
    bonus_out_ptr,
    proposed_ptr,
    proposed_count_ptr,
    stride_proposed_row: tl.constexpr,
    MAX_V: tl.constexpr,
):
    """One program per batch row: greedy accept length, bonus, packed proposed tokens."""
    pid = tl.program_id(axis=0)
    vlen = tl.load(actual_lens_ptr + pid).to(tl.int32)
    st = tl.load(start_ptr + pid).to(tl.int64)

    if vlen <= 0:
        tl.store(accept_out_ptr + pid, 0)
        tl.store(bonus_out_ptr + pid, 0)
        tl.store(proposed_count_ptr + pid, 0)
        return

    acc_len = tl.full((), 0, tl.int32)
    cont = tl.full((), 1, tl.int32)
    for j in tl.static_range(MAX_V - 1):
        active = (j < (vlen - 1)) & (cont != 0)
        c = tl.load(draft_ptr + st + j + 1, mask=active, other=0)
        p = tl.load(pred_ptr + st + j, mask=active, other=1)
        m = (c == p) & active
        acc_len = acc_len + (cont & m.to(tl.int32))
        cont = cont & m.to(tl.int32)

    bonus = tl.load(pred_ptr + st + acc_len.to(tl.int64))

    tl.store(accept_out_ptr + pid, acc_len)
    tl.store(bonus_out_ptr + pid, bonus.to(tl.int64))

    prop_count = acc_len + 1
    tl.store(proposed_count_ptr + pid, prop_count)

    row_base = pid * stride_proposed_row
    bonus_i64 = bonus.to(tl.int64)
    for k in tl.static_range(MAX_V):
        active_k = k < prop_count
        is_draft = k < acc_len
        dv = tl.load(draft_ptr + st + k + 1, mask=is_draft & active_k, other=0)
        val = tl.where(is_draft & active_k, dv.to(tl.int64), bonus_i64)
        tl.store(proposed_ptr + row_base + k, val, mask=active_k)


def accept_dflash_adaptive_verify(
    *,
    draft_token_flat: torch.Tensor,
    target_predict_flat: torch.Tensor,
    actual_verify_lens: torch.Tensor,
    start_offsets: torch.Tensor,
    max_verify_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched greedy DFlash accept length, bonus token, and packed proposed ids (Triton, CUDA only).

    Proposed tokens per request ``i`` are ``draft[start_i+1:start_i+1+accept_len_i]``
    followed by ``bonus_i`` (same rule as the historical Python list ``proposed``).

    Args:
        draft_token_flat: Packed draft token ids, shape ``[total_tokens]``.
        target_predict_flat: Per-position argmax ids aligned with the verify forward, shape ``[total_tokens]``.
        actual_verify_lens: Per-request verify lengths (unpadded), shape ``[bs]``, int32/int64.
        start_offsets: Exclusive start index per request into the flat buffers, shape ``[bs]``, int32/int64.
        max_verify_len: Upper bound for per-request verify length (e.g. draft block size).

    Returns:
        ``(accept_len, bonus, proposed_packed, proposed_count)``:
        ``accept_len`` int32 ``[bs]``, ``bonus`` int64 ``[bs]``,
        ``proposed_packed`` int64 ``[bs, max_v]`` (only first ``proposed_count[i]`` entries are valid),
        ``proposed_count`` int32 ``[bs]`` (equals ``accept_len + 1`` when ``actual_verify_lens[i] > 0``).
    """
    device = draft_token_flat.device
    if device.type != "cuda" or not draft_token_flat.is_cuda:
        raise RuntimeError(
            "accept_dflash_adaptive_verify requires CUDA tensors (Triton kernel)."
        )

    bs = int(actual_verify_lens.shape[0])
    if bs == 0:
        z = torch.empty((0,), dtype=torch.int32, device=device)
        zb = torch.empty((0,), dtype=torch.int64, device=device)
        zp = torch.empty((0, 0), dtype=torch.int64, device=device)
        zc = torch.empty((0,), dtype=torch.int32, device=device)
        return z, zb, zp, zc

    max_v = int(max_verify_len)
    if max_v <= 0:
        z = torch.zeros((bs,), dtype=torch.int32, device=device)
        zb = torch.zeros((bs,), dtype=torch.int64, device=device)
        zp = torch.zeros((bs, 0), dtype=torch.int64, device=device)
        zc = torch.zeros((bs,), dtype=torch.int32, device=device)
        return z, zb, zp, zc

    if not target_predict_flat.is_contiguous():
        target_predict_flat = target_predict_flat.contiguous()
    if not draft_token_flat.is_contiguous():
        draft_token_flat = draft_token_flat.contiguous()

    accept_len_t = torch.empty((bs,), dtype=torch.int32, device=device)
    bonus_t = torch.empty((bs,), dtype=torch.int64, device=device)
    proposed_packed = torch.empty((bs, max_v), dtype=torch.int64, device=device)
    proposed_count_t = torch.empty((bs,), dtype=torch.int32, device=device)

    _dflash_adaptive_verify_proposed_kernel[(bs,)](
        draft_token_flat,
        target_predict_flat,
        actual_verify_lens,
        start_offsets,
        accept_len_t,
        bonus_t,
        proposed_packed,
        proposed_count_t,
        stride_proposed_row=max_v,
        MAX_V=max_v,
    )

    return accept_len_t, bonus_t, proposed_packed, proposed_count_t


def prepare_dflash_adaptive_verify(
    *,
    draft_tokens: torch.Tensor,
    positions: Optional[torch.Tensor],
    verify_len_actual: torch.Tensor,
    target_total_tokens: torch.Tensor,
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
    device = verify_len_padded.device
    cap = (block_size_i - verify_len_padded).clamp(min=0)
    target_total_tokens_t = target_total_tokens.to(device=device, dtype=torch.int64)
    deficit = target_total_tokens_t - verify_len_padded.to(torch.int64).sum()
    max_add = cap.to(torch.int64).sum()
    add_total = torch.minimum(deficit, max_add).clamp(min=0)
    round_ids = torch.arange(block_size_i, device=device, dtype=torch.int64)[:, None]
    # Equivalent to historical per-round `can_take = verify_len_padded < block_size_i`:
    # request j can take one token in round r iff r < cap[j].
    can_take_slots = round_ids < cap.to(torch.int64)[None, :]
    # Preserve historical ordering: round-robin by round, then by request index.
    slots = can_take_slots.flatten()
    picked = slots & (slots.to(torch.int64).cumsum(dim=0) <= add_total)
    extras = picked.view(block_size_i, bs).sum(dim=0).to(verify_len_padded.dtype)
    verify_len_padded.add_(extras)

    lens_i64 = verify_len_padded.to(torch.int64)
    cum = lens_i64.cumsum(dim=0)
    starts = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=verify_len_padded.device),
            cum[:-1],
        ]
    ).to(dtype=start_offsets.dtype)
    start_offsets.copy_(starts)

    max_a = block_size_i
    if max_a > 0:
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
