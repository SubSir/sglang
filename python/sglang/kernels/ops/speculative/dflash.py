import torch
import triton
import triton.language as tl


@triton.jit
def _dflash_accept_bonus_contig_kernel(
    candidates_ptr,
    target_top1_ptr,
    accept_lens_out_ptr,
    commit_lens_out_ptr,
    bonus_ids_out_ptr,
    out_tokens_ptr,
    prefix_lens_ptr,
    new_seq_lens_out_ptr,
    candidates_row_stride,
    target_row_stride,
    accept_stride,
    commit_stride,
    bonus_stride,
    out_tokens_row_stride,
    prefix_lens_stride,
    new_seq_lens_stride,
    block_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size
    draft_mask = cols < (block_size - 1)

    candidate_row_ptr = candidates_ptr + row * candidates_row_stride
    target_row_ptr = target_top1_ptr + row * target_row_stride
    candidate_tail = tl.load(candidate_row_ptr + cols + 1, mask=draft_mask, other=0)

    accept_len = tl.full((), 0, tl.int32)
    prefix_live = tl.full((), 1, tl.int32)
    for col in range(BLOCK_SIZE - 1):
        in_range = col < (block_size - 1)
        candidate_id = tl.load(candidate_row_ptr + (col + 1), mask=in_range, other=0)
        target_id = tl.load(target_row_ptr + col, mask=in_range, other=0)
        match_i32 = (candidate_id == target_id).to(tl.int32)
        keep = in_range & (prefix_live != 0) & (match_i32 != 0)
        accept_len += keep.to(tl.int32)
        prefix_live = tl.where(in_range, prefix_live & match_i32, prefix_live)

    commit_len = accept_len + 1
    bonus_id = tl.load(target_row_ptr + accept_len.to(tl.int64))
    new_seq_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride) + commit_len

    tl.store(accept_lens_out_ptr + row * accept_stride, accept_len)
    tl.store(commit_lens_out_ptr + row * commit_stride, commit_len)
    tl.store(bonus_ids_out_ptr + row * bonus_stride, bonus_id)
    tl.store(new_seq_lens_out_ptr + row * new_seq_lens_stride, new_seq_len)

    out_val = tl.where(draft_mask, candidate_tail, 0)
    out_val = tl.where(cols == accept_len, bonus_id, out_val)
    tl.store(
        out_tokens_ptr + row * out_tokens_row_stride + cols, out_val, mask=row_mask
    )


def _pick_num_warps(block_size: int) -> int:
    if block_size <= 16:
        return 1
    if block_size <= 32:
        return 2
    if block_size <= 64:
        return 4
    return 8


def _is_row_major_contiguous_2d(x: torch.Tensor) -> bool:
    return x.ndim == 2 and x.is_contiguous()


def _compute_dflash_accept_bonus_triton_unchecked(
    candidates: torch.Tensor,
    target_top1: torch.Tensor,
    accept_lens_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    bonus_ids_out: torch.Tensor,
    out_tokens_out: torch.Tensor,
    prefix_lens: torch.Tensor,
    new_seq_lens_out: torch.Tensor,
) -> None:
    batch_size, block_size = candidates.shape
    if batch_size == 0:
        return

    if not _is_row_major_contiguous_2d(candidates):
        raise ValueError("DFLASH Triton accept_bonus requires contiguous candidates.")
    if not _is_row_major_contiguous_2d(target_top1):
        raise ValueError("DFLASH Triton accept_bonus requires contiguous target_top1.")
    if not _is_row_major_contiguous_2d(out_tokens_out):
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous out_tokens_out."
        )
    if not accept_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous accept_lens_out."
        )
    if not commit_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous commit_lens_out."
        )
    if not bonus_ids_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous bonus_ids_out."
        )
    if prefix_lens.ndim != 1:
        raise ValueError("DFLASH Triton accept_bonus requires 1D prefix_lens.")
    if not new_seq_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous new_seq_lens_out."
        )

    block = triton.next_power_of_2(block_size)
    num_warps = _pick_num_warps(block)
    _dflash_accept_bonus_contig_kernel[(batch_size,)](
        candidates,
        target_top1,
        accept_lens_out,
        commit_lens_out,
        bonus_ids_out,
        out_tokens_out,
        prefix_lens,
        new_seq_lens_out,
        candidates.stride(0),
        target_top1.stride(0),
        accept_lens_out.stride(0),
        commit_lens_out.stride(0),
        bonus_ids_out.stride(0),
        out_tokens_out.stride(0),
        prefix_lens.stride(0),
        new_seq_lens_out.stride(0),
        block_size,
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )


@triton.jit
def _dflash_tree_accept_compact_kernel(
    accept_index_ptr,  # [bs, block] int32, absolute (row*block baked in), -1 = no node
    predicts_ptr,  # [bs*block] int32, target prediction following each verify node
    commit_lens_ptr,  # [bs] int32 (= accept_len + 1)
    prefix_lens_ptr,  # [bs] int (KV len before this verify block)
    # outputs
    accept_local_safe_ptr,  # [bs, block] int64: valid -> block-local idx, else 0
    out_tokens_ptr,  # [bs, block] int64: valid -> predicts[accept], else 0
    bonus_ptr,  # [bs] int64: last accepted node's target prediction
    committed_positions_ptr,  # [bs, block] int64: prefix + col
    new_seq_lens_ptr,  # [bs] (prefix_lens.dtype): prefix + commit
    accept_index_row_stride,
    out_row_stride,
    block_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size

    commit = tl.load(commit_lens_ptr + row).to(tl.int32)
    prefix = tl.load(prefix_lens_ptr + row)
    valid = row_mask & (cols < commit)

    ai_ptr = accept_index_ptr + row * accept_index_row_stride
    accept_abs = tl.load(ai_ptr + cols, mask=row_mask, other=-1).to(tl.int64)
    base = (row * block_size).to(tl.int64)
    accept_local = accept_abs - base

    # predicts[safe_abs]; masked-off cols read predicts[base] (in-range, discarded).
    safe_abs = tl.where(valid, accept_abs, base)
    pred = tl.load(predicts_ptr + safe_abs).to(tl.int64)
    out_tokens = tl.where(valid, pred, 0)
    local_safe = tl.where(valid, accept_local, 0)

    tl.store(out_tokens_ptr + row * out_row_stride + cols, out_tokens, mask=row_mask)
    tl.store(
        accept_local_safe_ptr + row * out_row_stride + cols, local_safe, mask=row_mask
    )
    tl.store(
        committed_positions_ptr + row * out_row_stride + cols,
        prefix.to(tl.int64) + cols,
        mask=row_mask,
    )

    # bonus = out_tokens at the last accepted node (col == commit - 1 == accept_len).
    accept_len = commit - 1
    bonus = tl.load(predicts_ptr + tl.load(ai_ptr + accept_len).to(tl.int64)).to(
        tl.int64
    )
    tl.store(bonus_ptr + row, bonus)
    tl.store(new_seq_lens_ptr + row, prefix + commit)


def dflash_tree_accept_compact(
    accept_index: torch.Tensor,
    predicts: torch.Tensor,
    commit_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
):
    """One launch for the tree-accept compaction (replaces ~14 elementwise ops).

    Returns (accept_local_safe, out_tokens, bonus, committed_positions, new_seq_lens).
    """
    bs, block_size = accept_index.shape
    device = accept_index.device
    accept_local_safe = torch.empty((bs, block_size), dtype=torch.int64, device=device)
    out_tokens = torch.empty((bs, block_size), dtype=torch.int64, device=device)
    bonus = torch.empty((bs,), dtype=torch.int64, device=device)
    committed_positions = torch.empty(
        (bs, block_size), dtype=torch.int64, device=device
    )
    new_seq_lens = torch.empty((bs,), dtype=prefix_lens.dtype, device=device)
    if bs == 0:
        return accept_local_safe, out_tokens, bonus, committed_positions, new_seq_lens

    accept_index = accept_index.contiguous()
    predicts = predicts.contiguous()
    block = triton.next_power_of_2(block_size)
    _dflash_tree_accept_compact_kernel[(bs,)](
        accept_index,
        predicts,
        commit_lens.contiguous(),
        prefix_lens.contiguous(),
        accept_local_safe,
        out_tokens,
        bonus,
        committed_positions,
        new_seq_lens,
        accept_index.stride(0),
        out_tokens.stride(0),
        block_size,
        BLOCK_SIZE=block,
        num_warps=_pick_num_warps(block),
    )
    return accept_local_safe, out_tokens, bonus, committed_positions, new_seq_lens


@triton.jit
def _prepare_dflash_draft_block_contig_kernel(
    bonus_tokens_ptr,
    prefix_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    block_ids_out_ptr,
    positions_out_ptr,
    cache_loc_out_ptr,
    bonus_tokens_stride,
    prefix_lens_stride,
    req_pool_indices_stride,
    req_to_token_row_stride,
    block_ids_row_stride,
    positions_row_stride,
    cache_loc_row_stride,
    req_to_token_width,
    block_size,
    mask_token_id,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size

    prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride)
    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_indices_stride)
    bonus_token = tl.load(bonus_tokens_ptr + row * bonus_tokens_stride)

    logical_pos = prefix_len.to(tl.int64) + cols
    valid = row_mask & (logical_pos < req_to_token_width)
    req_row_ptr = req_to_token_ptr + req_idx * req_to_token_row_stride
    slot_ids = tl.load(req_row_ptr + logical_pos, mask=valid, other=0)

    block_ids = tl.full((BLOCK_SIZE,), mask_token_id, tl.int64)
    block_ids = tl.where(cols == 0, bonus_token.to(tl.int64), block_ids)
    tl.store(
        block_ids_out_ptr + row * block_ids_row_stride + cols, block_ids, mask=row_mask
    )
    tl.store(
        positions_out_ptr + row * positions_row_stride + cols,
        logical_pos,
        mask=row_mask,
    )
    tl.store(
        cache_loc_out_ptr + row * cache_loc_row_stride + cols,
        slot_ids.to(tl.int64),
        mask=row_mask,
    )


def _prepare_dflash_draft_block_unchecked(
    bonus_tokens: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    block_ids_out: torch.Tensor,
    positions_out: torch.Tensor,
    cache_loc_out: torch.Tensor,
    mask_token_id: int,
) -> None:
    batch_size = int(bonus_tokens.numel())
    if batch_size == 0:
        return

    if req_to_token.ndim != 2 or req_to_token.stride(1) != 1:
        raise ValueError("DFLASH Triton prepare_block requires row-major req_to_token.")
    if not _is_row_major_contiguous_2d(block_ids_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous block_ids_out."
        )
    if not _is_row_major_contiguous_2d(positions_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous positions_out."
        )
    if not _is_row_major_contiguous_2d(cache_loc_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous cache_loc_out."
        )

    block_size = int(block_ids_out.shape[1])
    block = triton.next_power_of_2(block_size)
    num_warps = _pick_num_warps(block)
    _prepare_dflash_draft_block_contig_kernel[(batch_size,)](
        bonus_tokens,
        prefix_lens,
        req_pool_indices,
        req_to_token,
        block_ids_out,
        positions_out,
        cache_loc_out,
        bonus_tokens.stride(0),
        prefix_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        block_ids_out.stride(0),
        positions_out.stride(0),
        cache_loc_out.stride(0),
        int(req_to_token.shape[1]),
        block_size,
        int(mask_token_id),
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    uniforms_ptr,
    temperatures_ptr,
    greedy_ptr,
    tokens_ptr,
    q_ptr,
    slots: tl.constexpr,
    top_k: tl.constexpr,
):
    """One program per request: a slot's K scores stay in registers and the walk is a
    loop, so the slot-to-slot dependency costs nothing instead of one kernel each."""
    row = tl.program_id(0)
    offsets = tl.arange(0, top_k)
    temperature = tl.load(temperatures_ptr + row)
    greedy = tl.load(greedy_ptr + row) != 0
    previous = 0
    for slot in range(slots):
        base = (row * slots + slot) * top_k
        scores = tl.load(scores_ptr + (base + previous) * top_k + offsets).to(
            tl.float32
        )
        if greedy:
            best = tl.max(scores, axis=0)
            index = tl.min(tl.where(scores == best, offsets, top_k), axis=0)
            probabilities = tl.where(offsets == index, 1.0, 0.0)
        else:
            scaled = scores / temperature
            exponentials = tl.exp(scaled - tl.max(scaled, axis=0))
            probabilities = exponentials / tl.sum(exponentials, axis=0)
            uniform = tl.load(uniforms_ptr + row * slots + slot)
            index = tl.sum(
                tl.where(uniform >= tl.cumsum(probabilities, axis=0), 1, 0), axis=0
            )
            index = tl.minimum(index, top_k - 1)
        tl.store(q_ptr + base + offsets, probabilities)
        tl.store(tokens_ptr + row * slots + slot, tl.load(candidate_ptr + base + index))
        previous = index


def selector_walk_triton(
    *,
    candidate_ids,
    scores,
    uniforms,
    temperatures,
    greedy_mask,
):
    batch, slots, top_k = candidate_ids.shape
    tokens = torch.empty((batch, slots), dtype=torch.int64, device=scores.device)
    q_rows = torch.empty(
        (batch, slots, top_k), dtype=torch.float32, device=scores.device
    )
    _selector_walk_kernel[(batch,)](
        scores.contiguous(),
        candidate_ids.contiguous(),
        uniforms.contiguous(),
        temperatures.contiguous(),
        greedy_mask.contiguous(),
        tokens,
        q_rows,
        slots=slots,
        top_k=top_k,
        num_warps=1,
    )
    return tokens, q_rows


@triton.jit
def _dflash_tree_move_kv_kernel(
    data_ptrs,  # [num_buffers] uint64: every K and V layer buffer of one pool
    strides,  # [num_buffers] row bytes
    loc2d_ptr,  # [bs, n] int64: the request's verify-window slots
    accept_local_ptr,  # [bs, n] int64: window index of the c-th accepted node
    commit_lens_ptr,  # [bs] int32
    n,
    N_POW2: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    buf = tl.program_id(0)
    req = tl.program_id(1)
    tile = tl.program_id(2)
    stride = tl.load(strides + buf)
    base = tl.cast(tl.load(data_ptrs + buf), tl.pointer_type(tl.uint8))
    cols = tl.arange(0, N_POW2)
    in_row = cols < n
    commit = tl.load(commit_lens_ptr + req)
    local = tl.load(accept_local_ptr + req * n + cols, mask=in_row, other=0)
    tgt = tl.load(loc2d_ptr + req * n + cols, mask=in_row, other=0)
    src = tl.load(loc2d_ptr + req * n + local, mask=in_row, other=0)
    rows = in_row & (cols < commit) & (src != tgt)
    byte_off = tile * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    tl.multiple_of(byte_off, 16)
    m = rows[:, None] & (byte_off < stride)[None, :]
    # A request's slots never overlap another's, and every load of this program
    # lands before its stores, so shifting its own rows in place is safe.
    vals = tl.load(base + src[:, None] * stride + byte_off[None, :], mask=m)
    tl.store(base + tgt[:, None] * stride + byte_off[None, :], vals, mask=m)


def dflash_tree_move_kv(
    pool,
    loc2d: torch.Tensor,
    accept_local: torch.Tensor,
    commit_lens: torch.Tensor,
) -> None:
    """Move each request's accepted tree nodes to the front of its verify window,
    one program per (layer buffer, request, byte tile). The generic
    copy_all_layer_kv_cache_tiled gives one program a whole batch of rows per layer
    so that any overlap stays safe; here overlaps are per request, so requests can
    run in parallel. Rows already in place are skipped."""
    bs, n = loc2d.shape
    if bs == 0:
        return
    cfg = pool._kv_copy_config
    grid = (pool.data_ptrs.numel(), bs, cfg["byte_tiles"])
    _dflash_tree_move_kv_kernel[grid](
        pool.data_ptrs,
        pool.data_strides,
        loc2d.contiguous(),
        accept_local.contiguous(),
        commit_lens.contiguous(),
        n,
        N_POW2=triton.next_power_of_2(n),
        BYTES_PER_TILE=cfg["bytes_per_tile"],
        num_warps=cfg["num_warps"],
    )


@triton.jit
def _selector_tree_expand_kernel(
    scores_ptr,  # [bs, S, K, K] selector edge scores
    cand_ptr,  # [bs, S, K] candidate ids
    out_scores_ptr,  # [bs, K + K*K*(S-1)] path probabilities, EAGLE score-list order
    out_tokens_ptr,  # [bs, K + K*K*(S-1)]
    out_parents_ptr,  # [bs, 1 + K*(S-1)]
    stride_sb,
    stride_ss,
    stride_sp,
    stride_cb,
    stride_cs,
    stride_ob,
    stride_pb,
    S: tl.constexpr,
    K: tl.constexpr,
    KK: tl.constexpr,
):
    """build_selector_tree's beam for topk == K, one program per request: each depth
    softmaxes the K beams' transition rows, scores all K*K children by path
    probability and keeps the top K by K rounds of argmax."""
    b = tl.program_id(0)
    offs_k = tl.arange(0, K)
    offs_kk = tl.arange(0, KK)
    s_base = scores_ptr + b * stride_sb
    c_base = cand_ptr + b * stride_cb

    row = tl.load(s_base + offs_k).to(tl.float32)  # slot 0, predecessor = anchor
    row = tl.exp(row - tl.max(row, axis=0))
    beam_score = row / tl.sum(row, axis=0)
    beam_idx = offs_k
    tl.store(out_scores_ptr + b * stride_ob + offs_k, beam_score)
    tl.store(out_tokens_ptr + b * stride_ob + offs_k, tl.load(c_base + offs_k))
    tl.store(out_parents_ptr + b * stride_pb, -1)
    tl.store(out_parents_ptr + b * stride_pb + 1 + offs_k, offs_k.to(tl.int64))

    for e in tl.static_range(1, S):
        m = tl.load(
            s_base + e * stride_ss + beam_idx[:, None] * stride_sp + offs_k[None, :]
        ).to(tl.float32)
        m = tl.exp(m - tl.max(m, axis=1)[:, None])
        m = m / tl.sum(m, axis=1)[:, None]
        expand = beam_score[:, None] * m  # [K parents, K children]
        base = K + (e - 1) * KK
        flat = offs_k[:, None] * K + offs_k[None, :]
        tl.store(out_scores_ptr + b * stride_ob + base + flat, expand)
        tok = tl.load(c_base + e * stride_cs + offs_k)
        tl.store(
            out_tokens_ptr + b * stride_ob + base + flat,
            tl.broadcast_to(tok[None, :], (K, K)),
        )
        vals = tl.reshape(expand, (KK,))
        remaining = offs_kk < KK
        top_v = tl.zeros((K,), tl.float32)
        top_i = tl.zeros((K,), tl.int32)
        for r in tl.static_range(K):
            masked = tl.where(remaining, vals, -1.0)
            mx = tl.max(masked, axis=0)
            pick = tl.min(tl.where(remaining & (masked == mx), offs_kk, KK), axis=0)
            remaining = remaining & (offs_kk != pick)
            top_v = tl.where(offs_k == r, mx, top_v)
            top_i = tl.where(offs_k == r, pick, top_i)
        if e < S - 1:
            tl.store(
                out_parents_ptr + b * stride_pb + (K + 1) + (e - 1) * K + offs_k,
                (top_i + base).to(tl.int64),
            )
        beam_score = top_v
        beam_idx = top_i % K


def selector_tree_expand(candidate_ids: torch.Tensor, scores: torch.Tensor):
    """Fused build_selector_tree beam for topk == K (a power of 2).
    Returns (score_list [bs, K+K*K*(S-1)], token_list, parent_list [bs, 1+K*(S-1)])."""
    bs, num_slots, k = candidate_ids.shape
    width = k + k * k * (num_slots - 1)
    device = candidate_ids.device
    out_scores = torch.empty((bs, width), dtype=torch.float32, device=device)
    out_tokens = torch.empty((bs, width), dtype=candidate_ids.dtype, device=device)
    out_parents = torch.empty(
        (bs, 1 + k * (num_slots - 1)), dtype=torch.int64, device=device
    )
    if bs == 0:
        return out_scores, out_tokens, out_parents
    scores = scores.contiguous()
    candidate_ids = candidate_ids.contiguous()
    _selector_tree_expand_kernel[(bs,)](
        scores,
        candidate_ids,
        out_scores,
        out_tokens,
        out_parents,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        candidate_ids.stride(0),
        candidate_ids.stride(1),
        out_scores.stride(0),
        out_parents.stride(0),
        S=num_slots,
        K=k,
        KK=k * k,
        num_warps=4,
    )
    return out_scores, out_tokens, out_parents
