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


@triton.jit
def _desc_order(vals, K: tl.constexpr):
    """Per last-dim row: candidate indices by descending value (ties: lower index
    first) and the sorted values. The float's order-preserving bits and the index
    share one int64 key, so one tl.sort carries both."""
    u = vals.to(tl.int32, bitcast=True)
    u = tl.where(u < 0, u ^ 0x7FFFFFFF, u)
    idx = K - 1 - tl.arange(0, K)
    key = (u.to(tl.int64) << 32) | idx.to(tl.int64)
    key = tl.sort(key, descending=True)
    hi = (key >> 32).to(tl.int32)
    hi = tl.where(hi < 0, hi ^ 0x7FFFFFFF, hi)
    return (K - 1 - (key & 0xFFFF)).to(tl.int32), hi.to(tl.float32, bitcast=True)


@triton.jit
def _gumbel_order(scores, unif, temperature, greedy, K: tl.constexpr):
    """Draw order without replacement from softmax(scores / T): Gumbel-top-K.
    Greedy rows take no noise, so the order is the argmax order."""
    logits = scores / temperature
    logq = logits - tl.max(logits, axis=-1, keep_dims=True)
    logq = logq - tl.log(tl.sum(tl.exp(logq), axis=-1, keep_dims=True))
    u = tl.minimum(tl.maximum(unif, 1e-10), 1.0 - 1e-7)
    gumbel = tl.where(greedy, 0.0, -tl.log(-tl.log(u)))
    order, _ = _desc_order(logq + gumbel, K)
    return order, logq


@triton.jit
def _selector_tree_expand_sampled_kernel(
    scores_ptr,  # [bs, S, K, K] contiguous
    cand_ptr,  # [bs, S, K] contiguous
    unif_ptr,  # [bs, S, K, K] contiguous
    temp_ptr,  # [bs] f32
    greedy_ptr,  # [bs] uint8
    out_scores_ptr,  # [bs, K + K*K*(S-1)]
    out_tokens_ptr,  # [bs, K + K*K*(S-1)]
    out_rows_ptr,  # [bs, K + K*K*(S-1)] int64: the entry's sampled candidate index
    out_parents_ptr,  # [bs, 1 + K*(S-1)]
    S: tl.constexpr,
    K: tl.constexpr,
    KK: tl.constexpr,
):
    """build_selector_tree_sampled for topk == K, one program per request. Entries
    of a beam are its children by rank: the shape keeps the unperturbed (T=1) rank
    order and path score, the token is the rank-th Gumbel draw from the beam's
    own sampled row."""
    b = tl.program_id(0).to(tl.int64)
    offs_k = tl.arange(0, K)
    offs_kk = tl.arange(0, KK)
    width = K + KK * (S - 1)
    s_base = scores_ptr + b * S * KK
    u_base = unif_ptr + b * S * KK
    c_base = cand_ptr + b * S * K
    o_base = b * width
    temperature = tl.load(temp_ptr + b)
    greedy = tl.load(greedy_ptr + b) != 0

    row = tl.load(s_base + offs_k).to(tl.float32)  # slot 0, predecessor = anchor
    shape_idx, shape_val = _desc_order(row, K)
    shape_p = tl.exp(shape_val - tl.max(row, axis=0))
    shape_p = shape_p / tl.sum(tl.exp(row - tl.max(row, axis=0)), axis=0)
    order, logq0 = _gumbel_order(row, tl.load(u_base + offs_k), temperature, greedy, K)
    beam_score = shape_p
    beam_u = shape_idx
    beam_s = order
    tl.store(out_scores_ptr + o_base + offs_k, beam_score)
    tl.store(out_tokens_ptr + o_base + offs_k, tl.load(c_base + order))
    tl.store(out_rows_ptr + o_base + offs_k, order.to(tl.int64))
    tl.store(out_parents_ptr + b * (1 + K * (S - 1)), -1)
    tl.store(out_parents_ptr + b * (1 + K * (S - 1)) + 1 + offs_k, offs_k.to(tl.int64))

    # A runtime loop: unrolling S depths of sorts stalls the compiler for minutes.
    for e in tl.range(1, S):
        m = tl.load(s_base + e * KK + beam_u[:, None] * K + offs_k[None, :]).to(
            tl.float32
        )
        child_idx, child_val = _desc_order(m, K)
        mx = tl.max(m, axis=1)[:, None]
        child_p = tl.exp(child_val - mx) / tl.sum(tl.exp(m - mx), axis=1)[:, None]
        sm = tl.load(s_base + e * KK + beam_s[:, None] * K + offs_k[None, :]).to(
            tl.float32
        )
        su = tl.load(u_base + e * KK + beam_s[:, None] * K + offs_k[None, :])
        child_order, logq_e = _gumbel_order(sm, su, temperature, greedy, K)
        expand = beam_score[:, None] * child_p  # [K beams, K ranks]
        base = K + (e - 1) * KK
        flat = offs_k[:, None] * K + offs_k[None, :]
        tl.store(out_scores_ptr + o_base + base + flat, expand)
        tl.store(
            out_tokens_ptr + o_base + base + flat,
            tl.load(c_base + e * K + child_order),
        )
        tl.store(out_rows_ptr + o_base + base + flat, child_order.to(tl.int64))
        # Next beams: the top K of the K*K entries (ties: lower index), row 0 of
        # the sorted order viewed as [K, K].
        order_kk, val_kk = _desc_order(tl.reshape(expand, (KK,)), KK)
        first = (offs_kk // K == 0)[:, None]
        top_i = tl.sum(
            tl.where(
                first & (offs_kk[:, None] % K == offs_k[None, :]), order_kk[:, None], 0
            ),
            axis=0,
        )
        top_v = tl.sum(
            tl.where(
                first & (offs_kk[:, None] % K == offs_k[None, :]), val_kk[:, None], 0.0
            ),
            axis=0,
        )
        pick = offs_kk[:, None] == top_i[None, :]  # [KK, K]
        new_u = tl.sum(tl.where(pick, tl.reshape(child_idx, (KK,))[:, None], 0), axis=0)
        new_s = tl.sum(
            tl.where(pick, tl.reshape(child_order, (KK,))[:, None], 0), axis=0
        )
        if e < S - 1:
            tl.store(
                out_parents_ptr
                + b * (1 + K * (S - 1))
                + (K + 1)
                + (e - 1) * K
                + offs_k,
                (top_i + base).to(tl.int64),
            )
        beam_score = top_v
        beam_u = new_u
        beam_s = new_s


@triton.jit
def _selector_tree_node_rows_kernel(
    scores_ptr,  # [bs, S, K, K] contiguous
    cand_ptr,  # [bs, S, K] contiguous
    unif_ptr,  # [bs, S, K, K] contiguous
    temp_ptr,  # [bs] f32
    greedy_ptr,  # [bs] uint8
    node_row_ptr,  # [bs, N] int64: the node's own candidate index at its slot
    child_slot_ptr,  # [bs, N] int64: the slot its children are drawn from
    node_cand_ptr,  # [bs, N, K] int64 out
    node_q_ptr,  # [bs, N, K] f32 out
    has_row_ptr,  # [bs, N] uint8 out
    N,
    S: tl.constexpr,
    K: tl.constexpr,
):
    """Each node's ranked child draws (the same Gumbel order the expand used) and
    their draft probabilities, for the RRS verify."""
    pid = tl.program_id(0).to(tl.int64)
    b = pid // N
    offs_k = tl.arange(0, K)
    slot = tl.load(child_slot_ptr + pid)
    has_row = slot < S
    slot = tl.minimum(slot, S - 1)
    r = tl.load(node_row_ptr + pid)
    off = ((b * S + slot) * K + r) * K
    sc = tl.load(scores_ptr + off + offs_k).to(tl.float32)
    order, logq = _gumbel_order(
        sc,
        tl.load(unif_ptr + off + offs_k),
        tl.load(temp_ptr + b),
        tl.load(greedy_ptr + b) != 0,
        K,
    )
    q = tl.exp(logq)
    q_rank = tl.sum(
        tl.where(order[:, None] == offs_k[None, :], q[None, :], 0.0), axis=1
    )
    tl.store(
        node_cand_ptr + pid * K + offs_k, tl.load(cand_ptr + (b * S + slot) * K + order)
    )
    tl.store(node_q_ptr + pid * K + offs_k, tl.where(has_row, q_rank, 0.0))
    tl.store(has_row_ptr + pid, has_row.to(tl.uint8))


def selector_tree_sampled(
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    anchor_ids: torch.Tensor,
    num_verify_tokens: int,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    uniforms: torch.Tensor,
):
    """Fused build_selector_tree_sampled for topk == K (a power of 2). Returns
    draft_tokens, parent_list, selected_index, node_cand, node_q, node_has_row."""
    bs, num_slots, k = candidate_ids.shape
    n = int(num_verify_tokens)
    width = k + k * k * (num_slots - 1)
    device = candidate_ids.device
    out_scores = torch.empty((bs, width), dtype=torch.float32, device=device)
    out_tokens = torch.empty((bs, width), dtype=candidate_ids.dtype, device=device)
    out_rows = torch.empty((bs, width), dtype=torch.int64, device=device)
    parents = torch.empty(
        (bs, 1 + k * (num_slots - 1)), dtype=torch.int64, device=device
    )
    scores = scores.contiguous()
    candidate_ids = candidate_ids.contiguous()
    uniforms = uniforms.contiguous()
    temperatures = temperatures.to(torch.float32).contiguous()
    greedy = greedy_mask.contiguous().view(torch.uint8)
    _selector_tree_expand_sampled_kernel[(bs,)](
        scores,
        candidate_ids,
        uniforms,
        temperatures,
        greedy,
        out_scores,
        out_tokens,
        out_rows,
        parents,
        S=num_slots,
        K=k,
        KK=k * k,
        num_warps=4,
    )
    selected = out_scores.topk(n - 1, dim=-1).indices.sort(dim=-1).values
    draft_tokens = torch.cat(
        [anchor_ids[:, None], out_tokens.gather(1, selected)], dim=1
    )
    # Index layout: K depth-1 entries, then K*K per deeper depth.
    depth = torch.where(
        selected < k, 1, 2 + (selected - k).div(k * k, rounding_mode="floor")
    )
    node_row = torch.cat(
        [torch.zeros_like(selected[:, :1]), out_rows.gather(1, selected)], dim=1
    )
    child_slot = torch.cat([torch.zeros_like(depth[:, :1]), depth], dim=1)
    node_cand = torch.empty((bs, n, k), dtype=candidate_ids.dtype, device=device)
    node_q = torch.empty((bs, n, k), dtype=torch.float32, device=device)
    node_has_row = torch.empty((bs, n), dtype=torch.bool, device=device)
    _selector_tree_node_rows_kernel[(bs * n,)](
        scores,
        candidate_ids,
        uniforms,
        temperatures,
        greedy,
        node_row,
        child_slot,
        node_cand,
        node_q,
        node_has_row.view(torch.uint8),
        n,
        S=num_slots,
        K=k,
        num_warps=1,
    )
    return draft_tokens, parents, selected, node_cand, node_q, node_has_row


@triton.jit
def _dflash_tree_rrs_kernel(
    next_token_ptr,  # [bs, N] int64, local first child (-1 none)
    next_sibling_ptr,  # [bs, N] int64, local next sibling (-1 none)
    tokens_ptr,  # [bs, N] int64
    cand_ptr,  # [bs, N, K] int64: node u's ranked child candidates
    has_row_ptr,  # [bs, N] uint8: node u has a candidate row
    q_ptr,  # [bs, N, K] f32: draft prob of the ranked candidates
    det_ptr,  # [bs] uint8: 1 = deterministic draft (greedy row)
    coins_ptr,  # [bs, N + 1] f32: one per child attempt, the last for the bonus
    probs_ptr,  # [bs, N, V] f32: filtered target distribution per node
    predicts_ptr,  # [bs * N] int32
    accept_index_ptr,  # [bs, N] int32
    accept_len_ptr,  # [bs] int32
    V,
    N: tl.constexpr,
    N_POW2: tl.constexpr,
    K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    ranks = tl.arange(0, K)
    det = tl.load(det_ptr + b) != 0
    base = b * N
    slots = tl.arange(0, N_POW2)
    tl.store(
        accept_index_ptr + base + slots,
        tl.where(slots == 0, base, -1).to(tl.int32),
        mask=slots < N,
    )
    u = tl.zeros((), tl.int64)
    na = tl.zeros((), tl.int32)
    moving = tl.full((), 1, tl.int32)
    R = tl.zeros([K], tl.float32)
    nm = tl.full((), 1.0, tl.float32)
    mnc = tl.full((), 1.0, tl.float32)
    cands = tl.zeros([K], tl.int64)
    has_row = tl.zeros((), tl.int1)
    while moving != 0:
        row = base + u
        has_row = tl.load(has_row_ptr + row) != 0
        cands = tl.load(cand_ptr + row * K + ranks)
        R = tl.where(has_row, tl.load(probs_ptr + row * V + cands), 0.0)
        mnc = tl.maximum(1.0 - tl.sum(R), 0.0)
        qv = tl.load(q_ptr + row * K + ranks)
        nm = tl.full((), 1.0, tl.float32)
        qused = tl.zeros((), tl.float32)
        child = tl.load(next_token_ptr + row)
        i = tl.zeros((), tl.int32)
        acc = tl.full((), -1, tl.int64)
        while (child != -1) & (acc == -1):
            s = tl.maximum(tl.sum(R) + nm * mnc, 1e-20)
            q_rem = tl.where(ranks >= i, qv / tl.maximum(1.0 - qused, 1e-20), 0.0)
            qi = tl.where(det, tl.where(ranks == i, 1.0, 0.0), q_rem)
            xq = tl.sum(tl.where(ranks == i, qi, 0.0))
            xp = tl.sum(tl.where(ranks == i, R, 0.0)) / s
            coin = tl.load(coins_ptr + b * (N + 1) + child)
            if (xq > 0.0) & (coin * xq < xp):
                acc = child
            else:
                R = tl.maximum(R / s - qi, 0.0)
                nm = nm / s
                qused += tl.sum(tl.where(ranks == i, qv, 0.0))
                child = tl.load(next_sibling_ptr + base + child)
                i += 1
        if acc == -1:
            moving = tl.zeros((), tl.int32)
        else:
            tok = tl.load(tokens_ptr + base + acc)
            tl.store(predicts_ptr + base + u, tok.to(tl.int32))
            na += 1
            tl.store(accept_index_ptr + base + na, (base + acc).to(tl.int32))
            u = acc
    tl.store(accept_len_ptr + b, na)

    # Bonus from the residual at u: candidates carry R, every other token
    # p * nm. Candidate draws never touch the vocabulary.
    row = base + u
    sum_r = tl.sum(R)
    t = tl.load(coins_ptr + b * (N + 1) + N) * tl.maximum(sum_r + nm * mnc, 1e-20)
    bonus = tl.full((), 0, tl.int64)
    if t < sum_r:
        pick = tl.minimum(tl.sum(tl.where(tl.cumsum(R, axis=0) <= t, 1, 0)), K - 1)
        bonus = tl.sum(tl.where(ranks == pick, cands, 0))
    else:
        t = t - sum_r
        p_c = tl.where(has_row, tl.load(probs_ptr + row * V + cands), 0.0)
        # mnc assumes the row sums to 1; if rounding leaves t past the scanned
        # mass, take the last token with mass rather than V - 1.
        bonus = tl.full((), -1, tl.int64)
        cum = tl.zeros((), tl.float32)
        found = tl.zeros((), tl.int32)
        for v0 in range(0, V, BLOCK_V):
            if found == 0:
                offs = v0 + tl.arange(0, BLOCK_V)
                pv = tl.load(probs_ptr + row * V + offs, mask=offs < V, other=0.0)
                in_blk = has_row & (cands >= v0) & (cands < v0 + BLOCK_V)
                blk = (tl.sum(pv) - tl.sum(tl.where(in_blk, p_c, 0.0))) * nm
                bonus = tl.maximum(
                    bonus, tl.max(tl.where(pv > 0.0, offs, -1)).to(tl.int64)
                )
                if cum + blk > t:
                    is_c = tl.sum(
                        tl.where(
                            (offs[:, None] == cands[None, :]) & in_blk[None, :], 1, 0
                        ),
                        axis=1,
                    )
                    val = tl.where(is_c > 0, 0.0, pv * nm)
                    hit = (cum + tl.cumsum(val, axis=0)) > t
                    bonus = (v0 + tl.argmax(hit.to(tl.int32), axis=0)).to(tl.int64)
                    found = 1
                cum += blk
    tl.store(predicts_ptr + row, bonus.to(tl.int32))


def dflash_tree_rrs(
    *,
    next_token: torch.Tensor,
    next_sibling: torch.Tensor,
    tokens: torch.Tensor,
    cand: torch.Tensor,
    has_row: torch.Tensor,
    q: torch.Tensor,
    det: torch.Tensor,
    target_probs: torch.Tensor,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_len: torch.Tensor,
) -> None:
    """Recursive rejection sampling without replacement down a draft tree, bonus
    included.

    A node's children are its ranked candidates' first m draws (without
    replacement from q); each is accepted with min(1, p'(x) / q'(x)) under the
    running target residual p' and draft remainder q', else p' <- max(p' - q', 0).
    Deterministic rows use q' = one-hot at the tried child. The last accepted
    node's predict is drawn from its final residual.
    """
    bs, n = tokens.shape
    if bs == 0:
        return
    k = cand.shape[-1]
    vocab = target_probs.shape[-1]
    _dflash_tree_rrs_kernel[(bs,)](
        next_token.contiguous(),
        next_sibling.contiguous(),
        tokens.contiguous(),
        cand.contiguous(),
        has_row.contiguous().view(torch.uint8),
        q.contiguous(),
        det.contiguous().view(torch.uint8),
        torch.rand((bs, n + 1), dtype=torch.float32, device=tokens.device),
        target_probs.contiguous(),
        predicts,
        accept_index,
        accept_len,
        vocab,
        N=n,
        N_POW2=triton.next_power_of_2(n),
        K=k,
        BLOCK_V=4096,
        num_warps=8,
    )
