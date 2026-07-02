# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Lean single-pass paged attention for DFlash tree verify.

Specialized replacement for the general two-stage ``extend_attention_fwd`` on the
DFlash tree-verify forward. It must be numerically lossless vs that oracle run
with ``compact_tree_mask=True`` + ``skip_prefix_custom_mask=True``.

Design (mirrors the oracle's KV reads, one online-softmax accumulation):
  - Grid ``(batch, q_head, cdiv(max_len_extend, BLOCK_M))`` — same as the oracle,
    so GQA is handled by the head grid dim (``cur_kv_head = cur_head // kv_group``)
    and variable N-per-seq by ``qo_indptr`` + ``mask_m``.
  - Prefix region ``[0, context_len)``: K/V pulled from the paged pool
    ``k_buffer``/``v_buffer`` via ``kv_indices``. Always visible (every tree
    token's abs pos >= context_len), no mask.
  - Extend/tree region ``[context_len, context_len+N)``: K/V read directly from
    ``k_extend``/``v_extend`` (contiguous per seq via ``qo_indptr``). The compact
    N*N tree mask gates visibility, indexed exactly like the oracle's
    COMPACT_TREE_MASK branch: row-stride = N (=cur_seq_len_extend).

Deliberately omitted (tree-verify only): sliding window, sinks, logit_cap,
fp8/descale, xai temperature, LSE, RoPE-PE split (BLOCK_DPE).
"""

import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.extend_attention import (
    _get_block_sizes_for_extend_attention,
)


@triton.jit
def _dflash_tree_verify_attn_kernel(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_ptr,
    mask_indptr,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    offs_q = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # Stage 1: prefix KV from the paged pool via kv_indices. Always visible.
    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix
        final_mask = mask_m[:, None] & mask_n[None, :]

        offs_kv_loc = tl.load(
            kv_indices + cur_seq_kv_start_idx + start_n + offs_n, mask=mask_n, other=0
        )
        offs_buf_k = (
            offs_kv_loc[None, :] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d[:, None]
        )
        k = tl.load(
            K_Buffer + offs_buf_k,
            mask=(mask_n[None, :]) & (mask_d[:, None]),
            other=0.0,
        )
        qk = tl.dot(q.to(k.dtype), k)
        qk *= sm_scale
        qk = tl.where(final_mask, qk, float("-inf"))

        row_max = tl.max(qk, 1)
        row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
        n_e_max = tl.maximum(row_max_fixed, e_max)

        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_buf_v = (
            offs_kv_loc[:, None] * stride_buf_vbs
            + cur_kv_head * stride_buf_vh
            + offs_dv[None, :]
        )
        v = tl.load(
            V_Buffer + offs_buf_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
        )
        acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
        e_max = n_e_max

    # Stage 2: extend/tree KV read directly; compact N*N tree mask gates it.
    for start_n in range(0, cur_seq_len_extend, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_extend

        custom_mask = tl.load(
            mask_ptr
            + cur_seq_mask_start_idx
            + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len_extend
            + start_n
            + offs_n[None, :],
            mask=(mask_m[:, None] & mask_n[None, :]),
            other=0,
        )
        final_mask = mask_m[:, None] & mask_n[None, :] & custom_mask

        SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0
        if not SKIP_TILE:
            offs_k = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
            )
            qk = tl.dot(q, k, out_dtype=tl.float32)
            qk *= sm_scale
            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            offs_v = (
                (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
                + cur_kv_head * stride_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
            )
            acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
            e_max = n_e_max

    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O_Extend + offs_o,
        acc / deno[:, None],
        mask=mask_m[:, None] & mask_dv[None, :],
    )


def dflash_tree_verify_attn_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    tree_mask,
    mask_indptr,
    max_len_extend,
    sm_scale=None,
):
    """Lean single-pass tree-verify attention. Lossless vs extend_attention_fwd
    with compact_tree_mask=True + skip_prefix_custom_mask=True.

    ``tree_mask`` is the compact N^2-per-seq boolean mask (row-stride N), the same
    buffer the oracle's COMPACT_TREE_MASK path reads. ``mask_indptr`` = cumsum(N^2).
    """
    Lq, Lv = q_extend.shape[-1], v_extend.shape[-1]

    BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = (
        _get_block_sizes_for_extend_attention(Lq, Lv)
    )
    assert BLOCK_DPE == 0, "dflash_tree_verify_attn_fwd: RoPE-PE split head dims unsupported"

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))

    _dflash_tree_verify_attn_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        tree_mask,
        mask_indptr,
        sm_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        Lq=Lq,
        Lv=Lv,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=1,
    )
