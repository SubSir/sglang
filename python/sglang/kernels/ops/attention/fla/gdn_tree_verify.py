# Adapted from trymirai/sglang (hikettei/ddtree): python/sglang/srt/layers/attention/
# fla/gdn_tree_triton.py, gdn_tree_fused.py and chunk_tree_verify.py.
"""Tree GDN target verify without per-token state caching.

Verify (four launches per layer, shape-static): K0 prefix = P @ g over the ancestor
matrix P, K1 the masked Grams and diagonal-block inverses, K2 a block forward solve
of (I + A) U = beta (V - exp(prefix) K H0^T), K3 O = exp(prefix) Q H0^T + QKD @ U.
A is nonzero only between a node and its proper ancestors, so a node's output equals
the chain recurrence over its root->node path; no intermediate state is written.

Commit: the accepted path is replayed from the per-layer (k, v, g, beta) stash into
the pooled state, one state read and one write per request per layer.
"""

import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl


class TreeStructure(NamedTuple):
    parent: torch.Tensor  # [N, T] int64, root = -1
    anc_mask: torch.Tensor  # [N, T, T] bool, inclusive ancestors
    anc_f: torch.Tensor
    anc_u8: torch.Tensor  # uint8, what the kernels read
    logmask_incl: torch.Tensor
    logmask_strict: torch.Tensor
    max_depth: int


def alloc_tree_structure_buffers(N: int, T: int, max_depth: int, device) -> TreeStructure:
    assert 0 <= max_depth < T
    z = lambda dt: torch.zeros(N, T, T, dtype=dt, device=device)  # noqa: E731
    return TreeStructure(
        parent=torch.zeros(N, T, dtype=torch.long, device=device),
        anc_mask=z(torch.bool),
        anc_f=z(torch.float32),
        anc_u8=z(torch.uint8),
        logmask_incl=z(torch.float32),
        logmask_strict=z(torch.float32),
        max_depth=max_depth,
    )


@triton.jit
def _tree_struct_walk_kernel(
    parent_ptr, anc_b_ptr, anc_f_ptr, anc_u8_ptr, lm_incl_ptr, lm_strict_ptr,
    stride_pn, stride_an, stride_ai,
    T: tl.constexpr, NP2: tl.constexpr,
):
    """One program per (row, node): walk the parent chain (root's entry is ignored)
    and write the inclusive-ancestor row plus both log-masks."""
    i_n = tl.program_id(0)
    i = tl.program_id(1)
    offs = tl.arange(0, NP2)
    m = offs < T
    NEG_INF = float("-inf")
    anc_row = tl.zeros((NP2,), tl.int1)
    cur = i
    it = 0
    while cur >= 0 and it <= T:
        anc_row = anc_row | (offs == cur)
        nxt = tl.load(parent_ptr + i_n * stride_pn + cur).to(tl.int32)
        cur = tl.where(cur == 0, -1, nxt)
        it += 1
    base = i_n * stride_an + i * stride_ai
    tl.store(anc_b_ptr + base + offs, anc_row.to(tl.int8), mask=m)
    tl.store(anc_f_ptr + base + offs, anc_row.to(tl.float32), mask=m)
    tl.store(anc_u8_ptr + base + offs, anc_row.to(tl.int8), mask=m)
    tl.store(lm_incl_ptr + base + offs, tl.where(anc_row, 0.0, NEG_INF), mask=m)
    strict = tl.where(anc_row & (offs != i), 0.0, NEG_INF)
    tl.store(lm_strict_ptr + base + offs, strict, mask=m)


def build_tree_structure_into(parent_src: torch.Tensor, buf: TreeStructure) -> TreeStructure:
    """Rebuild the first parent_src.shape[0] rows of `buf` in place (capture safe)
    and return views over them."""
    bs, T = parent_src.shape
    rows = TreeStructure(*(t[:bs] for t in buf[:-1]), buf.max_depth)
    rows.parent.copy_(parent_src)
    _tree_struct_walk_kernel[(bs, T)](
        rows.parent, rows.anc_mask, rows.anc_f, rows.anc_u8,
        rows.logmask_incl, rows.logmask_strict,
        rows.parent.stride(0), rows.anc_mask.stride(0), rows.anc_mask.stride(1),
        T=T, NP2=triton.next_power_of_2(T),
    )
    return rows


BT = 32  # tile/block size (rows of A per program, solve block size)

# Scratch reused across layers/calls (A/QKD/Ainv/beta are consumed within the
# call; safe on a single stream, and under CUDA graphs it avoids pinning one
# copy per captured layer). U and prefix are NOT here: the returned lazy
# state aliases them, so they must stay per-call.
_WORKSPACE: dict = {}
_TRI_CACHE: dict = {}


def _ws(key, alloc):
    buf = _WORKSPACE.get(key)
    if buf is None:
        buf = alloc()
        _WORKSPACE[key] = buf
    return buf


def _tri_tiles(NB: int, device) -> tuple[torch.Tensor, int]:
    """Flat [(i_t, i_j)] lookup for the lower-triangular K1 tile grid."""
    key = (NB, device)
    hit = _TRI_CACHE.get(key)
    if hit is None:
        pairs = [(it, jt) for it in range(NB) for jt in range(it + 1)]
        t = torch.tensor(pairs, dtype=torch.int32, device=device).flatten().contiguous()
        hit = (t, len(pairs))
        _TRI_CACHE[key] = hit
    return hit


@triton.jit
def _gate(a, A_log, dt_bias, sp_beta, sp_thr):
    x = (a + dt_bias) * sp_beta
    sp = tl.where(x <= sp_thr, tl.log(1.0 + tl.exp(x)) / sp_beta, a + dt_bias)
    return -tl.exp(A_log) * sp


@triton.jit
def _dot(a, b, precision: tl.constexpr, bf16_operands: tl.constexpr):
    if bf16_operands:
        return tl.dot(
            a.to(tl.bfloat16),
            b.to(tl.bfloat16),
            out_dtype=tl.float32,
        )
    return tl.dot(a, b, input_precision=precision)


@triton.jit
def _k0_scalars_kernel(
    a_ptr, b_ptr, A_log_ptr, dt_bias_ptr, p_ptr,
    prefix_ptr, beta_ptr,
    T, HV: tl.constexpr, sp_beta, sp_thr,
    stride_a_n, stride_a_t,
    stride_p_n,
    stride_s_n, stride_s_t,
    NB: tl.constexpr, BT: tl.constexpr, BH: tl.constexpr,
    BF16_K0: tl.constexpr,
):
    # prefix = P @ g for ALL heads at once: one [BT,BT]@[BT,BH] dot per
    # t-chunk (P is per-sequence — loading it per head wastes 48x traffic).
    # tf32x3 dots: tensor-core speed at ~fp32 accuracy (~5e-7 rel) — ieee
    # lowers to scalar FMAs and is 10x slower at this tiny grid.
    i_n, i_rb, i_hc = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = i_rb * BT + tl.arange(0, BT)
    m_r = rows < T
    hv = i_hc * BH + tl.arange(0, BH)
    m_h = hv < HV
    A_log = tl.load(A_log_ptr + hv, mask=m_h, other=0.0).to(tl.float32)
    dtb = tl.load(dt_bias_ptr + hv, mask=m_h, other=0.0).to(tl.float32)

    acc = tl.zeros((BT, BH), dtype=tl.float32)
    for tb in tl.static_range(NB):
        tcols = tb * BT + tl.arange(0, BT)
        m_t = tcols < T
        p_tile = tl.load(
            p_ptr + i_n * stride_p_n + rows[:, None] * T + tcols[None, :],
            mask=m_r[:, None] & m_t[None, :], other=0,
        ).to(tl.float32)
        a_tile = tl.load(
            a_ptr + i_n * stride_a_n + tcols[:, None] * stride_a_t + hv[None, :],
            mask=m_t[:, None] & m_h[None, :], other=0.0,
        ).to(tl.float32)
        g_tile = tl.where(
            m_t[:, None] & m_h[None, :],
            _gate(a_tile, A_log[None, :], dtb[None, :], sp_beta, sp_thr), 0.0,
        )
        acc += _dot(p_tile, g_tile, "tf32x3", BF16_K0)

    b_v = tl.load(
        b_ptr + i_n * stride_a_n + rows[:, None] * stride_a_t + hv[None, :],
        mask=m_r[:, None] & m_h[None, :], other=0.0,
    ).to(tl.float32)
    beta = 1.0 / (1.0 + tl.exp(-b_v))
    m_rh = m_r[:, None] & m_h[None, :]
    tl.store(prefix_ptr + i_n * stride_s_n + rows[:, None] * stride_s_t + hv[None, :], acc, mask=m_rh)
    tl.store(beta_ptr + i_n * stride_s_n + rows[:, None] * stride_s_t + hv[None, :], beta, mask=m_rh)


@triton.jit
def _k1_gram_kernel(
    q_ptr, k_ptr, p_ptr, prefix_ptr, beta_ptr,
    A_ptr, QKD_ptr, Ainv_ptr,
    T, scale,
    H: tl.constexpr, HV: tl.constexpr,
    stride_q_n, stride_q_t, stride_q_h,
    stride_p_n,
    stride_s_n, stride_s_t,
    stride_A_nh, stride_A_i,
    stride_inv_nh, stride_inv_b,
    USE_L2NORM: tl.constexpr,
    NBT: tl.constexpr, BK: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
    PREC: tl.constexpr, G: tl.constexpr, MAX_DEPTH: tl.constexpr,
    BF16_GRAM: tl.constexpr, BF16_INVERSE: tl.constexpr,
    tile_ptr=None,
):
    # lower-triangular tile grid: upper tiles (j > i) are identically zero
    # and never read by K2/K3, so they are not computed or stored.
    # grid is per K-HEAD (not per v-head): the G grouped v-heads share the
    # same KK/QK Grams and differ only in beta/prefix masks — computing the
    # dots once and applying G masks removes a Gx redundancy.
    i_l, i_nh_h = tl.program_id(0), tl.program_id(1)
    i_t = tl.load(tile_ptr + 2 * i_l)
    i_j = tl.load(tile_ptr + 2 * i_l + 1)
    i_n, i_h = i_nh_h // H, i_nh_h % H

    rows = i_t * BT + tl.arange(0, BT)
    cols = i_j * BT + tl.arange(0, BT)
    m_r, m_c = rows < T, cols < T
    offs_k = tl.arange(0, BK)
    m_k = offs_k < K

    k_i = tl.load(
        k_ptr + i_n * stride_q_n + rows[:, None] * stride_q_t + i_h * stride_q_h + offs_k[None, :],
        mask=m_r[:, None] & m_k[None, :], other=0.0,
    ).to(tl.float32)
    k_j = tl.load(
        k_ptr + i_n * stride_q_n + cols[:, None] * stride_q_t + i_h * stride_q_h + offs_k[None, :],
        mask=m_c[:, None] & m_k[None, :], other=0.0,
    ).to(tl.float32)
    q_i = tl.load(
        q_ptr + i_n * stride_q_n + rows[:, None] * stride_q_t + i_h * stride_q_h + offs_k[None, :],
        mask=m_r[:, None] & m_k[None, :], other=0.0,
    ).to(tl.float32)
    if USE_L2NORM:
        k_i = k_i / tl.sqrt(tl.sum(k_i * k_i, 1) + 1e-6)[:, None]
        k_j = k_j / tl.sqrt(tl.sum(k_j * k_j, 1) + 1e-6)[:, None]
        q_i = q_i / tl.sqrt(tl.sum(q_i * q_i, 1) + 1e-6)[:, None]
    q_i = q_i * scale

    kk = _dot(k_i, tl.trans(k_j), PREC, BF16_GRAM)
    qk = _dot(q_i, tl.trans(k_j), PREC, BF16_GRAM)

    p_tile = tl.load(
        p_ptr + i_n * stride_p_n + rows[:, None] * T + cols[None, :],
        mask=m_r[:, None] & m_c[None, :], other=0,
    )
    incl = p_tile > 0
    strict = incl & (rows[:, None] != cols[None, :])
    eye = (tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :]).to(tl.float32)
    ij = tl.arange(0, BT)[:, None] * BT + tl.arange(0, BT)[None, :]

    for g_i in tl.static_range(G):
        i_hv = i_h * G + g_i
        i_nh = i_n * HV + i_hv
        pre_i = tl.load(prefix_ptr + i_n * stride_s_n + rows * stride_s_t + i_hv, mask=m_r, other=0.0)
        pre_j = tl.load(prefix_ptr + i_n * stride_s_n + cols * stride_s_t + i_hv, mask=m_c, other=0.0)
        beta_i = tl.load(beta_ptr + i_n * stride_s_n + rows * stride_s_t + i_hv, mask=m_r, other=0.0)
        dexp = tl.exp(pre_i[:, None] - pre_j[None, :])

        a_tile = tl.where(strict, beta_i[:, None] * dexp * kk, 0.0)
        qkd_tile = tl.where(incl, dexp * qk, 0.0)

        base = i_nh * stride_A_nh + rows[:, None] * stride_A_i + cols[None, :]
        tl.store(A_ptr + base, a_tile)
        tl.store(QKD_ptr + base, qkd_tile)

        if i_t == i_j:
            # (I + A_bb)^-1 = sum_j (-A_bb)^j, A_bb nilpotent (strictly
            # lower) — repeated squaring covers j < 32 = BT
            n1 = -a_tile
            x = eye + n1
            m2 = _dot(n1, n1, PREC, BF16_INVERSE)
            x = x + _dot(x, m2, PREC, BF16_INVERSE)
            if MAX_DEPTH > 3:
                m2 = _dot(m2, m2, PREC, BF16_INVERSE)
                x = x + _dot(x, m2, PREC, BF16_INVERSE)
            if MAX_DEPTH > 7:
                m2 = _dot(m2, m2, PREC, BF16_INVERSE)
                x = x + _dot(x, m2, PREC, BF16_INVERSE)
            if MAX_DEPTH > 15:
                m2 = _dot(m2, m2, PREC, BF16_INVERSE)
                x = x + _dot(x, m2, PREC, BF16_INVERSE)
            if MAX_DEPTH > 31:
                m2 = _dot(m2, m2, PREC, BF16_INVERSE)
                x = x + _dot(x, m2, PREC, BF16_INVERSE)
            inv_base = i_nh * stride_inv_nh + i_t * stride_inv_b
            tl.store(Ainv_ptr + inv_base + ij, x)


@triton.jit
def _k2_solve_kernel(
    k_ptr, v_ptr, prefix_ptr, beta_ptr,
    A_ptr, Ainv_ptr, h0_ptr, h0_idx_ptr,
    U_ptr,
    T, scale,
    H: tl.constexpr, HV: tl.constexpr,
    stride_k_n, stride_k_t, stride_k_h,
    stride_v_n, stride_v_t, stride_v_h,
    stride_s_n, stride_s_t,
    stride_A_nh, stride_A_i,
    stride_inv_nh, stride_inv_b,
    stride_U_nh, stride_U_t,
    USE_L2NORM: tl.constexpr, USE_H0: tl.constexpr,
    NB: tl.constexpr, BK: tl.constexpr, K: tl.constexpr,
    BV: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
    PREC: tl.constexpr,
    BF16_STATE: tl.constexpr, BF16_FORWARD: tl.constexpr,
):
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    offs_k = tl.arange(0, BK)
    offs_v = i_v * BV + tl.arange(0, BV)
    m_k, m_v = offs_k < K, offs_v < V

    h0 = tl.zeros((BK, BV), dtype=tl.float32)
    if USE_H0:
        idx = tl.load(h0_idx_ptr + i_n).to(tl.int64)
        if idx >= 0:
            h0 = tl.load(
                h0_ptr + idx * HV * V * K + i_hv * V * K
                + offs_v[None, :] * K + offs_k[:, None],
                mask=m_k[:, None] & m_v[None, :], other=0.0,
            ).to(tl.float32)

    ij = tl.arange(0, BT)[:, None] * BT + tl.arange(0, BT)[None, :]
    for cur in tl.static_range(NB):
        rows = cur * BT + tl.arange(0, BT)
        m_r = rows < T
        k_b = tl.load(
            k_ptr + i_n * stride_k_n + rows[:, None] * stride_k_t + i_h * stride_k_h + offs_k[None, :],
            mask=m_r[:, None] & m_k[None, :], other=0.0,
        ).to(tl.float32)
        if USE_L2NORM:
            k_b = k_b / tl.sqrt(tl.sum(k_b * k_b, 1) + 1e-6)[:, None]
        v_b = tl.load(
            v_ptr + i_n * stride_v_n + rows[:, None] * stride_v_t + i_hv * stride_v_h + offs_v[None, :],
            mask=m_r[:, None] & m_v[None, :], other=0.0,
        ).to(tl.float32)
        pre_b = tl.load(prefix_ptr + i_n * stride_s_n + rows * stride_s_t + i_hv, mask=m_r, other=0.0)
        beta_b = tl.load(beta_ptr + i_n * stride_s_n + rows * stride_s_t + i_hv, mask=m_r, other=0.0)
        kh0 = _dot(k_b, h0, PREC, BF16_STATE)
        acc = beta_b[:, None] * (v_b - tl.exp(pre_b)[:, None] * kh0)
        for pb in tl.static_range(cur):
            pcols = pb * BT + tl.arange(0, BT)
            a_blk = tl.load(
                A_ptr + i_nh * stride_A_nh + rows[:, None] * stride_A_i + pcols[None, :]
            )
            u_blk = tl.load(
                U_ptr + i_nh * stride_U_nh + pcols[:, None] * stride_U_t + offs_v[None, :]
            )
            acc -= _dot(a_blk, u_blk, PREC, BF16_FORWARD)
        inv = tl.load(Ainv_ptr + i_nh * stride_inv_nh + cur * stride_inv_b + ij)
        u_cur = _dot(inv, acc, PREC, BF16_FORWARD)
        tl.store(
            U_ptr + i_nh * stride_U_nh + rows[:, None] * stride_U_t + offs_v[None, :], u_cur
        )


@triton.jit
def _k3_out_kernel(
    q_ptr, prefix_ptr,
    QKD_ptr, U_ptr, h0_ptr, h0_idx_ptr,
    O_ptr,
    T, scale,
    H: tl.constexpr, HV: tl.constexpr,
    stride_q_n, stride_q_t, stride_q_h,
    stride_s_n, stride_s_t,
    stride_A_nh, stride_A_i,
    stride_U_nh, stride_U_t,
    stride_o_n, stride_o_t, stride_o_h,
    USE_L2NORM: tl.constexpr, USE_H0: tl.constexpr,
    NB: tl.constexpr, BK: tl.constexpr, K: tl.constexpr,
    BV: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
    PREC: tl.constexpr,
    BF16_STATE: tl.constexpr, BF16_READOUT: tl.constexpr,
):
    i_t, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    rows = i_t * BT + tl.arange(0, BT)
    m_r = rows < T
    offs_k = tl.arange(0, BK)
    offs_v = i_v * BV + tl.arange(0, BV)
    m_k, m_v = offs_k < K, offs_v < V

    q_i = tl.load(
        q_ptr + i_n * stride_q_n + rows[:, None] * stride_q_t + i_h * stride_q_h + offs_k[None, :],
        mask=m_r[:, None] & m_k[None, :], other=0.0,
    ).to(tl.float32)
    if USE_L2NORM:
        q_i = q_i / tl.sqrt(tl.sum(q_i * q_i, 1) + 1e-6)[:, None]
    q_i = q_i * scale

    acc = tl.zeros((BT, BV), dtype=tl.float32)
    if USE_H0:
        idx = tl.load(h0_idx_ptr + i_n).to(tl.int64)
        if idx >= 0:
            h0 = tl.load(
                h0_ptr + idx * HV * V * K + i_hv * V * K
                + offs_v[None, :] * K + offs_k[:, None],
                mask=m_k[:, None] & m_v[None, :], other=0.0,
            ).to(tl.float32)
            pre_i = tl.load(
                prefix_ptr + i_n * stride_s_n + rows * stride_s_t + i_hv, mask=m_r, other=0.0
            )
            acc = tl.exp(pre_i)[:, None] * _dot(q_i, h0, PREC, BF16_STATE)

    for jt in range(0, i_t + 1):  # tiles with jt > i_t are zero (no ancestors above)
        cols = jt * BT + tl.arange(0, BT)
        qkd = tl.load(
            QKD_ptr + i_nh * stride_A_nh + rows[:, None] * stride_A_i + cols[None, :]
        )
        u_blk = tl.load(
            U_ptr + i_nh * stride_U_nh + cols[:, None] * stride_U_t + offs_v[None, :]
        )
        acc += _dot(qkd, u_blk, PREC, BF16_READOUT)

    tl.store(
        O_ptr + i_n * stride_o_n + rows[:, None] * stride_o_t + i_hv * stride_o_h + offs_v[None, :],
        acc.to(O_ptr.dtype.element_ty),
        mask=m_r[:, None] & m_v[None, :],
    )


def tree_gdn_triton_verify(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    tree: TreeStructure,
    scale=None,
    use_qk_l2norm_in_kernel: bool = False,
    precision: str = "ieee",
    bf16_mode: str | None = None,
):
    """Same contract as gdn_tree_fused.tree_gdn_fused_verify. precision:
    "ieee" = fp32-exact dots (validation); "tf32" = tensor cores (serving,
    ~2e-4 abs vs ieee — below bf16 input noise)."""
    if bf16_mode is None:
        bf16_mode = os.environ.get("SGLANG_GDN_TREE_BF16_OPERANDS", "none")
    assert initial_state_source.stride(0) == v.shape[2] * v.shape[3] * q.shape[3], (
        "tree GDN verify reads a per-layer contiguous [pool, HV, V, K] state"
    )

    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if scale is None:
        scale = K**-0.5
    NB = triton.cdiv(T, BT)
    NBT = NB * BT
    BK = max(16, triton.next_power_of_2(K))
    BV = min(64, max(16, triton.next_power_of_2(V)))
    BH = 16  # head-chunk width; chunked grid keeps K0 parallel
    NHC = triton.cdiv(HV, BH)
    NV = triton.cdiv(V, BV)
    G = HV // H
    dev = q.device
    assert bf16_mode in {
        "none", "gram", "state", "forward", "readout", "body", "solve", "all"
    }
    bf16_gram = bf16_mode != "none"
    bf16_state = bf16_mode not in {"none", "gram"}
    bf16_forward = bf16_mode in {"forward", "body", "solve", "all"}
    bf16_readout = bf16_mode in {"readout", "body", "solve", "all"}
    bf16_inverse = bf16_mode in {"solve", "all"}
    bf16_k0 = bf16_mode == "all"

    p_u8 = tree.anc_u8  # [N, T, T], built once per verify step
    prefix = torch.empty(B, NBT, HV, device=dev, dtype=torch.float32)
    beta = _ws(("beta", B, NBT, HV, dev), lambda: torch.empty(B, NBT, HV, device=dev, dtype=torch.float32))
    A = _ws(("A", B * HV, NBT, dev), lambda: torch.empty(B * HV, NBT, NBT, device=dev, dtype=torch.float32))
    QKD = _ws(("QKD", B * HV, NBT, dev), lambda: torch.empty(B * HV, NBT, NBT, device=dev, dtype=torch.float32))
    Ainv = _ws(("Ainv", B * HV, NB, dev), lambda: torch.empty(B * HV, NB, BT, BT, device=dev, dtype=torch.float32))
    U = torch.empty(B * HV, NBT, V, device=dev, dtype=torch.float32)
    O = torch.empty(B, T, HV, V, device=dev, dtype=v.dtype)
    tile_t, n_tiles = _tri_tiles(NB, dev)

    _k0_scalars_kernel[(B, NB, NHC)](
        a, b, A_log, dt_bias, p_u8, prefix, beta,
        T, HV, softplus_beta, softplus_threshold,
        a.stride(0), a.stride(1),
        p_u8.stride(0),
        prefix.stride(0), prefix.stride(1),
        NB=NB, BT=BT, BH=BH, BF16_K0=bf16_k0,
        num_warps=4,
    )
    _k1_gram_kernel[(n_tiles, B * H)](
        q, k, p_u8, prefix, beta, A, QKD, Ainv,
        T, scale, H, HV,
        q.stride(0), q.stride(1), q.stride(2),
        p_u8.stride(0),
        prefix.stride(0), prefix.stride(1),
        A.stride(0), A.stride(1),
        Ainv.stride(0), Ainv.stride(1),
        USE_L2NORM=use_qk_l2norm_in_kernel,
        NBT=NBT, BK=BK, K=K, BT=BT, PREC=precision, G=G,
        BF16_GRAM=bf16_gram, BF16_INVERSE=bf16_inverse,
        MAX_DEPTH=tree.max_depth,
        tile_ptr=tile_t,
        num_warps=4,
    )
    _k2_solve_kernel[(B * HV, NV)](
        k, v, prefix, beta, A, Ainv,
        initial_state_source, initial_state_indices, U,
        T, scale, H, HV,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        prefix.stride(0), prefix.stride(1),
        A.stride(0), A.stride(1),
        Ainv.stride(0), Ainv.stride(1),
        U.stride(0), U.stride(1),
        USE_L2NORM=use_qk_l2norm_in_kernel,
        USE_H0=initial_state_source is not None,
        NB=NB, BK=BK, K=K, BV=BV, V=V, BT=BT, PREC=precision,
        BF16_STATE=bf16_state, BF16_FORWARD=bf16_forward,
        num_warps=8,
    )
    _k3_out_kernel[(NB, NV, B * HV)](
        q, prefix, QKD, U,
        initial_state_source, initial_state_indices, O,
        T, scale, H, HV,
        q.stride(0), q.stride(1), q.stride(2),
        prefix.stride(0), prefix.stride(1),
        QKD.stride(0), QKD.stride(1),
        U.stride(0), U.stride(1),
        O.stride(0), O.stride(1), O.stride(2),
        USE_L2NORM=use_qk_l2norm_in_kernel,
        USE_H0=initial_state_source is not None,
        NB=NB, BK=BK, K=K, BV=BV, V=V, BT=BT, PREC=precision,
        BF16_STATE=bf16_state, BF16_READOUT=bf16_readout,
        num_warps=4,
    )

    return O


TREE_CHUNK_SIZE = 64


def tree_mask_capacity(T: int) -> int:
    """Bitset slot capacity for T draft tokens: power of two, at least 64.

    Power-of-two keeps the word count (capacity / 64) a power of two so the
    kernels can tl.arange over words, and keeps the T <= 64 layout
    byte-identical to the original single-word [B, 64] masks.
    """
    cap = TREE_CHUNK_SIZE
    while cap < T:
        cap *= 2
    return cap


@triton.jit(do_not_specialize=["T"])
def tree_ancestor_bitmask_kernel(
    parent_tokens,
    ancestor_masks,
    T,
    stride_parent_seq,
    stride_parent_token,
    BT: tl.constexpr,
    NW: tl.constexpr,
):
    """ancestor_masks[n, i, w] = int64 bitset word w (tokens 64w..64w+63) of
    the proper ancestors of token i.

    parent_tokens follows the retrieve_parent_token convention: tree-local
    parent index per token, token 0 is the root (its parent entry is ignored).
    """
    i_n = tl.program_id(0)
    o_t = tl.arange(0, BT)
    o_w = tl.arange(0, NW)
    m_t = o_t < T

    b_parents = tl.load(
        parent_tokens + i_n * stride_parent_seq + o_t * stride_parent_token,
        mask=m_t,
        other=0,
    ).to(tl.int64)
    # b_bits[i, w] = token i's own bit, one-hot in its word
    b_bit = tl.full([BT], 1, tl.int64) << (o_t % 64).to(tl.int64)
    b_bits = tl.where((o_t // 64)[:, None] == o_w[None, :], b_bit[:, None], 0)
    b_anc = tl.zeros([BT, NW], tl.int64)
    for i in range(1, T):
        parent = tl.sum(tl.where(o_t == i, b_parents, 0))
        anc_parent = tl.sum(tl.where(o_t[:, None] == parent, b_anc, 0), 0)
        bit_parent = tl.sum(tl.where(o_t[:, None] == parent, b_bits, 0), 0)
        b_anc = tl.where(
            o_t[:, None] == i, (anc_parent | bit_parent)[None, :], b_anc
        )
    tl.store(
        ancestor_masks + (i_n * BT + o_t[:, None]) * NW + o_w[None, :],
        b_anc,
        mask=m_t[:, None],
    )


def build_tree_ancestor_masks(
    parent_tokens: torch.Tensor,
    T: int,
    BT: int = None,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Build [B, BT, BT // 64] int64 proper-ancestor bitsets from tree-local
    parent indices. Word w of row i covers ancestors with index in
    [64w, 64w + 64); for BT == 64 the layout is byte-identical to the original
    single-word [B, 64] masks.

    Layer-invariant: build once per verify step and reuse across all GDN layers.
    Pass a preallocated `out` to stay allocation-free under CUDA graph capture.
    """
    B = parent_tokens.shape[0]
    if out is not None:
        ancestor_masks = out[:B]
        assert ancestor_masks.dim() == 3 and ancestor_masks.dtype == torch.int64
        BT, NW = ancestor_masks.shape[-2], ancestor_masks.shape[-1]
    else:
        if BT is None:
            BT = tree_mask_capacity(T)
        NW = BT // 64
        ancestor_masks = torch.zeros(
            B, BT, NW, dtype=torch.int64, device=parent_tokens.device
        )
    assert BT & (BT - 1) == 0 and NW == BT // 64, (BT, NW)
    assert T <= BT, f"tree verify supports at most {BT} draft tokens, got {T}"
    tree_ancestor_bitmask_kernel[(B,)](
        parent_tokens=parent_tokens,
        ancestor_masks=ancestor_masks,
        T=T,
        stride_parent_seq=parent_tokens.stride(0),
        stride_parent_token=parent_tokens.stride(1),
        BT=BT,
        NW=NW,
    )
    return ancestor_masks


@triton.jit(do_not_specialize=["T"])
def tree_verify_state_advance_kernel(
    k_stash,
    v_stash,
    g_stash,
    beta_stash,
    ssm_states,
    cache_indices,
    ancestor_masks,
    last_correct_steps,
    track_steps,
    track_slots,
    T,
    stride_kl,
    stride_vl,
    stride_gl,
    stride_bl,
    stride_sl,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    NW: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_TRACK: tl.constexpr,
):
    """Commit SSM states after verification by replaying the delta rule along
    each request's accept path (root -> last correct draft).

    The accept path is recovered GPU-side from the leaf's proper-ancestor
    bitset words: path = ancestor_masks[leaf] | (1 << leaf), word w covering
    tree slots [64w, 64w + 64). Ancestors of a node are totally ordered and
    topologically sorted, so iterating tree slots in increasing index order
    replays the path root-first.

    Replaces the recurrent verify kernel's per-token intermediate state cache
    (T full states per layer per request) with one state read + one write.

    k_stash holds raw k (l2-normalized here in fp32), beta_stash raw b (sigmoid here);
    g_stash is the gate.
    Stash layout per layer l (leading stride stride_*l):
      k_stash: [B, T, Hg*K], v_stash: [B, T, H*V], g/beta_stash: [B, T, H]
    ssm_states: [L, pool, H, V, K] with per-layer stride stride_sl.
    """
    i_l, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // H, i_nh % H
    i_h = i_hv // (H // Hg)
    # int64: a layer's offset into the whole state pool overflows int32.
    i_l = i_l.to(tl.int64)

    slot = tl.load(cache_indices + i_n).to(tl.int64)
    if slot < 0:
        # Padded request (PAD_SLOT_ID): nothing to commit.
        return

    leaf = tl.load(last_correct_steps + i_n).to(tl.int64)
    leaf = tl.minimum(tl.maximum(leaf, 0), T - 1)
    o_w = tl.arange(0, NW)
    b_anc = tl.load(ancestor_masks + (i_n * BT + leaf) * NW + o_w)
    path_words = b_anc | tl.where(
        o_w == (leaf // 64), tl.full((), 1, tl.int64) << (leaf % 64), 0
    )

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    p_h = (
        ssm_states
        + i_l * stride_sl
        + slot * H * K * V
        + i_hv * K * V
        + o_v[None, :] * K
        + o_k[:, None]
    )
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)

    if HAS_TRACK:
        track_step = tl.load(track_steps + i_n).to(tl.int64)
        track_slot = tl.load(track_slots + i_n).to(tl.int64)
        # Rows without a track slot (-1) snapshot nothing.
        track_step = tl.where(track_slot >= 0, track_step, -1)
    else:
        track_step = -1
        track_slot = 0

    p_k = k_stash + i_l * stride_kl + i_n * T * Hg * K + i_h * K + o_k
    p_v = v_stash + i_l * stride_vl + i_n * T * H * V + i_hv * V + o_v
    p_g = g_stash + i_l * stride_gl + i_n * T * H + i_hv
    p_b = beta_stash + i_l * stride_bl + i_n * T * H + i_hv

    for t in range(0, T):
        word = tl.sum(tl.where(o_w == (t // 64), path_words, 0))
        on_path = ((word >> (t % 64)) & 1) != 0
        if on_path:
            b_k = tl.load(p_k + t * Hg * K, mask=mask_k, other=0).to(tl.float32)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_v = tl.load(p_v + t * H * V, mask=mask_v, other=0).to(tl.float32)
            b_g = tl.load(p_g + t * H).to(tl.float32)
            b_beta = 1.0 / (1.0 + tl.exp(-tl.load(p_b + t * H).to(tl.float32)))

            b_h *= tl.exp(b_g)
            b_v -= tl.sum(b_h * b_k[:, None], 0)
            b_v *= b_beta
            b_h += b_k[:, None] * b_v[None, :]

            if HAS_TRACK:
                if t == track_step:
                    p_track = (
                        ssm_states
                        + i_l * stride_sl
                        + track_slot * H * K * V
                        + i_hv * K * V
                        + o_v[None, :] * K
                        + o_k[:, None]
                    )
                    tl.store(
                        p_track, b_h.to(p_track.dtype.element_ty), mask=mask_h
                    )

    tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)


def advance_ssm_states_along_accept_paths(
    k_stash: torch.Tensor,
    v_stash: torch.Tensor,
    g_stash: torch.Tensor,
    beta_stash: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    ancestor_masks: torch.Tensor,
    last_correct_steps: torch.Tensor,
    T: int,
    Hg: int,
    K: int,
    V: int,
    track_steps: torch.Tensor = None,
    track_slots: torch.Tensor = None,
):
    """Advance pooled SSM states along each request's accept path.

    Stashes are [L, S, T, ...] flattened on the last dims; only the first
    len(cache_indices) stash slots are read. ssm_states is the full
    [L, pool, H, V, K] pool tensor; states are updated in place.

    track_steps/track_slots optionally snapshot the state right after a given
    accepted tree slot into another pool slot (mamba prefix-cache tracking).
    """
    L = ssm_states.shape[0]
    H = ssm_states.shape[2]
    B = cache_indices.shape[0]
    assert ancestor_masks.dim() == 3, "ancestor_masks must be [B, BT, BT // 64]"
    BT, NW = ancestor_masks.shape[-2], ancestor_masks.shape[-1]
    assert NW == BT // 64 and T <= BT, (BT, NW, T)
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    grid = (L, triton.cdiv(V, BV), B * H)
    tree_verify_state_advance_kernel[grid](
        k_stash=k_stash,
        v_stash=v_stash,
        g_stash=g_stash,
        beta_stash=beta_stash,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        ancestor_masks=ancestor_masks,
        last_correct_steps=last_correct_steps,
        track_steps=track_steps,
        track_slots=track_slots,
        T=T,
        stride_kl=k_stash.stride(0),
        stride_vl=v_stash.stride(0),
        stride_gl=g_stash.stride(0),
        stride_bl=beta_stash.stride(0),
        stride_sl=ssm_states.stride(0),
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        NW=NW,
        BK=BK,
        BV=BV,
        HAS_TRACK=track_steps is not None,
        num_warps=1,
        num_stages=2,
    )


