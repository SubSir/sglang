"""DFlash2 sublayer boundary: finish conv + residual add + RMSNorm + coefficient GEMM +
prepare conv, in two Triton kernels instead of five.

Between two GEMMs of a DFlash2 draft layer (o_proj -> gate_up, down -> next qkv) the
unfused path runs the previous sublayer's finish conv, a fused add+RMSNorm, the
coefficient GEMM (cuBLAS split-K + reduce) and the prepare conv. Here:

  finish_norm  one program per row:
                   h_t = (fb0 + fc_t,0) * y_t + [t>0] (fb1 + fc_t,1) * y_{t-1}
                   s_t = residual_t + h_t  -> residual out,  u_t = s_t * rstd_t * w -> u
  gemm_conv    split-K GEMM d = u . W^T into per-split fp32 slots; the last program to
               arrive for a (group block, row block) sums the slots and runs the prepare
               conv  x_t = (pb0 + d_t,0) * u_t + [t>0] (pb1 + d_t,1) * u_{t-1}

t is the position inside the draft block, fc/d are per (row, conv group of 16 channels)
coefficient pairs (the kernel_projection output), fb/pb per-channel base taps. The
arrival counters are reset by the last program, so both kernels replay in CUDA graphs.
For many rows the GEMM is left to the dense path (finish_norm + the module's own
projection and convolution), which measured faster there.
"""
import functools

import torch
import triton
import triton.language as tl

GROUP = 16


@triton.jit
def _residual_sum(y_ptr, fcoef_ptr, fbase_ptr, res_ptr, rows, kk, mask,
                  H: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr,
                  HAS_FINISH: tl.constexpr, HAS_RES: tl.constexpr):
    """s = residual + finish(y) at (rows, kk), in fp32."""
    off = rows * H + kk
    y0 = tl.load(y_ptr + off, mask=mask, other=0.0).to(tl.float32)
    if HAS_FINISH:
        g = kk // 16
        pm = mask & (rows % BLOCK != 0)
        y1 = tl.load(y_ptr + off - H, mask=pm, other=0.0).to(tl.float32)
        c0 = (tl.load(fbase_ptr + kk).to(tl.float32)
              + tl.load(fcoef_ptr + rows * (4 * G) + 2 * G + g, mask=mask, other=0.0).to(tl.float32))
        c1 = (tl.load(fbase_ptr + H + kk).to(tl.float32)
              + tl.load(fcoef_ptr + rows * (4 * G) + 3 * G + g, mask=mask, other=0.0).to(tl.float32))
        h = c0 * y0 + tl.where(pm, c1 * y1, 0.0)
    else:
        h = y0
    if HAS_RES:
        h = h + tl.load(res_ptr + off, mask=mask, other=0.0).to(tl.float32)
    return h


@triton.jit
def _finish_norm_kernel(y_ptr, fcoef_ptr, fbase_ptr, res_ptr, nw_ptr, out_ptr, res_out_ptr, eps,
                        H: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr, WIDTH: tl.constexpr,
                        HAS_FINISH: tl.constexpr, HAS_RES: tl.constexpr):
    # the GEMM after this (launched with PDL) may start launching right away
    tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    kk = tl.arange(0, WIDTH)
    mask = kk < H
    s = _residual_sum(y_ptr, fcoef_ptr, fbase_ptr, res_ptr, row, kk, mask, H, G, BLOCK, HAS_FINISH, HAS_RES)
    tl.store(res_out_ptr + row * H + kk, s.to(tl.bfloat16), mask=mask)
    rstd = 1.0 / tl.sqrt(tl.sum(s * s, axis=0) / H + eps)
    w = tl.load(nw_ptr + kk, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * H + kk, (s * rstd * w).to(tl.bfloat16), mask=mask)


@triton.jit
def _gc_finish(tot, coef_ptr, x_ptr, ws_off, off3, rmask, m3, p3, u3, up3, pb0, pb1,
               BM: tl.constexpr, GB: tl.constexpr):
    tl.store(coef_ptr + ws_off, tot.to(tl.bfloat16), mask=rmask[:, None])
    d4 = tl.permute(tl.reshape(tot, (BM, 2, 2, GB)), (0, 3, 2, 1))  # [BM, GB, tap, side]
    side0, _ = tl.split(d4)
    c0, c1 = tl.split(side0)
    x = (pb0 + c0[:, :, None]) * u3 + tl.where(p3, (pb1 + c1[:, :, None]) * up3, 0.0)
    tl.store(x_ptr + off3, x.to(tl.bfloat16), mask=m3)


@triton.jit
def _gemm_conv_kernel(u_ptr, w_ptr, pbase_ptr, x_ptr, coef_ptr, ws_ptr, cnt_ptr, M,
                      H: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr,
                      BM: tl.constexpr, GB: tl.constexpr, NSPLIT: tl.constexpr, BK: tl.constexpr,
                      MAX_M: tl.constexpr):
    ks = tl.program_id(0)
    gb = tl.program_id(1)
    rb = tl.program_id(2)
    NGB: tl.constexpr = G // GB
    KS: tl.constexpr = H // NSPLIT
    rows = rb * BM + tl.arange(0, BM)
    rmask = rows < M
    pm = rmask & (rows % BLOCK != 0)
    # coefficient rows owned here: j -> (side*2 + tap, group) = (j // GB, gb*GB + j % GB)
    j = tl.arange(0, 4 * GB)
    wrow = (j // GB) * G + gb * GB + (j % GB)
    acc = tl.zeros((BM, 4 * GB), tl.float32)
    # PDL overlaps only this program's launch with the norm kernel; loading a weight tile
    # before the wait measured 0.2-0.4 us slower than leaving the loop to tl.range's pipeliner.
    tl.extra.cuda.gdc_wait()
    for k0 in tl.range(ks * KS, ks * KS + KS, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(u_ptr + rows[:, None] * H + k[None, :], mask=rmask[:, None], other=0.0)
        w = tl.load(w_ptr + wrow[None, :] * H + k[:, None])
        acc = tl.dot(a, w, acc)

    # the conv inputs, loaded before arriving so the last program only waits on the slots
    ch = (gb * GB + tl.arange(0, GB))[:, None] * 16 + tl.arange(0, 16)[None, :]
    r3 = rows[:, None, None]
    m3 = rmask[:, None, None]
    p3 = pm[:, None, None]
    off3 = r3 * H + ch[None, :, :]
    u3 = tl.load(u_ptr + off3, mask=m3, other=0.0).to(tl.float32)
    up3 = tl.load(u_ptr + off3 - H, mask=p3, other=0.0).to(tl.float32)
    pb0 = tl.load(pbase_ptr + ch).to(tl.float32)[None, :, :]
    pb1 = tl.load(pbase_ptr + H + ch).to(tl.float32)[None, :, :]

    ws_off = rows[:, None] * (4 * G) + wrow[None, :]
    if NSPLIT == 1:
        _gc_finish(acc, coef_ptr, x_ptr, ws_off, off3, rmask, m3, p3, u3, up3, pb0, pb1, BM, GB)
    else:
        SLOT: tl.constexpr = MAX_M * 4 * G
        tl.store(ws_ptr + ks * SLOT + ws_off, acc, mask=rmask[:, None])
        tl.debug_barrier()
        arrived = tl.atomic_add(cnt_ptr + rb * NGB + gb, 1, sem="acq_rel")
        if arrived == NSPLIT - 1:
            tot = tl.zeros((BM, 4 * GB), tl.float32)
            for q in tl.static_range(NSPLIT):
                tot += tl.load(ws_ptr + q * SLOT + ws_off, mask=rmask[:, None], other=0.0, cache_modifier=".cg")
            tl.store(cnt_ptr + rb * NGB + gb, 0)
            _gc_finish(tot, coef_ptr, x_ptr, ws_off, off3, rmask, m3, p3, u3, up3, pb0, pb1, BM, GB)


def finish_norm(y, res, norm_weight, eps, out, res_out, fcoef=None, fbase=None, block=8):
    """res_out = res + finish(y) (or y alone for the first layer), out = RMSNorm(res_out).
    fcoef [M, 4G] is the previous kernel_projection output, fbase its conv's base_kernel[1]."""
    M, H = y.shape
    has_finish, has_res = fcoef is not None, res is not None
    _finish_norm_kernel[(M,)](
        y, fcoef if has_finish else y, fbase if has_finish else norm_weight, res if has_res else y,
        norm_weight, out, res_out, eps, H=H, G=H // GROUP, BLOCK=block,
        WIDTH=triton.next_power_of_2(H), HAS_FINISH=has_finish, HAS_RES=has_res, num_warps=8)


# (BM, GB, NSPLIT, BK, warps) per hidden size, for rows up to the key; None where the dense
# path (finish_norm + TGV GEMM + conv) measured faster. Measured on GB300 (inco/final_bench.py).
_CONFIGS = {
    2560: {8: (16, 4, 4, 128, 4), 16: (16, 4, 4, 128, 4), 32: (16, 16, 5, 128, 8)},
    5120: {8: None, 16: (16, 16, 5, 128, 8)},
}


# CuTe configs per hidden size, for rows up to the key (measured like _CONFIGS):
# (BM, GB, NSPLIT, atom_m, atom_n, BMV) for the one-launch cluster kernel.
CUTE_CONFIGS = {
    2560: {
        8: (16, 16, 8, 1, 2, 8),
        16: (16, 16, 8, 1, 4, 16),
        24: (16, 16, 4, 1, 4, 8),
        48: (16, 32, 8, 1, 4, 16),
        64: (32, 32, 8, 2, 4, 32),
    },
    5120: {
        8: (16, 32, 8, 1, 4, 8),
        16: (16, 16, 16, 1, 4, 16),
    },
}


def pick_config(H, M):
    table = _CONFIGS.get(H)
    if not table:
        return None
    for rows in sorted(table):
        if M <= rows:
            return table[rows]
    return None


_TGV_MAX_ROWS = 256


@functools.cache
def _tgv_gemm():
    """sglang's CuTe-DSL TGV GEMM on Blackwell (it beat cuBLAS at every coefficient shape
    measured), else None."""
    if torch.cuda.get_device_capability()[0] != 10:
        return None
    from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import cutedsl_bf16_gemm

    return cutedsl_bf16_gemm


def coef_gemm(u, weight):
    """The coefficient projection for the dense path."""
    tgv = _tgv_gemm()
    if tgv is not None and u.shape[0] <= _TGV_MAX_ROWS:
        return tgv(u, weight)
    return torch.nn.functional.linear(u, weight)


class DFlashBoundary:
    """One per draft model. Owns the per-split fp32 slots (overwritten on every call) and
    the arrival counters (reset by the last program), sized for up to max_m rows."""

    def __init__(self, H: int, device, max_m: int = 64, block: int = 8):
        self.H, self.G, self.block, self.max_m, self.device = H, H // GROUP, block, max_m, device
        self.cnt = torch.zeros(max_m * self.G, device=device, dtype=torch.int32)
        # allocated up front: a buffer created inside a CUDA graph capture would belong to it
        self._ws = {}
        for cfg in _CONFIGS.get(H, {}).values():
            if cfg is not None:
                self._workspace(cfg[2])

    def fused_rows(self, M):
        return M <= self.max_m and pick_config(self.H, M) is not None

    def _workspace(self, nsplit):
        if nsplit not in self._ws:
            self._ws[nsplit] = torch.empty(nsplit, self.max_m, 4 * self.G, device=self.device, dtype=torch.float32)
        return self._ws[nsplit]

    def gemm_conv(self, u, weight, pbase, x, coef):
        M = u.shape[0]
        BM, GB, NSPLIT, BK, warps = pick_config(self.H, M)
        _gemm_conv_kernel[(NSPLIT, self.G // GB, triton.cdiv(M, BM))](
            u, weight, pbase, x, coef, self._workspace(NSPLIT), self.cnt, M,
            H=self.H, G=self.G, BLOCK=self.block, BM=BM, GB=GB, NSPLIT=NSPLIT, BK=BK, MAX_M=self.max_m,
            num_warps=warps, launch_pdl=True)
