"""DFlash2 sublayer boundary in CuTe DSL (the CuTe twin of dflash_boundary.py):

    h_t = (fb0 + fc_t,0) * y_t + [t>0] (fb1 + fc_t,1) * y_{t-1}      finish of the last sublayer
    s_t = residual_t + h_t,  u_t = s_t * rstd_t * w                  residual add + RMSNorm
    d_t = u_t . W^T                                                  coefficient GEMM (mma.sync)
    x_t = (pb0 + d_t,0) * u_t + [t>0] (pb1 + d_t,1) * u_{t-1}        prepare conv

Two shapes of it, both on thread-block clusters of NSPLIT CTAs along K that reduce over
distributed shared memory (no global workspace, deterministic):
  FusedBoundaryKernel   everything in one launch; each cluster recomputes the norm of its
                        K slice, which only pays while the rows are few; the GEMM partials
                        are pushed (st.async) to the rank that finishes them
  FinishNormKernel +    row-parallel finish/add/RMSNorm, then the GEMM + prepare conv
  GemmConvMmaKernel
The weight is permuted once per group block so a CTA's 4 * GB coefficient rows are contiguous.
"""
import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

GROUP = 16
COMPILE_OPTIONS = "--enable-tvm-ffi"


@dsl_user_op
def _cluster_sync(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;", "",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def _elem_ptr(t: cute.Tensor, crd, *, loc=None, ip=None) -> cute.Pointer:
    return t.iterator + cute.crd2idx(crd, t.layout, loc=loc, ip=ip)


@dsl_user_op
def _ld_peer(ptr: cute.Pointer, rank: Int32, *, loc=None, ip=None) -> Float32:
    """Load a float from the same shared-memory offset of CTA `rank` in this cluster."""
    return Float32(llvm.inline_asm(
        T.f32(), [ptr.toint(loc=loc, ip=ip).ir_value(), Int32(rank).ir_value(loc=loc, ip=ip)],
        "{ .reg .u32 r; mapa.shared::cluster.u32 r, $1, $2; ld.shared::cluster.f32 $0, [r]; }", "=f,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def _push_f32(val: Float32, smem_ptr: cute.Pointer, mbar_ptr: cute.Pointer, rank: Int32, *, loc=None, ip=None):
    """st.async a float to the same shared-memory offset in CTA `rank`, completing on that CTA's mbarrier."""
    llvm.inline_asm(
        None,
        [smem_ptr.toint(loc=loc, ip=ip).ir_value(), Float32(val).ir_value(loc=loc, ip=ip),
         mbar_ptr.toint(loc=loc, ip=ip).ir_value(), Int32(rank).ir_value(loc=loc, ip=ip)],
        "{ .reg .u32 ra, rb; mapa.shared::cluster.u32 ra, $0, $3; mapa.shared::cluster.u32 rb, $2, $3; "
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [ra], $1, [rb]; }",
        "r,f,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def _warp_sum(val: Float32) -> Float32:
    for i in cutlass.range_constexpr(5):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


# ----------------------------------------------------------------------------- K1


class FinishNormKernel:
    def __init__(self, H: int, has_finish: bool, has_res: bool, block: int = 8):
        self.H, self.G, self.has_finish, self.has_res, self.block = H, H // GROUP, has_finish, has_res, block
        self.T = H // 8
        assert self.T % 32 == 0 and self.T <= 1024

    @cute.jit
    def __call__(self, mY: cute.Tensor, mFc: cute.Tensor, mFb: cute.Tensor, mR: cute.Tensor,
                 mNw: cute.Tensor, mU: cute.Tensor, mRo: cute.Tensor, M: Int32, eps: Float32, stream):
        self.kernel(mY, mFc, mFb, mR, mNw, mU, mRo, eps).launch(
            grid=[M, 1, 1], block=[self.T, 1, 1], smem=4 * 64, stream=stream)

    @cute.kernel
    def kernel(self, mY: cute.Tensor, mFc: cute.Tensor, mFb: cute.Tensor, mR: cute.Tensor,
               mNw: cute.Tensor, mU: cute.Tensor, mRo: cute.Tensor, eps: Float32):
        H, G = self.H, self.G
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        smem = cutlass.utils.SmemAllocator()
        sRed = smem.allocate_tensor(Float32, cute.make_layout((32,)), byte_alignment=16)
        mNw2 = cute.make_tensor(mNw.iterator, cute.make_layout((1, H), stride=(0, 1)))

        h = cute.local_tile(mY, (1, 8), (row, tid)).load().to(Float32)
        if cutlass.const_expr(self.has_finish):
            g = tid // 2
            has_prev = min(row % self.block, 1)
            k0 = cute.local_tile(mFb, (1, 8), (0, tid)).load().to(Float32) + mFc[row, 2 * G + g].to(Float32)
            k1 = cute.local_tile(mFb, (1, 8), (1, tid)).load().to(Float32) + mFc[row, 3 * G + g].to(Float32)
            yp = cute.local_tile(mY, (1, 8), (row - has_prev, tid)).load().to(Float32)
            h = k0 * h + k1 * yp * Float32(has_prev)
        if cutlass.const_expr(self.has_res):
            h = h + cute.local_tile(mR, (1, 8), (row, tid)).load().to(Float32)
        nw = cute.local_tile(mNw2, (1, 8), (0, tid)).load().to(Float32)
        cute.local_tile(mRo, (1, 8), (row, tid)).store(h.to(BFloat16))

        sq = _warp_sum((h * h).reduce(cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0))
        if lane == 0:
            sRed[warp] = sq
        cute.arch.barrier()
        tot = Float32(0.0)
        if lane < self.T // 32:
            tot = sRed[lane]
        tot = _warp_sum(tot)
        rstd = cute.math.rsqrt(tot / Float32(H) + eps)
        cute.local_tile(mU, (1, 8), (row, tid)).store((h * rstd * nw).to(BFloat16))


# ----------------------------------------------------------------------------- K2


class GemmConvMmaKernel:
    """K2 on tensor cores: each CTA of a cluster copies its whole K slice of u (BM x KS)
    and of the group-block-permuted weight (4 * GB x KS) into swizzled shared memory
    with cp.async, waits once, and runs KS / 16 mma.sync steps; the accumulators go to
    shared memory, are summed across the cluster through DSMEM, and the prepare conv
    runs on the finishing rank. The weight is permuted on the host so a group block's
    4 * GB coefficient rows are contiguous: Wp[gb * 4GB + st * GB + g] = W[st * G + gb * GB + g]."""

    def __init__(self, H: int, BM: int, GB: int, NSPLIT: int, atom_m: int, atom_n: int, block: int = 8):
        self.H, self.G, self.BM, self.GB, self.NSPLIT = H, H // GROUP, BM, GB, NSPLIT
        self.KS = H // NSPLIT
        self.BN = 4 * GB
        self.atom_m, self.atom_n = atom_m, atom_n
        self.T = atom_m * atom_n * 32
        self.GPR = GB // NSPLIT
        assert H % NSPLIT == 0 and self.KS % 64 == 0 and BM % (16 * atom_m) == 0 and self.BN % (16 * atom_n) == 0
        assert self.G % GB == 0 and GB % NSPLIT == 0 and NSPLIT <= 16
        assert BM % block == 0 and BM % (self.T // 8) == 0 and self.BN % (self.T // 8) == 0
        self.CH = self.KS // 64
        self.block = block

    def _smem_layout(self, rows):
        # (rows, 64, KS / 64): 64-column K-major atoms with a 3-bit swizzle, as the Ampere
        # tensorop example; the third mode plays the role of its pipeline stages
        atom = cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0,
                                         cute.make_layout((8, 64), stride=(64, 1)))
        return cute.tile_to_shape(atom, (rows, 64, self.CH), (0, 1, 2))

    def smem_bytes(self):
        return 2 * (self.BM + self.BN) * self.KS + 4 * self.BM * self.BN + 4 * self.BM * 4 * self.GPR + 1024

    @cute.jit
    def __call__(self, mU: cute.Tensor, mW: cute.Tensor, mPb: cute.Tensor, mX: cute.Tensor,
                 mC: cute.Tensor, M: Int32, stream):
        sA_layout = self._smem_layout(self.BM)
        sB_layout = self._smem_layout(self.BN)
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            BFloat16, num_bits_per_copy=128)
        tcopy = cute.make_tiled_copy_tv(atom_copy, cute.make_layout((self.T // 8, 8), stride=(8, 1)),
                                        cute.make_layout((1, 8)))
        op = cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(op, cute.make_layout((self.atom_m, self.atom_n, 1)),
                                        permutation_mnk=(self.atom_m * 16, self.atom_n * 16, 16))
        self.kernel(mU, mW, mPb, mX, mC, M, sA_layout, sB_layout, tcopy, tiled_mma).launch(
            grid=[self.NSPLIT, self.G // self.GB, cute.ceil_div(M, self.BM)],
            block=[self.T, 1, 1], cluster=[self.NSPLIT, 1, 1], smem=self.smem_bytes(), stream=stream)

    @cute.kernel
    def kernel(self, mU: cute.Tensor, mW: cute.Tensor, mPb: cute.Tensor, mX: cute.Tensor, mC: cute.Tensor,
               M: Int32, sA_layout: cute.ComposedLayout, sB_layout: cute.ComposedLayout,
               tcopy: cute.TiledCopy, tiled_mma: cute.TiledMma):
        H, G, BM, BN, GB, KS, GPR, NSPLIT = self.H, self.G, self.BM, self.BN, self.GB, self.KS, self.GPR, self.NSPLIT
        FIN = BM * 4 * GPR
        CONV_TASKS = BM * GPR * 2
        tid, _, _ = cute.arch.thread_idx()
        ks, gb, rb = cute.arch.block_idx()
        row0 = rb * BM

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(BFloat16, sA_layout, byte_alignment=128)
        sB = smem.allocate_tensor(BFloat16, sB_layout, byte_alignment=128)
        sP = smem.allocate_tensor(Float32, cute.make_layout((BM, BN), stride=(BN, 1)), byte_alignment=16)
        sD = smem.allocate_tensor(Float32, cute.make_layout((BM, 4 * GPR), stride=(4 * GPR, 1)), byte_alignment=16)

        # the whole K slice of u and of this group block's weight rows, in one go
        CH = self.CH
        gA = cute.local_tile(mU, (BM, 64), (rb, None))            # (BM, 64, H / 64)
        gB = cute.local_tile(mW, (BN, 64), (gb, None))
        cA = cute.local_tile(cute.make_identity_tensor(mU.shape), (BM, 64), (rb, None))
        thr_copy = tcopy.get_slice(tid)
        tAgA = thr_copy.partition_S(gA)                           # (CPY, CPY_M, CPY_K, H / 64)
        tAsA = thr_copy.partition_D(sA)                           # (CPY, CPY_M, CPY_K, CH)
        tAcA = thr_copy.partition_S(cA)
        tBgB = thr_copy.partition_S(gB)
        tBsB = thr_copy.partition_D(sB)
        for m in cutlass.range_constexpr(cute.size(tAgA, mode=[1])):
            if cute.elem_less(tAcA[0, m, 0, 0][0], M):
                for c in cutlass.range_constexpr(CH):
                    cute.copy(tcopy, tAgA[None, m, None, ks * CH + c], tAsA[None, m, None, c])
        for c in cutlass.range_constexpr(CH):
            cute.copy(tcopy, tBgB[None, None, None, ks * CH + c], tBsB[None, None, None, c])
        cute.arch.cp_async_commit_group()

        # the conv inputs this rank may need, also issued before waiting
        CONV_IT = -(-CONV_TASKS // self.T)
        conv_in = []
        for it in cutlass.range_constexpr(CONV_IT):
            t = min(tid + it * self.T, CONV_TASKS - 1)
            c_r = t // (GPR * 2)
            c_q = t % (GPR * 2)
            c_c8 = ((gb * GB + ks * GPR) * GROUP + c_q * 8) // 8
            c_row = min(row0 + c_r, M - 1)
            c_prev = min(c_row % self.block, 1)
            conv_in.append((c_r, c_q, c_c8, c_row, c_prev,
                            cute.local_tile(mU, (1, 8), (c_row, c_c8)).load().to(Float32),
                            cute.local_tile(mU, (1, 8), (c_row - c_prev, c_c8)).load().to(Float32),
                            cute.local_tile(mPb, (1, 8), (0, c_c8)).load().to(Float32),
                            cute.local_tile(mPb, (1, 8), (1, c_c8)).load().to(Float32)))

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        thr_mma = tiled_mma.get_slice(tid)
        tCsA = thr_mma.partition_A(sA)                            # (MMA, MMA_M, MMA_K, CH)
        tCsB = thr_mma.partition_B(sB)
        tCsP = thr_mma.partition_C(sP)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCsP)
        tCrC.fill(0.0)
        s2r_A = cute.make_tiled_copy_A(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), BFloat16),
                                       tiled_mma)
        s2r_B = cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), BFloat16),
                                       tiled_mma)
        thr_s2r_A = s2r_A.get_slice(tid)
        thr_s2r_B = s2r_B.get_slice(tid)
        tCsA_v = thr_s2r_A.partition_S(sA)
        tCrA_v = thr_s2r_A.retile(tCrA)
        tCsB_v = thr_s2r_B.partition_S(sB)
        tCrB_v = thr_s2r_B.retile(tCrB)
        for c in cutlass.range_constexpr(CH):
            for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                cute.copy(s2r_A, tCsA_v[None, None, kb, c], tCrA_v[None, None, kb])
                cute.copy(s2r_B, tCsB_v[None, None, kb, c], tCrB_v[None, None, kb])
                cute.gemm(tiled_mma, tCrC, tCrA[None, None, kb], tCrB[None, None, kb], tCrC)
        cute.autovec_copy(tCrC, tCsP)

        _cluster_sync()
        FIN_IT = -(-FIN // self.T)
        fin = []
        for it in cutlass.range_constexpr(FIN_IT):
            fidx = min(tid + it * self.T, FIN - 1)
            f_r = fidx // (4 * GPR)
            f_jl = fidx % (4 * GPR)
            f_st = f_jl // GPR
            f_g = f_jl % GPR
            fin.append((f_r, f_jl, f_st, f_g,
                        [_ld_peer(_elem_ptr(sP, (f_r, f_st * GB + ks * GPR + f_g)), q) for q in range(NSPLIT)]))
        for it in cutlass.range_constexpr(FIN_IT):
            f_r, f_jl, f_st, f_g, parts = fin[it]
            if tid + it * self.T < FIN:
                tot = Float32(0.0)
                for q in cutlass.range_constexpr(NSPLIT):
                    tot = tot + parts[q]
                sD[f_r, f_jl] = tot
                if row0 + f_r < M:
                    mC[row0 + f_r, f_st * G + gb * GB + ks * GPR + f_g] = tot.to(BFloat16)
        cute.arch.barrier()
        for it in cutlass.range_constexpr(CONV_IT):
            c_r, c_q, c_c8, c_row, c_prev, cu, cup, pb0, pb1 = conv_in[it]
            if tid + it * self.T < CONV_TASKS:
                if row0 + c_r < M:
                    g = c_q // 2
                    x = (pb0 + sD[c_r, g]) * cu + (pb1 + sD[c_r, GPR + g]) * cup * Float32(c_prev)
                    cute.local_tile(mX, (1, 8), (c_row, c_c8)).store(x.to(BFloat16))
        _cluster_sync()


class FusedBoundaryKernel:
    def __init__(self, H: int, BM: int, GB: int, NSPLIT: int, atom_m: int, atom_n: int,
                 has_finish: bool, has_res: bool, block: int = 8, phases: int = 7, BMV: int = 0):
        self.H, self.G, self.BM, self.GB, self.NSPLIT = H, H // GROUP, BM, GB, NSPLIT
        # BMV rows are real per row block; the MMA tile is BM (>= 16) and the rest is zeros
        self.BMV = BMV or BM
        assert self.BMV <= BM and self.BMV % block == 0
        self.KS = H // NSPLIT
        self.BN = 4 * GB
        self.atom_m, self.atom_n = atom_m, atom_n
        self.T = atom_m * atom_n * 32
        self.GPR = GB // NSPLIT
        self.CH = self.KS // 64
        assert H % NSPLIT == 0 and self.KS % 64 == 0 and BM % (16 * atom_m) == 0 and self.BN % (16 * atom_n) == 0
        assert self.G % GB == 0 and GB % NSPLIT == 0 and NSPLIT <= 16 and BM % block == 0
        assert self.BN % (self.T // 8) == 0
        # the s slice goes row-major over threads: TPR consecutive lanes per row
        self.TPR = self.T // self.BMV
        assert self.T % self.BMV == 0 and self.TPR >= 1 and self.TPR <= 32 and 32 % self.TPR == 0 and (self.KS // 8) % self.TPR == 0
        self.has_finish, self.has_res, self.block = has_finish, has_res, block
        self.phases = phases  # bit 0: s + norm, bit 1: MMA, bit 2: finish -- for timing the parts

    def _smem_layout(self, rows):
        atom = cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0,
                                         cute.make_layout((8, 64), stride=(64, 1)))
        return cute.tile_to_shape(atom, (rows, 64, self.CH), (0, 1, 2))

    def smem_bytes(self):
        return (2 * (self.BM + self.BN) * self.KS + 4 * self.BM * self.BN + 4 * self.BMV * 4 * self.GPR
                + 4 * self.NSPLIT * self.BMV * 4 * self.GPR + 8 * self.BM + 1024)

    @cute.jit
    def __call__(self, mY: cute.Tensor, mFc: cute.Tensor, mFb: cute.Tensor, mR: cute.Tensor,
                 mNw: cute.Tensor, mW: cute.Tensor, mPb: cute.Tensor, mX: cute.Tensor,
                 mRo: cute.Tensor, mC: cute.Tensor, M: Int32, eps: Float32, stream):
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            BFloat16, num_bits_per_copy=128)
        tcopy = cute.make_tiled_copy_tv(atom_copy, cute.make_layout((self.T // 8, 8), stride=(8, 1)),
                                        cute.make_layout((1, 8)))
        op = cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(op, cute.make_layout((self.atom_m, self.atom_n, 1)),
                                        permutation_mnk=(self.atom_m * 16, self.atom_n * 16, 16))
        self.kernel(mY, mFc, mFb, mR, mNw, mW, mPb, mX, mRo, mC, M, eps,
                    self._smem_layout(self.BM), self._smem_layout(self.BN), tcopy, tiled_mma).launch(
            grid=[self.NSPLIT, self.G // self.GB, cute.ceil_div(M, self.BMV)],
            block=[self.T, 1, 1], cluster=[self.NSPLIT, 1, 1], smem=self.smem_bytes(), stream=stream)

    @cute.jit
    def residual_sum8(self, mY, mFc, mFb, mR, row, c8):
        """s at channels 8*c8 .. 8*c8+7 of `row` (one conv group), in fp32; branch-free."""
        G = self.G
        h = cute.local_tile(mY, (1, 8), (row, c8)).load().to(Float32)
        if cutlass.const_expr(self.has_finish):
            g = c8 // 2
            has_prev = min(row % self.block, 1)
            k0 = cute.local_tile(mFb, (1, 8), (0, c8)).load().to(Float32) + mFc[row, 2 * G + g].to(Float32)
            k1 = cute.local_tile(mFb, (1, 8), (1, c8)).load().to(Float32) + mFc[row, 3 * G + g].to(Float32)
            yp = cute.local_tile(mY, (1, 8), (row - has_prev, c8)).load().to(Float32)
            h = k0 * h + k1 * yp * Float32(has_prev)
        if cutlass.const_expr(self.has_res):
            h = h + cute.local_tile(mR, (1, 8), (row, c8)).load().to(Float32)
        return h

    @cute.kernel
    def kernel(self, mY: cute.Tensor, mFc: cute.Tensor, mFb: cute.Tensor, mR: cute.Tensor,
               mNw: cute.Tensor, mW: cute.Tensor, mPb: cute.Tensor, mX: cute.Tensor,
               mRo: cute.Tensor, mC: cute.Tensor, M: Int32, eps: Float32,
               sA_layout: cute.ComposedLayout, sB_layout: cute.ComposedLayout,
               tcopy: cute.TiledCopy, tiled_mma: cute.TiledMma):
        H, G, BM, BN, GB, KS, GPR, NSPLIT, CH = (self.H, self.G, self.BM, self.BN, self.GB, self.KS,
                                                 self.GPR, self.NSPLIT, self.CH)
        BMV = self.BMV
        T = self.T
        VEC = KS // 8                       # 8-channel vectors per row of the slice
        FIN = BMV * 4 * GPR
        FIN_IT = -(-FIN // T)
        CONV = BMV * GPR * 2
        CONV_IT = -(-CONV // T)
        tid, _, _ = cute.arch.thread_idx()
        ks, gb, rb = cute.arch.block_idx()
        row0 = rb * BMV
        mNw2 = cute.make_tensor(mNw.iterator, cute.make_layout((1, H), stride=(0, 1)))

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(BFloat16, sA_layout, byte_alignment=128)
        sB = smem.allocate_tensor(BFloat16, sB_layout, byte_alignment=128)
        sP = smem.allocate_tensor(Float32, cute.make_layout((BM, BN), stride=(BN, 1)), byte_alignment=16)
        sD = smem.allocate_tensor(Float32, cute.make_layout((BMV, 4 * GPR), stride=(4 * GPR, 1)), byte_alignment=16)
        sSq = smem.allocate_tensor(Float32, cute.make_layout((BMV,)), byte_alignment=16)
        sRstd = smem.allocate_tensor(Float32, cute.make_layout((BMV,)), byte_alignment=16)
        # partials pushed by every rank for the groups this rank finishes: [src rank, row, coefficient]
        sRecv = smem.allocate_tensor(Float32, cute.make_layout((NSPLIT, BMV, 4 * GPR), stride=(BMV * 4 * GPR, 4 * GPR, 1)),
                                     byte_alignment=16)
        mbar = smem.allocate_array(cutlass.Int64, num_elems=1)
        if tid == 0:
            cute.arch.mbarrier_init(mbar, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, NSPLIT * FIN * 4)

        # 1. the weight slice (cp.async) and every input of this CTA, before any use
        gB = cute.local_tile(mW, (BN, 64), (gb, None))
        thr_copy = tcopy.get_slice(tid)
        tBgB = thr_copy.partition_S(gB)
        tBsB = thr_copy.partition_D(sB)
        for c in cutlass.range_constexpr(CH):
            cute.copy(tcopy, tBgB[None, None, None, ks * CH + c], tBsB[None, None, None, c])
        cute.arch.cp_async_commit_group()

        # thread -> (row, VPT vectors of it); VPT = VEC / TPR
        TPR = self.TPR
        VPT = VEC // TPR
        s_r = tid // TPR
        s_row = min(row0 + s_r, M - 1)
        s_v0 = (tid % TPR) * VPT
        VPT_ON = VPT if self.phases & 1 else 0
        svals = [self.residual_sum8(mY, mFc, mFb, mR, s_row, ks * VEC + s_v0 + i) for i in range(VPT_ON)]
        nvals = [cute.local_tile(mNw2, (1, 8), (0, ks * VEC + s_v0 + i)).load().to(Float32) for i in range(VPT_ON)]
        conv_in = []
        for it in cutlass.range_constexpr(CONV_IT if self.phases & 4 else 0):
            t = min(tid + it * T, CONV - 1)
            c_r = t // (GPR * 2)
            c_c8 = ((gb * GB + ks * GPR) * GROUP) // 8 + t % (GPR * 2)
            c_row = min(row0 + c_r, M - 1)
            c_prev = min(c_row % self.block, 1)
            conv_in.append((c_r, t % (GPR * 2), c_c8, c_row, c_prev,
                            self.residual_sum8(mY, mFc, mFb, mR, c_row, c_c8),
                            self.residual_sum8(mY, mFc, mFb, mR, c_row - c_prev, c_c8),
                            cute.local_tile(mNw2, (1, 8), (0, c_c8)).load().to(Float32),
                            cute.local_tile(mPb, (1, 8), (0, c_c8)).load().to(Float32),
                            cute.local_tile(mPb, (1, 8), (1, c_c8)).load().to(Float32)))

        # 2. partial sums of squares of this slice: TPR lanes per row, one store per row
        sq = Float32(0.0)
        for i in cutlass.range_constexpr(VPT_ON):
            sq = sq + (svals[i] * svals[i]).reduce(cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0)
        for i in cutlass.range_constexpr(TPR.bit_length() - 1):
            sq = sq + cute.arch.shuffle_sync_bfly(sq, offset=1 << i)
        if tid % TPR == 0:
            sSq[s_r] = sq

        # 3. rstd from every rank's partial
        _cluster_sync()
        if tid < BMV:
            parts = [_ld_peer(_elem_ptr(sSq, (tid,)), q) for q in range(NSPLIT)]
            tot = Float32(0.0)
            for q in cutlass.range_constexpr(NSPLIT):
                tot = tot + parts[q]
            sRstd[tid] = cute.math.rsqrt(tot / Float32(H) + eps)
        cute.arch.barrier()

        if cutlass.const_expr(BMV < BM):
            zero8 = cute.make_rmem_tensor(cute.make_layout((1, 8)), BFloat16)
            zero8.fill(0.0)
            for i in cutlass.range(tid, (BM - BMV) * VEC, T):
                col = (i % VEC) * 8
                dst = cute.make_tensor(_elem_ptr(sA, (BMV + i // VEC, col % 64, col // 64)).align(16),
                                       cute.make_layout((1, 8)))
                dst.store(zero8.load())
        # 4. u = s * rstd * w into the MMA operand (16-byte stores: the 3-bit swizzle moves
        # whole 16-byte chunks), then the GEMM of this slice
        for i in cutlass.range_constexpr(VPT_ON):
            col = (s_v0 + i) * 8
            dst = cute.make_tensor(_elem_ptr(sA, (s_r, col % 64, col // 64)).align(16), cute.make_layout((1, 8)))
            dst.store((svals[i] * sRstd[s_r] * nvals[i]).to(BFloat16))
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        thr_mma = tiled_mma.get_slice(tid)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCsP = thr_mma.partition_C(sP)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCsP)
        tCrC.fill(0.0)
        s2r_A = cute.make_tiled_copy_A(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), BFloat16),
                                       tiled_mma)
        s2r_B = cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), BFloat16),
                                       tiled_mma)
        thr_s2r_A = s2r_A.get_slice(tid)
        thr_s2r_B = s2r_B.get_slice(tid)
        tCsA_v = thr_s2r_A.partition_S(sA)
        tCrA_v = thr_s2r_A.retile(tCrA)
        tCsB_v = thr_s2r_B.partition_S(sB)
        tCrB_v = thr_s2r_B.retile(tCrB)
        for c in cutlass.range_constexpr(CH if self.phases & 2 else 0):
            for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                cute.copy(s2r_A, tCsA_v[None, None, kb, c], tCrA_v[None, None, kb])
                cute.copy(s2r_B, tCsB_v[None, None, kb, c], tCrB_v[None, None, kb])
                cute.gemm(tiled_mma, tCrC, tCrA[None, None, kb], tCrB[None, None, kb], tCrC)
        cute.autovec_copy(tCrC, tCsP)
        cute.arch.barrier()

        # 5. push each rank the partials of the groups it finishes (the first cluster barrier
        # published every mbarrier), then finish this rank's GPR groups from local memory
        PUSH_IT = -(-(NSPLIT * FIN) // T)
        for it in cutlass.range_constexpr(PUSH_IT if self.phases & 4 else 0):
            idx = tid + it * T
            if idx < NSPLIT * FIN:
                q = idx // FIN
                rem = idx % FIN
                p_r = rem // (4 * GPR)
                p_jl = rem % (4 * GPR)
                val = sP[p_r, (p_jl // GPR) * GB + q * GPR + p_jl % GPR]
                _push_f32(val, _elem_ptr(sRecv, (ks, p_r, p_jl)), mbar, q)
        if cutlass.const_expr(self.phases & 4):
            cute.arch.mbarrier_wait(mbar, 0)
        for it in cutlass.range_constexpr(FIN_IT if self.phases & 4 else 0):
            fidx = tid + it * T
            if fidx < FIN:
                f_r = fidx // (4 * GPR)
                f_jl = fidx % (4 * GPR)
                f_st = f_jl // GPR
                f_g = f_jl % GPR
                tot = Float32(0.0)
                for q in cutlass.range_constexpr(NSPLIT):
                    tot = tot + sRecv[q, f_r, f_jl]
                sD[f_r, f_jl] = tot
                if row0 + f_r < M:
                    mC[row0 + f_r, f_st * G + gb * GB + ks * GPR + f_g] = tot.to(BFloat16)
        cute.arch.barrier()
        for it in cutlass.range_constexpr(CONV_IT if self.phases & 4 else 0):
            c_r, c_q, c_c8, c_row, c_prev, cs, csp, nw8, pb0, pb1 = conv_in[it]
            if tid + it * T < CONV:
                if row0 + c_r < M:
                    g = c_q // 2
                    cute.local_tile(mRo, (1, 8), (c_row, c_c8)).store(cs.to(BFloat16))
                    u = (cs * sRstd[c_r] * nw8).to(BFloat16).to(Float32)
                    up = (csp * sRstd[c_r - c_prev] * nw8).to(BFloat16).to(Float32)
                    x = (pb0 + sD[c_r, g]) * u + (pb1 + sD[c_r, GPR + g]) * up * Float32(c_prev)
                    cute.local_tile(mX, (1, 8), (c_row, c_c8)).store(x.to(BFloat16))


def _act(sym_m, width):
    return cute.runtime.make_fake_compact_tensor(BFloat16, (sym_m, width), stride_order=(1, 0), assumed_align=16)


def _const(shape):
    order = (0,) if len(shape) == 1 else (1, 0)
    return cute.runtime.make_fake_compact_tensor(BFloat16, shape, stride_order=order, assumed_align=16)


@functools.cache
def _k1(H, has_finish, has_res):
    G = H // GROUP
    m = cute.sym_int()
    return cute.compile(FinishNormKernel(H, has_finish, has_res), _act(m, H), _act(m, 4 * G), _const((2, H)),
                        _act(m, H), _const((H,)), _act(m, H), _act(m, H), Int32(8), Float32(1e-6),
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options=COMPILE_OPTIONS)


@functools.cache
def _k2mma(H, BM, GB, NSPLIT, atom_m, atom_n):
    G = H // GROUP
    m = cute.sym_int()
    return cute.compile(GemmConvMmaKernel(H, BM, GB, NSPLIT, atom_m, atom_n), _act(m, H), _const((4 * G, H)),
                        _const((2, H)), _act(m, H), _act(m, 4 * G), Int32(8),
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options=COMPILE_OPTIONS)


@functools.cache
def _k_fused(H, BM, GB, NSPLIT, atom_m, atom_n, has_finish, has_res, phases=7, BMV=0):
    G = H // GROUP
    m = cute.sym_int()
    kern = FusedBoundaryKernel(H, BM, GB, NSPLIT, atom_m, atom_n, has_finish, has_res, phases=phases, BMV=BMV)
    return cute.compile(kern, _act(m, H), _act(m, 4 * G), _const((2, H)), _act(m, H), _const((H,)),
                        _const((4 * G, H)), _const((2, H)), _act(m, H), _act(m, H), _act(m, 4 * G),
                        Int32(8), Float32(1e-6), cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options=COMPILE_OPTIONS)



def permute_weight(weight: torch.Tensor, GB: int) -> torch.Tensor:
    """[4G, H] -> group-block-major rows: Wp[gb * 4GB + st * GB + g] = W[st * G + gb * GB + g]."""
    G4, H = weight.shape
    G = G4 // 4
    return weight.view(4, G // GB, GB, H).permute(1, 0, 2, 3).reshape(G4, H).contiguous()




class DFlashBoundaryCute:
    """Same call contract as the Triton boundary in dflash_boundary.py; cfg per row count:
    ("fused", BM, GB, NSPLIT, atom_m, atom_n, BMV) or ("split", BM, GB, NSPLIT, atom_m, atom_n)."""

    def __init__(self, H: int, device, configs: dict, block: int = 8):
        self.H, self.G, self.block, self.configs = H, H // GROUP, block, configs
        self._perm = {}

    def config(self, M):
        for rows in sorted(self.configs):
            if M <= rows:
                return self.configs[rows]
        return None

    def prepare_weight(self, weight):
        """Permuted copies for every group-block size in the table; call before graph capture."""
        for cfg in self.configs.values():
            key = (weight.data_ptr(), cfg[2])
            if key not in self._perm:
                self._perm[key] = permute_weight(weight, cfg[2])

    def __call__(self, y, res, norm_w, weight, pbase, x, res_out, coef, *, fcoef=None, fbase=None, eps=1e-6):
        M = y.shape[0]
        cfg = self.config(M)
        kind, BM, GB, NSPLIT, atom_m, atom_n = cfg[:6]
        wp = self._perm[(weight.data_ptr(), GB)]
        has_finish, has_res = fcoef is not None, res is not None
        args = (fcoef if has_finish else coef, fbase if has_finish else pbase, res if has_res else y)
        if kind == "fused":
            _k_fused(self.H, BM, GB, NSPLIT, atom_m, atom_n, has_finish, has_res, BMV=cfg[6])(
                y, *args, norm_w, wp, pbase, x, res_out, coef, M, eps)
        else:
            u = torch.empty_like(y)
            _k1(self.H, has_finish, has_res)(y, *args, norm_w, u, res_out, M, eps)
            _k2mma(self.H, BM, GB, NSPLIT, atom_m, atom_n)(u, wp, pbase, x, coef, M)
