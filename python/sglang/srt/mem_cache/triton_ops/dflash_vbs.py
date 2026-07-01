"""Fused per-request VBS estimate for DFlash dynamic verify-block-size.

Replaces the sqrt -> cumprod -> sum -> ceil -> clamp -> int32 chain (~6 small
kernels queued *before* the shape .cpu() sync) with one Triton kernel: one program
per request does the tiny sequential cumprod-sum over the <=15 confidence values.
Fewer kernels before the sync => the .cpu() drains a shorter stream.

Only the default scale=0.5 (sqrt) path is fused; other scales fall back to torch.
The sequential product matches torch.cumprod's order, so per_req_vbs is bit-identical.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _vbs_from_conf_kernel(
    conf_ptr,      # [bs, M1] float32, per-draft-token top-1 confidence proxy
    out_ptr,       # [bs] int32 per-request verify length
    margin,
    lo,
    hi,
    bs,
    M1: tl.constexpr,   # block_size - 1 (unrolled)
):
    b = tl.program_id(0)
    if b >= bs:
        return
    prod = 1.0
    acc = 0.0
    for i in tl.static_range(M1):
        c = tl.load(conf_ptr + b * M1 + i)
        prod = prod * tl.sqrt(c)      # scale=0.5 -> sqrt; est_accept = sum_t prod_{<=t}
        acc = acc + prod
    vbs = tl.math.ceil(acc + margin)
    vbs = tl.minimum(tl.maximum(vbs, lo), hi)
    tl.store(out_ptr + b, vbs.to(tl.int32))


def compute_per_req_vbs(draft_conf, margin, lo, hi):
    """draft_conf: [bs, block-1] float32. Returns per_req_vbs [bs] int32 =
    clamp(ceil(sum(cumprod(sqrt(conf))) + margin), lo, hi)."""
    bs, m1 = draft_conf.shape
    out = torch.empty(bs, dtype=torch.int32, device=draft_conf.device)
    _vbs_from_conf_kernel[(bs,)](
        draft_conf.contiguous(), out, float(margin), int(lo), int(hi), bs, M1=m1
    )
    return out


def _demo():
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("no CUDA; skipping"); return
    dev = "cuda"
    for bs, block in [(1, 16), (5, 16), (32, 16), (128, 16), (7, 8)]:
        conf = torch.rand(bs, block - 1, device=dev, dtype=torch.float32) * 0.9 + 0.05
        for margin in (0.0, 1.0):
            for lo in (4,):
                scaled = torch.sqrt(conf)
                est = torch.cumprod(scaled, dim=1).sum(dim=1)
                ref = torch.ceil(est + margin).clamp(lo, block).to(torch.int32)
                got = compute_per_req_vbs(conf, margin, lo, block)
                diff = (ref - got).abs()
                assert int(diff.max()) == 0, (bs, block, margin, "maxdiff", int(diff.max()))
    print("dflash_vbs self-check OK (bit-identical to torch)")


if __name__ == "__main__":
    _demo()
