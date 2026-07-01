"""Fused tight-packing for DFlash dynamic verify-block-size.

Packs the 3 [bs, block] verify-input tensors (draft tokens, positions, cache locs)
into the ragged [total_padded] layout in one Triton kernel, replacing ~8 torch ops
(col/valid/valid_dest/unused_cum/unused_dest/where) + 3 separate scatter_ launches.
Used together with the fused vbs kernel so no small ops surround the shape .cpu()
sync. Sync-free (total_real is the already-transferred shape scalar).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_verify_inputs_kernel(
    per_req_vbs_ptr,  # [bs] int32
    vbs_cumsum_ptr,   # [bs+1] int32 (exclusive cumsum; [bs] = total_real)
    unused_cum_ptr,   # [bs] int32 (exclusive cumsum of block - per_req_vbs)
    ids_src_ptr, pos_src_ptr, loc_src_ptr,   # [bs, BLOCK]
    ids_dst_ptr, pos_dst_ptr, loc_dst_ptr,   # [bs*BLOCK]
    total_real,
    bs,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    if b >= bs:
        return
    vbs = tl.load(per_req_vbs_ptr + b)
    if c < vbs:
        dest = tl.load(vbs_cumsum_ptr + b) + c
    else:
        dest = total_real + tl.load(unused_cum_ptr + b) + (c - vbs)
    src = b * BLOCK + c
    tl.store(ids_dst_ptr + dest, tl.load(ids_src_ptr + src))
    tl.store(pos_dst_ptr + dest, tl.load(pos_src_ptr + src))
    tl.store(loc_dst_ptr + dest, tl.load(loc_src_ptr + src))


def pack_verify_inputs(per_req_vbs, vbs_cumsum, ids2d, pos2d, loc2d, total_real, total_padded):
    bs, block = ids2d.shape
    device = ids2d.device
    per32 = per_req_vbs.to(torch.int32)
    unused_cum = torch.zeros(bs, dtype=torch.int32, device=device)
    if bs > 1:
        unused_cum[1:] = torch.cumsum(block - per32, dim=0)[:-1]
    ids_dst = torch.empty(bs * block, dtype=ids2d.dtype, device=device)
    pos_dst = torch.empty(bs * block, dtype=pos2d.dtype, device=device)
    loc_dst = torch.empty(bs * block, dtype=loc2d.dtype, device=device)
    _pack_verify_inputs_kernel[(bs, block)](
        per32, vbs_cumsum.to(torch.int32), unused_cum,
        ids2d, pos2d, loc2d, ids_dst, pos_dst, loc_dst,
        int(total_real), bs, BLOCK=block,
    )
    return ids_dst[:total_padded], pos_dst[:total_padded], loc_dst[:total_padded]


def _demo():
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("no CUDA; skipping"); return
    dev = "cuda"
    for bs, block in [(1, 16), (5, 16), (32, 16), (7, 8)]:
        per = torch.randint(1, block + 1, (bs,), device=dev, dtype=torch.int32)
        total_real = int(per.sum())
        vbs_cumsum = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
        vbs_cumsum[1:] = torch.cumsum(per, dim=0)
        eff = max(1, int(per.float().mean().ceil()))
        total_padded = bs * eff
        ids = torch.randint(0, 99999, (bs, block), device=dev, dtype=torch.int64)
        pos = torch.randint(0, 4096, (bs, block), device=dev, dtype=torch.int64)
        loc = torch.randint(0, 8192, (bs, block), device=dev, dtype=torch.int64)
        col = torch.arange(block, device=dev, dtype=torch.int64)
        per64 = per.to(torch.int64)
        valid = col[None, :] < per64[:, None]
        valid_dest = vbs_cumsum[:bs].to(torch.int64)[:, None] + col[None, :]
        ucum = torch.zeros(bs, dtype=torch.int64, device=dev)
        if bs > 1:
            ucum[1:] = torch.cumsum(block - per64, dim=0)[:-1]
        unused_dest = total_real + ucum[:, None] + (col[None, :] - per64[:, None])
        dest = torch.where(valid, valid_dest, unused_dest).reshape(-1)
        def ref(s):
            o = torch.empty(bs * block, dtype=s.dtype, device=dev)
            o.scatter_(0, dest, s.reshape(-1)); return o[:total_padded]
        fi, fp, fl = pack_verify_inputs(per, vbs_cumsum, ids, pos, loc, total_real, total_padded)
        assert torch.equal(ref(ids), fi) and torch.equal(ref(pos), fp) and torch.equal(ref(loc), fl), (bs, block)
    print("dflash_pack self-check OK")


if __name__ == "__main__":
    _demo()
