"""Equal-bytes pools for the Table 9 rerun on B300 (stock c1c287a4 + ~/ev/ovl_skv1): base vs shared context K/V.

    python agent_sizing_b300.py [tokens per checkpoint slot (1300)] [blocks, e.g. bf16:26.65,fp8:20 (bf16:32,fp8:24)]
prints, per block, each arm's N (int8 checkpoint slots), P (full-attention tokens) and the predicted GiB
(checkpoint + target KV + draft KV), to check against the boot log within +-0.05 GiB.

Cache bytes = int8 checkpoint pool + full-attention KV + draft KV, the GB300 rule (agent_sizing.py `up`) with
B300 unit costs read from the boot log of lane_agent_b300.sh (base, N=450, P=584960, 2026-10-27):
- checkpoint: "int8 mamba checkpoint pool: 450 slots, 7.43GB (qdata 6.61 + scale 0.21 + conv 0.62)"
  = 30 GDN layers x (32 x 128 x 128 int8 + 32 x 128 fp32 scale + 8192 x 3 bf16 conv) = 17,694,720 B per slot,
  allocated for N + 1 slots (6.61 = 451 x 15,728,640 B);
- target KV: "K size: 5.58 GB" for 584,960 tokens = 10 layers x 2 heads x 256 x (K, V) x 2 B = 20 KiB/token
  (fp8_e4m3: 10 KiB);
- draft KV: "K size: 6.70 GB" = 6 layers x 8 heads x 128 x (K, V) x 2 B = 24 KiB/token (base, bf16 in both blocks);
  shared = one layer, 4 KiB/token (ovl_skv1 log: DFLASH_SHARED_KV_ONE_COPY ... K+V bytes = 4096 x size).
The active mamba pool (160 slots), ReplaySSM rings and CUDA graphs are outside the budget, as on GB300.
"""
import sys

GiB, PAGE = 1 << 30, 64
CKPT = 30 * (32 * 128 * 128 * 1 + 32 * 128 * 4 + 8192 * 3 * 2)
FULL = {"bf16": 10 * 2 * 256 * 2 * 2, "fp8": 10 * 2 * 256 * 2 * 1}
DRAFT = {"base": 6 * 8 * 128 * 2 * 2, "shared": 1 * 8 * 128 * 2 * 2}
BLOCKS = (("bf16", 32), ("fp8", 24))


def size(budget_gib, kv, arm, ratio=1300):
    n = int(budget_gib * GiB // (CKPT + ratio * (FULL[kv] + DRAFT[arm])))
    p = n * ratio // PAGE * PAGE
    gib = ((n + 1) * CKPT + p * (FULL[kv] + DRAFT[arm])) / GiB
    return n, p, gib


if __name__ == "__main__":
    ratio = int(sys.argv[1]) if len(sys.argv) > 1 else 1300
    assert size(32, "bf16", "base") == (450, 584960, size(32, "bf16", "base")[2])  # the calibration boot
    blocks = [(k, float(b)) for k, b in (x.split(":") for x in sys.argv[2].split(","))] if len(sys.argv) > 2 else BLOCKS
    for kv, b in blocks:
        for arm in DRAFT:
            n, p, g = size(b, kv, arm, ratio)
            print(f"{kv} {b} GiB {arm:6s} N {n:4d} P {p:7d}  predicted {g:.2f} GiB")
