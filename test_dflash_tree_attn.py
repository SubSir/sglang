"""Losslessness test: lean dflash tree-verify kernel vs extend_attention_fwd oracle.

# ponytail: needs CUDA
Run on a GPU box:  python3 test_dflash_tree_attn.py
"""

import torch

from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd,
)
from sglang.srt.layers.attention.triton_ops.dflash_tree_attn import (
    dflash_tree_verify_attn_fwd,
)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    dtype = torch.bfloat16

    bs = 2
    context_len = 48
    N = 16  # tree tokens per seq (extend len)
    Hq, Hkv = 8, 2
    D = 128
    sm_scale = 1.0 / (D**0.5)
    pool_size = 256

    # Ragged extend layout: N tokens per seq, contiguous.
    qo_indptr = torch.tensor([0, N, 2 * N], dtype=torch.int32, device=dev)
    # Prefix KV indptr + gathered pool slots (exercise the kv_indices gather).
    kv_indptr = torch.tensor(
        [0, context_len, 2 * context_len], dtype=torch.int32, device=dev
    )
    kv_indices = torch.randperm(pool_size, device=dev)[: bs * context_len].to(
        torch.int64
    )

    q_extend = torch.randn(bs * N, Hq, D, dtype=dtype, device=dev)
    k_extend = torch.randn(bs * N, Hkv, D, dtype=dtype, device=dev)
    v_extend = torch.randn(bs * N, Hkv, D, dtype=dtype, device=dev)
    k_buffer = torch.randn(pool_size, Hkv, D, dtype=dtype, device=dev)
    v_buffer = torch.randn(pool_size, Hkv, D, dtype=dtype, device=dev)

    # Compact N*N tree mask per seq: lower-triangular random ancestor set, self True.
    masks = []
    for _ in range(bs):
        m = torch.rand(N, N, device=dev) < 0.5
        m = torch.tril(m)
        m.fill_diagonal_(True)
        masks.append(m.reshape(-1))
    compact_mask = torch.cat(masks).to(torch.uint8)
    mask_indptr = torch.tensor([0, N * N, 2 * N * N], dtype=torch.int32, device=dev)

    o_a = torch.empty(bs * N, Hq, D, dtype=dtype, device=dev)
    o_b = torch.empty(bs * N, Hq, D, dtype=dtype, device=dev)

    extend_attention_fwd(
        q_extend,
        k_extend,
        v_extend,
        o_a,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        compact_mask,
        True,  # is_causal
        mask_indptr,
        N,  # max_len_extend
        1.0,  # k_scale
        1.0,  # v_scale
        sm_scale=sm_scale,
        skip_prefix_custom_mask=True,
        compact_tree_mask=True,
    )

    dflash_tree_verify_attn_fwd(
        q_extend,
        k_extend,
        v_extend,
        o_b,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        compact_mask,
        mask_indptr,
        N,
        sm_scale=sm_scale,
    )

    ok = torch.allclose(o_a, o_b, rtol=2e-2, atol=2e-2)
    max_diff = (o_a.float() - o_b.float()).abs().max().item()
    print(f"max abs diff = {max_diff:.4e}")
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    main()
