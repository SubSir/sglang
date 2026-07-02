"""Correctness oracle for SGLANG_DFLASH_COMPACT_TREE_MASK.

Runs extend_attention_fwd twice from the SAME q/k/v and the SAME logical tree
ancestor relation:
  1. FULL_MASK layout: custom_mask [N x (context+N)], prefix cols all-True,
     tail block = ancestor. compact_tree_mask=False.
  2. QLEN_ONLY compact layout: custom_mask [N x N] = ancestor.
     compact_tree_mask=True, skip_prefix_custom_mask=True.
Both use skip_prefix_custom_mask=True (the DFlash default), so the prefix is
treated as always-visible causal in both runs -> outputs must match.

# ponytail: needs CUDA + triton. Cannot run on CPU/mac; run on a B200.
Run:  python test_compact_mask.py
"""

import torch

from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd


def build_scenario(bs, context, N, n_heads, head_dim, device, dtype):
    """One extend batch: each seq has `context` prefix tokens + N verify tokens.

    Returns q/k/v extend tensors, the k/v buffers (prefix KV in the pool),
    indptrs, and per-seq [N x N] ancestor relation (lower-tri-ish tree mask).
    """
    torch.manual_seed(0)
    total_extend = bs * N
    total_kv = bs * context  # prefix KV lives in the buffer

    q = torch.randn(total_extend, n_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_extend, n_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_extend, n_heads, head_dim, device=device, dtype=dtype)

    k_buffer = torch.randn(total_kv, n_heads, head_dim, device=device, dtype=dtype)
    v_buffer = torch.randn(total_kv, n_heads, head_dim, device=device, dtype=dtype)

    qo_indptr = torch.arange(0, (bs + 1) * N, step=N, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(
        0, (bs + 1) * context, step=context, dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(total_kv, dtype=torch.int32, device=device)

    # Per-seq N x N ancestor relation: a random tree, each node attends to its
    # ancestors + itself. Node i's parent is a random j < i; ancestor set is the
    # transitive closure. Row = query node, col = key node.
    ancestor = torch.zeros(bs, N, N, dtype=torch.bool, device=device)
    for b in range(bs):
        parent = [-1] * N
        for i in range(1, N):
            parent[i] = int(torch.randint(0, i, (1,)).item())
        for i in range(N):
            j = i
            while j != -1:
                ancestor[b, i, j] = True
                j = parent[j]
    return q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, ancestor


def run(compact, ancestor, context, N, bs, device, **kw):
    if compact:
        # QLEN_ONLY: [bs * N * N] flattened, row-stride N, no prefix cols.
        custom_mask = ancestor.reshape(-1).contiguous()
        seq_mask_len = torch.full((bs,), N * N, dtype=torch.int32, device=device)
    else:
        # FULL_MASK: [N x (context+N)] per seq, prefix all-True + tail = ancestor.
        full = torch.ones(bs, N, context + N, dtype=torch.bool, device=device)
        full[:, :, context:] = ancestor
        custom_mask = full.reshape(-1).contiguous()
        seq_mask_len = torch.full(
            (bs,), N * (context + N), dtype=torch.int32, device=device
        )
    mask_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    mask_indptr[1:] = torch.cumsum(seq_mask_len, dim=0)

    o = torch.empty_like(kw["q"])
    extend_attention_fwd(
        kw["q"],
        kw["k"],
        kw["v"],
        o,
        kw["k_buffer"],
        kw["v_buffer"],
        kw["qo_indptr"],
        kw["kv_indptr"],
        kw["kv_indices"],
        custom_mask,
        True,  # is_causal (ignored under custom mask for extend tail)
        mask_indptr,
        N,  # max_len_extend
        1.0,  # k_scale
        1.0,  # v_scale
        skip_prefix_custom_mask=True,
        compact_tree_mask=compact,
    )
    return o


def main():
    assert torch.cuda.is_available(), "needs CUDA (run on B200)"
    device = "cuda"
    dtype = torch.float16
    bs, context, N, n_heads, head_dim = 2, 64, 16, 4, 128

    q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, ancestor = (
        build_scenario(bs, context, N, n_heads, head_dim, device, dtype)
    )
    kw = dict(
        q=q, k=k, v=v, k_buffer=k_buffer, v_buffer=v_buffer,
        qo_indptr=qo_indptr, kv_indptr=kv_indptr, kv_indices=kv_indices,
    )
    o_full = run(False, ancestor, context, N, bs, device, **kw)
    o_compact = run(True, ancestor, context, N, bs, device, **kw)

    max_diff = (o_full - o_compact).abs().max().item()
    print(f"max abs diff = {max_diff:.3e}")
    assert torch.allclose(o_full, o_compact, rtol=1e-3, atol=1e-3), (
        f"compact mask output diverged: max_diff={max_diff}"
    )
    print("PASS: compact QLEN_ONLY mask matches FULL_MASK layout")


if __name__ == "__main__":
    main()
