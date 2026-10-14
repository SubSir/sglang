"""GPU check: the fused chain verify (GDNAttnBackend._tree_fused_target_verify + the replayed
commit) against the varlen recurrent verify kernel, static and compact layouts.

Compact: q/k/v arrive dense (as from the conv path) with large finite junk in the pad slots,
and the persistent a/b staging buffers start full of junk, so a pad leaking into a real row or
into the committed state shows up as an error.

Per case: outputs of every real token, and the committed SSM state of every request after
accepting a random prefix (reference: the recurrent kernel's per-step intermediate state).
Two layers, so the once-per-verify tree build (ordinal 0) is reused by the second.
Every case also runs with SGLANG_GDN_PACKED_VERIFY off (dense a/b staging, gathered output,
gating and stash copies outside the kernel): outputs must agree bit for bit, and the committed
state to fp32 rounding (the stash's g then comes from fused_gdn_gating instead of the verify kernel). A separate check runs the conv update on packed tokens against the same
tokens zero-padded to the dense layout: output, conv state and per-step windows bit-identical.

  python test/manual/gdn_fused_chain_check.py
"""
import types

import torch

from sglang.kernels.ops.attention.fla.gdn_tree_verify import advance_ssm_states_along_accept_paths
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.kda_backend import ragged_verify_dense_scatter_indices

H, HG, K, V, T, L, POOL, MAX_BS = 16, 8, 128, 128, 8, 2, 40, 16


def backend(packed=True):
    be = object.__new__(GDNAttnBackend)
    be.packed_fused_verify = packed
    be.req_to_token_pool = types.SimpleNamespace(mamba_map={i: i for i in range(L)})
    be.verify_intermediate_state_indices = torch.arange(MAX_BS, device="cuda")
    be.tree_stash, be._chain_parents = None, None
    be._gather_key = be._gather_idx = None
    g = torch.Generator(device="cuda").manual_seed(123)
    junk = lambda: torch.randn(MAX_BS * T + 1, H, generator=g, device="cuda").bfloat16() * 50  # noqa: E731
    be._dense_bufs = {"a": junk(), "b": junk()}
    return be


def case(lens, extra_pad, compact, seed, packed=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rnd = lambda *s, sc=1.0: torch.randn(*s, generator=g, device="cuda") * sc  # noqa: E731
    bs = len(lens)
    lens_t = torch.tensor(lens, device="cuda")
    qsl = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    qsl[1:] = lens_t.cumsum(0)
    n_real = int(qsl[-1])
    seq = n_real + extra_pad  # graph-padding tokens past the real ones
    slots = torch.randperm(POOL - 1, generator=g, device="cuda")[:bs].int() + 1
    accept = torch.tensor([int(torch.randint(0, n, (1,), generator=g, device="cuda")) for n in lens], device="cuda")
    pool0 = rnd(L, POOL, H, V, K, sc=0.1)
    be, pool = backend(packed), pool0.clone()
    idx = ragged_verify_dense_scatter_indices(query_start_loc=qsl, seq_len=seq, draft_token_num=T) if compact else None
    worst_o, worst_s, outs = 0.0, 0.0, []
    ref_committed = pool0.clone()
    for l in range(L):
        layer = types.SimpleNamespace(layer_id=l, num_v_heads=H, head_v_dim=V, num_k_heads=HG, head_k_dim=K,
                                      A_log=rnd(H, sc=0.5), dt_bias=rnd(H, sc=0.5))
        a, b = rnd(seq, H).bfloat16(), rnd(seq, H).bfloat16()
        if compact:  # dense q/k/v with junk pads; the packed real tokens are their dense slots
            n = bs * T
            qd, kd = rnd(1, n, HG, K, sc=20.0).bfloat16(), rnd(1, n, HG, K, sc=20.0).bfloat16()
            vd = rnd(1, n, H, V, sc=20.0).bfloat16()
            real = idx[:n_real]
            for t, s in ((qd, 1.0), (kd, 1.0), (vd, 1.0)):
                t[0, real] = rnd(n_real, *t.shape[2:], sc=s).bfloat16()
            q, k, v = qd[:, real], kd[:, real], vd[:, real]
            out = be._tree_fused_target_verify(layer=layer, batch_size=bs, draft_token_num=T, query=qd, key=kd,
                                               value=vd, a=a, b=b, ssm_states=pool[l], cache_indices=slots,
                                               dense_token_indices=idx, query_start_loc=qsl)
        else:
            q, k = rnd(1, seq, HG, K).bfloat16(), rnd(1, seq, HG, K).bfloat16()
            v = rnd(1, seq, H, V).bfloat16()
            out = be._tree_fused_target_verify(layer=layer, batch_size=bs, draft_token_num=T, query=q, key=k,
                                               value=v, a=a, b=b, ssm_states=pool[l], cache_indices=slots,
                                               dense_token_indices=None, query_start_loc=qsl)
        # Reference: varlen recurrent verify over the real tokens, per-step states kept.
        inter = torch.zeros(MAX_BS, T, H, V, K, device="cuda")
        ref = fused_sigmoid_gating_delta_rule_update(
            A_log=layer.A_log, dt_bias=layer.dt_bias, q=q[:, :n_real], k=k[:, :n_real], v=v[:, :n_real],
            a=a[:n_real], b=b[:n_real], initial_state_source=pool0[l].clone(), initial_state_indices=slots,
            cu_seqlens=qsl, use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            is_kda=False, disable_state_update=True, intermediate_states_buffer=inter,
            intermediate_state_indices=torch.arange(bs, device="cuda", dtype=torch.int32), cache_steps=T,
            retrieve_parent_token=None)
        outs.append(out.reshape(seq, H, V)[:n_real].clone())
        o = out.reshape(seq, H, V)[:n_real].float()
        r = ref.reshape(n_real, H, V).float()
        worst_o = max(worst_o, float((o - r).abs().max() / r.abs().max()))
        for i in range(bs):
            ref_committed[l, slots[i]] = inter[i, accept[i]]
    st = be.tree_stash
    advance_ssm_states_along_accept_paths(st["k"], st["v"], st["g"], st["beta"], pool, slots, st["anc"],
                                          accept, st["k"].shape[2], HG, K, V)
    worst_s = float((pool - ref_committed).abs().max() / ref_committed.abs().max())
    assert all(bool(torch.isfinite(b).all()) for b in be._dense_bufs.values()), "a staging buffer went non-finite"
    untouched = [s for s in range(POOL) if s not in set(slots.tolist())]
    assert torch.equal(pool[:, untouched], pool0[:, untouched]), "commit wrote a slot outside the batch"
    return worst_o, worst_s, outs, pool


def conv_case(lens, extra_pad, seed, dim=384, width=4):
    """Packed-x conv update vs the same tokens zero-padded to [bs, T]: everything bit-identical."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    bs = len(lens)
    qsl = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    qsl[1:] = torch.tensor(lens, device="cuda").cumsum(0)
    n_real = int(qsl[-1])
    x = torch.randn(n_real + extra_pad, dim, generator=g, device="cuda").bfloat16()
    w = torch.randn(dim, width, generator=g, device="cuda").bfloat16()
    bias = torch.randn(dim, generator=g, device="cuda").bfloat16()
    state0 = torch.randn(POOL, dim, width - 1, generator=g, device="cuda").bfloat16()
    slots = torch.randperm(POOL - 1, generator=g, device="cuda")[:bs].int() + 1
    inter_idx = torch.arange(bs, device="cuda", dtype=torch.int32)
    idx = ragged_verify_dense_scatter_indices(query_start_loc=qsl, seq_len=x.shape[0], draft_token_num=T)
    dense = torch.zeros(bs * T + 1, dim, device="cuda", dtype=x.dtype)
    dense.index_copy_(0, idx, x)
    res = []
    for packed in (True, False):
        state = state0.clone()
        inter = torch.full((bs, T, dim, width - 1), 7.0, device="cuda", dtype=x.dtype)
        kw = dict(conv_state_indices=slots, intermediate_conv_window=inter, intermediate_state_indices=inter_idx)
        if packed:
            out = causal_conv1d_update(x, state, w, bias, "silu", query_start_loc=qsl, max_seqlen=T, **kw)
        else:
            out = causal_conv1d_update(dense[: bs * T].view(bs, T, dim).transpose(1, 2), state, w, bias, "silu", **kw)
        res.append((out.transpose(1, 2).reshape(bs * T, dim), state, inter))
    return all(torch.equal(p, d) for p, d in zip(*res))


def main():
    torch.manual_seed(0)
    cases = [("static", [T] * 5, 0, False), ("compact", [1, 3, T, 2, 5, 7, 1, 4], 0, True),
             ("compact+graph pad", [2, T, 1, 6], 3, True), ("compact bs1", [4], 1, True)]
    ok = True
    for seed in range(3):
        for name, lens, pad, compact in cases:
            eo, es, outs, pool = case(lens, pad, compact, seed)
            good = eo < 2e-2 and es < 1e-4
            _, _, outs_g, pool_g = case(lens, pad, compact, seed, packed=False)
            same = all(torch.equal(a, b) for a, b in zip(outs, outs_g))
            d_pool = float((pool - pool_g).abs().max() / pool_g.abs().max())
            good &= same and d_pool < 1e-6
            ok &= good
            print(f"seed {seed} {name:18s} lens {lens}: out rel err {eo:.2e}, committed state rel err {es:.2e}, "
                  f"packed vs off: outputs identical {same}, state diff {d_pool:.1e} {'ok' if good else 'FAIL'}")
        for lens, pad in (([1, 3, T, 2, 5, 7, 1, 4], 0), ([2, T, 1, 6], 3), ([4], 1), ([T] * 3, 0)):
            same = conv_case(lens, pad, seed)
            ok &= same
            print(f"seed {seed} conv packed vs zero-padded dense lens {lens} pad {pad}: identical {same}")
    print("FUSED_CHAIN_OK" if ok else "FUSED_CHAIN_FAIL")


if __name__ == "__main__":
    main()
