"""Varlen chain GDN verify (SGLANG_GDN_CHAIN_VERIFY) vs an fp32 torch reference.

Checks, on mixed lengths 0..T (0 = graph padding):
  1. outputs of fused_sigmoid_gating_delta_rule_update(cu_seqlens, disable_state_update,
     stash=...) against the gated delta rule, packed rows only; the state pool untouched;
  2. the stash rows written for real tokens only (pad slots keep a sentinel);
  3. the commit replay (advance_ssm_states_along_accept_paths, recurrent_stash=True) equals
     the stock recurrent kernel run over the accepted prefix with a state write-back;
  4. packed conv output (causal_conv1d_update out_packed=True) equals the dense conv at
     every real row, and the conv window is written for real slots only.

GPU:  python test/manual/gdn_chain_verify_check.py [--bs 128 --hv 48 --h 16 --k 128]
CPU:  TRITON_INTERPRET=1 python test/manual/gdn_chain_verify_check.py   (small default dims)
"""

import argparse
import os

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.gdn_tree_verify import (
    advance_ssm_states_along_accept_paths,
    alloc_tree_structure_buffers,
    build_tree_ancestor_masks,
    build_tree_structure_into,
    tree_gdn_triton_verify,
    tree_mask_capacity,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

SENT = 7.0


def reference(q, k, v, a, b, A_log, dt_bias, h0, lens, scale):
    """fp32 gated delta rule per request; h0 [N, HV, V, K]; q/k [n, H, K], v [n, HV, V]."""
    HV, H = v.shape[1], q.shape[1]
    G = HV // H
    out = torch.zeros(v.shape, dtype=torch.float32)
    states = []
    start = 0
    for i, L in enumerate(lens):
        S = h0[i].float().transpose(-1, -2).clone()  # [HV, K, V]
        for t in range(start, start + L):
            qt = q[t].float().repeat_interleave(G, 0)
            kt = k[t].float().repeat_interleave(G, 0)
            qt = qt / torch.sqrt((qt * qt).sum(-1, keepdim=True) + 1e-6) * scale
            kt = kt / torch.sqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
            g = -A_log.float().exp() * F.softplus(a[t].float() + dt_bias.float())
            beta = torch.sigmoid(b[t].float()).to(b.dtype).float()
            S = S * g.exp()[:, None, None]
            vt = beta[:, None] * (v[t].float() - torch.einsum("hkv,hk->hv", S, kt))
            S = S + kt[:, :, None] * vt[:, None, :]
            out[t] = torch.einsum("hkv,hk->hv", S, qt)
        states.append(S.transpose(-1, -2))
        start += L
    return out, states


def main():
    ap = argparse.ArgumentParser()
    interp = os.environ.get("TRITON_INTERPRET") == "1"
    ap.add_argument("--bs", type=int, default=5 if interp else 128)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--h", type=int, default=2 if interp else 16)
    ap.add_argument("--hv", type=int, default=6 if interp else 48)
    ap.add_argument("--k", type=int, default=32 if interp else 128)
    ap.add_argument("--state-dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cpu" if interp else "cuda"
    torch.manual_seed(args.seed)
    bs, T, H, HV, K = args.bs, args.T, args.h, args.hv, args.k
    V = K
    sdt = getattr(torch, args.state_dtype)
    dt = torch.bfloat16

    lens = torch.randint(0, T + 1, (bs,)).tolist()
    lens[0], lens[-1] = T, 0
    n = sum(lens)
    n_pad = n + 3  # graph-padding tokens no request covers
    qsl = torch.tensor([0] + lens).cumsum(0).to(torch.int32)

    pool = bs + 4
    slots = torch.randperm(pool)[:bs].to(torch.int32)
    ssm = (torch.randn(pool, HV, V, K) * 0.1).to(sdt)
    q = torch.randn(n_pad, H, K, dtype=dt)
    k = torch.randn(n_pad, H, K, dtype=dt)
    v = torch.randn(n_pad, HV, V, dtype=dt)
    a = torch.randn(n_pad, HV, dtype=dt)
    b = torch.randn(n_pad, HV, dtype=dt)
    A_log = torch.randn(HV) * 0.5
    dt_bias = torch.randn(HV) * 0.5
    scale = K**-0.5

    ref_o, ref_states = reference(
        q[:n], k[:n], v[:n], a[:n], b[:n], A_log, dt_bias, ssm[slots.long()], lens, scale
    )

    to = lambda *ts: [t.to(dev) for t in ts]  # noqa: E731
    q_d, k_d, v_d, a_d, b_d, ssm_d, slots_d, qsl_d, A_d, dtb_d = to(
        q, k, v, a, b, ssm, slots, qsl, A_log, dt_bias
    )
    ssm_before = ssm_d.clone()
    rows = bs + 2
    stash = (
        torch.full((rows, T, H * K), SENT, dtype=dt, device=dev),
        torch.full((rows, T, HV * V), SENT, dtype=dt, device=dev),
        torch.full((rows, T, HV), SENT, dtype=torch.float32, device=dev),
        torch.full((rows, T, HV), SENT, dtype=torch.float32, device=dev),
    )
    out = torch.zeros(n_pad, HV * V, dtype=dt, device=dev)
    o = fused_sigmoid_gating_delta_rule_update(
        A_log=A_d, a=a_d, dt_bias=dtb_d, softplus_beta=1.0, softplus_threshold=20.0,
        q=q_d.unsqueeze(0), k=k_d.unsqueeze(0), v=v_d.unsqueeze(0), b=b_d,
        initial_state_source=ssm_d, initial_state_indices=slots_d, scale=scale,
        use_qk_l2norm_in_kernel=True, cu_seqlens=qsl_d, disable_state_update=True,
        stash=stash, output=out,
    )
    o = o.float().cpu().view(n_pad, HV, V)
    err = (o[:n] - ref_o).abs()
    rel = err.max() / ref_o.abs().max()
    print(f"[1] output vs fp32 ref: max abs {err.max():.3e}  rel {rel:.3e}  (n={n}, lens={lens[:16]}...)")
    assert rel < 2e-2, rel
    assert (o[n:] == 0).all(), "uncovered rows must keep the output buffer's zeros"
    assert torch.equal(ssm_d, ssm_before), "verify must not write the state pool"

    # [2] stash: real slots written, pad slots untouched
    for i, L in enumerate(lens):
        assert (stash[1][i, L:] == SENT).all() and (stash[2][i, L:] == SENT).all()
        if L:
            s0 = int(qsl[i])
            assert torch.equal(stash[1][i, :L].cpu(), v[s0 : s0 + L].reshape(L, -1))
            assert torch.equal(stash[0][i, :L].cpu(), k[s0 : s0 + L].reshape(L, -1))
    print("[2] stash: real slots exact, pad slots untouched")

    # [3] commit replay along a random accepted prefix vs the stock kernel with write-back
    leaf = torch.tensor([int(torch.randint(0, L, ())) if L else 0 for L in lens])
    acc = [int(x) + 1 if L else 0 for x, L in zip(leaf, lens)]
    commit_slots = slots_d.clone()
    commit_slots[torch.tensor(lens, device=dev) == 0] = -1  # graph padding commits nothing
    parents = (torch.arange(T, device=dev) - 1).clamp(min=0).expand(bs, T).contiguous()
    cap = tree_mask_capacity(T)
    anc = torch.zeros(bs, cap, cap // 64, dtype=torch.int64, device=dev)
    build_tree_ancestor_masks(parents, T, out=anc)
    ssm_commit = ssm_before.clone().unsqueeze(0)
    advance_ssm_states_along_accept_paths(
        stash[0].unsqueeze(0), stash[1].unsqueeze(0), stash[2].unsqueeze(0), stash[3].unsqueeze(0),
        ssm_commit, commit_slots, anc, leaf.to(dev), T, H, K, V, recurrent_stash=True,
    )
    # stock kernel over the accepted prefix, writing the state back
    pre_idx = torch.cat([torch.arange(int(qsl[i]), int(qsl[i]) + acc[i]) for i in range(bs)]).long()
    pre_qsl = torch.tensor([0] + acc).cumsum(0).to(torch.int32).to(dev)
    ssm_stock = ssm_before.clone()
    fused_sigmoid_gating_delta_rule_update(
        A_log=A_d, a=a_d[pre_idx], dt_bias=dtb_d, softplus_beta=1.0, softplus_threshold=20.0,
        q=q_d[pre_idx].unsqueeze(0), k=k_d[pre_idx].unsqueeze(0), v=v_d[pre_idx].unsqueeze(0),
        b=b_d[pre_idx], initial_state_source=ssm_stock, initial_state_indices=commit_slots,
        scale=scale, use_qk_l2norm_in_kernel=True, cu_seqlens=pre_qsl,
    )
    d = (ssm_commit[0].float() - ssm_stock.float()).abs().max().item()
    print(f"[3] commit replay vs stock recurrent write-back: max abs {d:.3e}")
    assert d == 0.0, d

    # [4] packed conv vs dense conv
    D, W = 3 * 64, 4
    cdt = torch.float32 if interp else dt  # the interpreter's bf16 emulation reads garbage in this kernel
    x = torch.randn(n_pad, D, dtype=cdt, device=dev)
    w = torch.randn(D, W, dtype=cdt, device=dev) * 0.3
    bias = torch.randn(D, dtype=cdt, device=dev) * 0.1
    conv0 = torch.randn(pool, W - 1, D, dtype=cdt, device=dev).transpose(1, 2)  # pool layout
    win_p = torch.full((rows, T, D, W - 1), SENT, dtype=cdt, device=dev)
    win_d = torch.full_like(win_p, SENT)
    cs_p = conv0.transpose(1, 2).clone().transpose(1, 2)
    cs_d = conv0.transpose(1, 2).clone().transpose(1, 2)
    inter_idx = torch.arange(bs, dtype=torch.int32, device=dev)
    yp = causal_conv1d_update(
        x, cs_p, w, bias, "silu", conv_state_indices=slots_d, intermediate_conv_window=win_p,
        intermediate_state_indices=inter_idx, query_start_loc=qsl_d, max_seqlen=T, out_packed=True,
    )
    yd = causal_conv1d_update(
        x, cs_d, w, bias, "silu", conv_state_indices=slots_d, intermediate_conv_window=win_d,
        intermediate_state_indices=inter_idx, query_start_loc=qsl_d, max_seqlen=T,
    )  # (bs, D, T) dense
    yd = yd.transpose(1, 2)
    for i, L in enumerate(lens):
        s0 = int(qsl[i])
        assert torch.equal(yp[s0 : s0 + L], yd[i, :L]), (i, L, (yp[s0 : s0 + L].float() - yd[i, :L].float()).abs().amax(-1))
        assert torch.equal(win_p[i, :L], win_d[i, :L]), i
        assert (win_p[i, L:] == SENT).all(), i
    print("[4] packed conv == dense conv at real rows; window written for real slots only")

    # [5] full length: chain kernel vs the fused tree kernel (fp32 sigmoid beta, so ~bf16 apart)
    nf = bs * T
    qf, kf, vf = q_d[:1].repeat(nf, 1, 1), k_d[:1].repeat(nf, 1, 1), v_d[:1].repeat(nf, 1, 1)
    qf, kf, vf = torch.randn_like(qf), torch.randn_like(kf), torch.randn_like(vf)
    af, bf = torch.randn(nf, HV, dtype=dt, device=dev), torch.randn(nf, HV, dtype=dt, device=dev)
    qsl_f = (torch.arange(bs + 1, dtype=torch.int32, device=dev) * T)
    oc = fused_sigmoid_gating_delta_rule_update(
        A_log=A_d, a=af, dt_bias=dtb_d, softplus_beta=1.0, softplus_threshold=20.0,
        q=qf.unsqueeze(0), k=kf.unsqueeze(0), v=vf.unsqueeze(0), b=bf,
        initial_state_source=ssm_d, initial_state_indices=slots_d, scale=scale,
        use_qk_l2norm_in_kernel=True, cu_seqlens=qsl_f, disable_state_update=True,
    ).float().view(nf, HV, V)
    tree = build_tree_structure_into(parents, alloc_tree_structure_buffers(bs, T, T - 1, dev))
    ot = tree_gdn_triton_verify(
        A_d, af.view(bs, T, HV), dtb_d, 1.0, 20.0, qf.view(bs, T, H, K), kf.view(bs, T, H, K),
        vf.view(bs, T, HV, V), bf.view(bs, T, HV), ssm_d, slots_d.long(), tree, scale=scale,
        use_qk_l2norm_in_kernel=True, precision="ieee",
    ).float().view(nf, HV, V)
    ref_f, _ = reference(qf.cpu(), kf.cpu(), vf.cpu(), af.cpu(), bf.cpu(), A_log, dt_bias, ssm[slots.long()], [T] * bs, scale)
    print(f"[5] full length: chain vs tree max abs {(oc - ot).abs().max():.3e}; "
          f"chain vs ref {(oc.cpu() - ref_f).abs().max():.3e}, tree vs ref {(ot.cpu() - ref_f).abs().max():.3e} "
          f"(ref max {ref_f.abs().max():.3f})")
    print("ALL OK")


if __name__ == "__main__":
    main()
