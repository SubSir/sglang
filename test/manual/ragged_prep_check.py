"""GPU check, bit-exact: the fused ragged verify prep against the unfused paths on random batches.
  * build_ragged_verify_window_fused vs build_ragged_verify_window (torch) and the unfused Triton builder;
  * scatter_compact_to_strided_into (one launch) vs scatter_compact_to_strided (torch);
  * build_ragged_target_verify_geometry on CUDA (fused kernel) vs its torch formula;
  * padded_to_bucket served from the layout's cache is the same object, and equals a fresh pad.
Batches include length-1 requests, a full-width one, and graph padding past the real tokens.

  python test/manual/ragged_prep_check.py
"""
import types

import torch

from sglang.kernels.ops.speculative.dspark import dspark_verify_window as W
from sglang.srt.environ import envs
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout, build_ragged_target_verify_geometry

T, POOL, CTX = 16, 300, 6000


def batch(bs, g):
    lens = torch.randint(1, T + 1, (bs,), generator=g)
    lens[0] = 1
    lens[-1] = T
    total = int(lens.sum())
    pad = int(torch.randint(0, 9, (1,), generator=g))
    layout = RaggedVerifyLayout.from_verify_lens_device(verify_lens=lens.int().cuda(), graph_num_tokens=total + pad)
    seq = torch.randint(10, CTX - T, (bs,), generator=g).cuda()
    pool_idx = torch.randperm(POOL, generator=g)[:bs].cuda()
    req_to_token = torch.randint(0, 10**6, (POOL, CTX), generator=g).cuda()
    b = types.SimpleNamespace(seq_lens=seq, req_pool_indices=pool_idx)
    mr = types.SimpleNamespace(req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token))
    block_ids = torch.randint(0, 150000, (bs, T), generator=g).cuda()
    drafts = torch.randint(0, 150000, (bs, T - 1), generator=g).cuda()
    return lens, layout, b, mr, block_ids, drafts


def main():
    g = torch.Generator().manual_seed(0)
    ok = True
    for bs in (1, 2, 7, 32, 64, 200):
        for _ in range(5):
            lens, layout, b, mr, block_ids, drafts = batch(bs, g)
            kw = dict(batch=b, layout=layout, draft_block_ids=block_ids, draft_tokens=drafts, bs=bs, device="cuda",
                      model_runner=mr)
            fused = W.build_ragged_verify_window_fused(**kw)
            ref = W.build_ragged_verify_window(**kw, verify_num_draft_tokens=T)
            with envs.SGLANG_SPEC_FUSED_VERIFY_WINDOW.override(False):
                unfused = W.build_ragged_verify_window_triton(**kw, verify_num_draft_tokens=T)
            for name in ("positions", "verify_cache_loc", "verify_ids"):
                a, r, u = getattr(fused, name), getattr(ref, name), getattr(unfused, name)
                good = torch.equal(a.cpu(), r.long().cpu()) and torch.equal(a.cpu(), u.long().cpu())
                ok &= good
                if not good:
                    print(f"FAIL window {name} bs={bs}")
            dims = [(1, torch.int64), (4096, torch.bfloat16)] + ([(151936, torch.float32)] if bs <= 32 else [])
            for dim, dt in dims:
                compact = (torch.randn(layout.graph_num_tokens, dim, generator=g) * 100).to(dt).cuda()
                out = torch.empty(bs * T, dim, dtype=dt, device="cuda")
                W.scatter_compact_to_strided_into(compact=compact, verify_lens=layout.verify_lens, out=out, stride=T,
                                                  fill_value=0)
                ref_s = W.scatter_compact_to_strided(compact=compact, layout=layout, fill_value=0,
                                                     verify_num_draft_tokens=T)
                good = torch.equal(out.cpu(), ref_s.cpu())
                ok &= good
                if not good:
                    print(f"FAIL scatter dim={dim} bs={bs}")
            geo = build_ragged_target_verify_geometry(seq_lens=b.seq_lens, layout=layout)
            cache_ref = (b.seq_lens + layout.verify_lens).to(torch.int32)
            cuk_ref = torch.nn.functional.pad(torch.cumsum(cache_ref, 0, dtype=torch.int32), (1, 0))
            good = torch.equal(geo.cache_seqlens_int32, cache_ref) and torch.equal(geo.cu_seqlens_k, cuk_ref)
            ok &= good
            if not good:
                print(f"FAIL geometry bs={bs}")
            p1 = layout.padded_to_bucket(padded_bs=bs + 3)
            p2 = layout.padded_to_bucket(padded_bs=bs + 3)
            fresh = RaggedVerifyLayout.from_verify_lens_device(verify_lens=layout.verify_lens,
                                                               graph_num_tokens=layout.graph_num_tokens)
            p3 = fresh.padded_to_bucket(padded_bs=bs + 3)
            good = p1 is p2 and torch.equal(p1.verify_lens, p3.verify_lens) and torch.equal(p1.qo_indptr_device,
                                                                                            p3.qo_indptr_device)
            ok &= good
            if not good:
                print(f"FAIL padded cache bs={bs}")
        print(f"bs {bs}: {'ok' if ok else 'FAIL'}", flush=True)
    print("RAGGED_PREP_OK" if ok else "RAGGED_PREP_FAIL")


if __name__ == "__main__":
    main()
