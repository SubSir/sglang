"""Losslessness check for the generalized DFlash fused tree-build Triton kernel.

Compares the fused kernel (dflash_tree_verify_select_topk4_fused) against the eager
oracle (_select_top_k_tokens_no_hidden loop, exactly as build_tree_verify_tokens uses it)
for topk in (4, 8). Asserts score_list_cat / ss_token_list / parent_list match.

# ponytail: needs CUDA — the fused kernel is Triton/CUDA-only; run on a GPU box.
Run: python test_fused_topk.py
"""

import torch

from sglang.srt.speculative.dflash_utils import _select_top_k_tokens_no_hidden
from sglang.srt.speculative.triton_ops.dflash_tree_expand_topk import (
    dflash_tree_verify_select_topk4_fused,
)


def eager_reference(topk_probs, topk_ids, topk):
    """Mirror of build_tree_verify_tokens' eager branch: returns the three cat'd tensors."""
    bs, num_steps, _ = topk_probs.shape
    device = topk_probs.device

    expanded_topk_probs = topk_probs[:, 1:].repeat_interleave(topk, dim=0)
    expanded_topk_ids = topk_ids[:, 1:].repeat_interleave(topk, dim=0)

    score_list, token_list, parents_list = [], [], []
    scores = None
    for i in range(num_steps):
        if i == 0:
            step_p, step_ids = topk_probs[:, 0], topk_ids[:, 0]
        else:
            step_p, step_ids = expanded_topk_probs[:, i - 1], expanded_topk_ids[:, i - 1]
        scores, tree_info = _select_top_k_tokens_no_hidden(i, step_p, step_ids, scores, topk)
        score_list.append(tree_info[0])
        token_list.append(tree_info[1])
        parents_list.append(tree_info[2])

    score_list_cat = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        parent_list = torch.empty((bs, 0), dtype=torch.long, device=device)
    return score_list_cat, ss_token_list, parent_list


def run_one(topk, bs=3, num_steps=15, device="cuda", seed=0):
    torch.manual_seed(seed)
    # probs: softmax over topk so each row sums to 1.
    logits = torch.randn(bs, num_steps, topk, device=device, dtype=torch.float32)
    topk_probs = torch.softmax(logits, dim=-1)
    # distinct ids per (batch, step) row.
    vocab = 10000
    topk_ids = torch.stack(
        [
            torch.stack([torch.randperm(vocab, device=device)[:topk] for _ in range(num_steps)])
            for _ in range(bs)
        ]
    ).to(torch.int64)

    f_scores, f_tokens, f_parents = dflash_tree_verify_select_topk4_fused(topk_probs, topk_ids)
    e_scores, e_tokens, e_parents = eager_reference(topk_probs, topk_ids, topk)

    ok = True
    if not torch.allclose(f_scores, e_scores, rtol=1e-5, atol=1e-6):
        ok = False
        print(f"  scores mismatch: max abs diff {(f_scores - e_scores).abs().max().item():.3e}")
    if not torch.equal(f_tokens, e_tokens.to(f_tokens.dtype)):
        ok = False
        print(f"  tokens mismatch: {(f_tokens != e_tokens.to(f_tokens.dtype)).sum().item()} elems")
    if not torch.equal(f_parents, e_parents):
        ok = False
        print(f"  parents mismatch: {(f_parents != e_parents).sum().item()} elems")
    print(f"topk={topk}: {'PASS' if ok else 'FAIL'} "
          f"(scores {tuple(f_scores.shape)}, tokens {tuple(f_tokens.shape)}, parents {tuple(f_parents.shape)})")
    return ok


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs CUDA"
    all_ok = True
    for topk in (4, 8):
        all_ok &= run_one(topk)
    print("ALL PASS" if all_ok else "SOME FAILED")
    raise SystemExit(0 if all_ok else 1)
