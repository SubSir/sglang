"""Patch DDTree PR (#27509) tree-mode bugs, applied at image-build time on the overlaid python.

Usage: python ddtree_fix.py <path-to-srt>   (the sglang/srt dir)

Two independent bugs:

(1) cuda_graph_runner.py — verify-graph token count.
    Capture sizes the verify graph with
    num_tokens_per_bs = get_num_tokens_per_bs_for_target_verify(speculative_num_draft_tokens) = block_size (16).
    But DDTree's verify spec_info uses draft_token_num = tree_budget + 1 (17/33/65).
    Backends build qo_indptr from draft_token_num while the captured q has bs*16 tokens
    -> q.shape[0] != qo_indptr[-1]. Fix: for the TARGET worker size with budget+1.

(2) flashattention_backend.py (fa3) — DDTree cuda-graph metadata-dict mismatch.
    REPLAY-prepare (init_forward_metadata_out_graph) routes DDTree target-verify to the
    topk>1 dicts (target_verify_metadata_topk_normal/_expand) because spec_info has tree_budget.
    But init_cuda_graph_state creates those dicts only under `if self.topk > 1:`, and DDTree
    forces eagle-topk=1 -> AttributeError at capture. Capture (_bind_metadata_buffers) and
    replay-fill (_apply_cuda_graph_metadata) also gate the topk (cascade/tree-mask) path on
    `self.topk <= 1`, so DDTree would silently use a plain-causal layout (wrong tree verify).
    Fix: make all four sites treat DDTree-target like topk>1 (use the cascade path), and
    size the cascade buffers/mask-extraction with budget+1 (= self._verify_ndt) instead of
    block_size (= self.speculative_num_draft_tokens, which for DDTree stays 16).
"""
import re
import sys
import os

SRT = sys.argv[1].rstrip("/")


def patch_cuda_graph_runner(path):
    src = open(path).read()
    old = """            self.num_tokens_per_bs = (
                model_runner.spec_algorithm.get_num_tokens_per_bs_for_target_verify(
                    self.speculative_num_draft_tokens, model_runner.is_draft_worker
                )
            )"""
    new = old + """
            # DDTREE-FIX: the TARGET verify graph must be sized to
            # max_tree_nodes (budget+1) = the draft_token_num used by
            # DDTreeVerifyInput at runtime. The DRAFT worker still runs the
            # draft model over block_size tokens, so leave it untouched.
            if (
                model_runner.spec_algorithm.is_ddtree()
                and not model_runner.is_draft_worker
            ):
                _b = getattr(model_runner.server_args, "speculative_ddtree_budget", None)
                if _b is not None:
                    self.num_tokens_per_bs = int(_b) + 1"""
    assert old in src, "cuda_graph_runner anchor not found"
    open(path, "w").write(src.replace(old, new))
    print("DDTREE-FIX (1) applied:", path)


def _replace_in_block(lines, start_idx, num_draft="self.speculative_num_draft_tokens",
                      repl="self._verify_ndt"):
    """Replace num_draft -> repl from start_idx until the block dedents below its indent."""
    base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
    i = start_idx
    n = 0
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()
        if i > start_idx and stripped and (len(ln) - len(ln.lstrip())) <= base_indent:
            break
        if num_draft in ln:
            lines[i] = ln.replace(num_draft, repl)
            n += ln.count(num_draft)
        i += 1
    return n


def patch_fa3(path):
    src = open(path).read()

    # --- A) __init__: define _verify_ndt = budget+1 for ddtree target, else block_size ---
    anchor = """        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )"""
    assert anchor in src, "fa3 __init__ anchor not found"
    src = src.replace(anchor, anchor + """
        # DDTREE-FIX: target-verify window for DDTree = budget+1 (max_tree_nodes),
        # which differs from speculative_num_draft_tokens (draft block_size).
        self._ddtree_target_verify = (
            model_runner.spec_algorithm.is_ddtree() and not model_runner.is_draft_worker
        )
        _ddt_b = getattr(model_runner.server_args, "speculative_ddtree_budget", None)
        self._verify_ndt = (
            int(_ddt_b) + 1
            if (self._ddtree_target_verify and _ddt_b is not None)
            else self.speculative_num_draft_tokens
        )""")

    # --- B) init_cuda_graph_state: create topk verify dicts for ddtree too, sized budget+1 ---
    # The block creating target_verify_metadata_topk_normal/_expand/_swa is guarded by
    # `if self.topk > 1:` immediately preceding the topk_normal dict.
    guard = """        if self.topk > 1:
            self.target_verify_metadata_topk_normal = {"""
    new_guard = """        if self.topk > 1 or self._ddtree_target_verify:
            self.target_verify_metadata_topk_normal = {"""
    assert guard in src, "fa3 init topk-verify guard not found"
    src = src.replace(guard, new_guard)

    # Now scope-replace speculative_num_draft_tokens -> _verify_ndt inside that creation block.
    lines = src.split("\n")
    gi = next(i for i, l in enumerate(lines)
              if l == "        if self.topk > 1 or self._ddtree_target_verify:"
              and "target_verify_metadata_topk_normal" in lines[i + 1])
    _replace_in_block(lines, gi)
    src = "\n".join(lines)

    # --- C) _bind_metadata_buffers (capture): route ddtree to the topk (else) branch ---
    # and size with _verify_ndt. The branch is `if self.topk <= 1:` under target_verify
    # that reads self.target_verify_metadata["cache_seqlens"].
    cap_guard = """            if self.topk <= 1:
                metadata.cache_seqlens_int32 = self.target_verify_metadata[
                    "cache_seqlens"
                ][:bs]
                metadata.max_seq_len_q = self.speculative_num_draft_tokens"""
    cap_new = """            if self.topk <= 1 and not self._ddtree_target_verify:
                metadata.cache_seqlens_int32 = self.target_verify_metadata[
                    "cache_seqlens"
                ][:bs]
                metadata.max_seq_len_q = self.speculative_num_draft_tokens"""
    assert cap_guard in src, "fa3 capture verify guard not found"
    src = src.replace(cap_guard, cap_new)
    # scope-replace num_draft in the capture else-branch (starts at the matching `else:`)
    lines = src.split("\n")
    ci = next(i for i, l in enumerate(lines)
              if l.strip() == "else:"
              and "topk>1: two (or three" in (lines[i + 1] if i + 1 < len(lines) else ""))
    _replace_in_block(lines, ci)
    src = "\n".join(lines)

    # --- D) _apply_cuda_graph_metadata (replay-fill): same routing + sizing ---
    rep_guard = """        elif forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata = self.target_verify_metadata[bs]"""
    rep_new = """        elif forward_mode.is_target_verify():
            if self.topk <= 1 and not self._ddtree_target_verify:
                metadata = self.target_verify_metadata[bs]"""
    assert rep_guard in src, "fa3 replay verify guard not found"
    src = src.replace(rep_guard, rep_new)
    lines = src.split("\n")
    ri = next(i for i, l in enumerate(lines)
              if l.strip() == "else:"
              and "When topk > 1, we need two specific target verify" in
                  (lines[i + 1] if i + 1 < len(lines) else ""))
    _replace_in_block(lines, ri)
    src = "\n".join(lines)

    open(path, "w").write(src)
    print("DDTREE-FIX (2) applied:", path)


patch_cuda_graph_runner(os.path.join(SRT, "model_executor/cuda_graph_runner.py"))
patch_fa3(os.path.join(SRT, "layers/attention/flashattention_backend.py"))
