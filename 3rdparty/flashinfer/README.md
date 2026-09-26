# flashinfer patches for DFLASH tree verify

Against flashinfer 0.6.18 (the version in `lmsysorg/sglang:nightly-dev-cu13-20260926-1f6ce4b0`).
Apply to the flashinfer source tree with `git apply`.

- `trtllm-gen-spec-dec-tree-mask.patch`: selects trtllm-gen's Custom /
  SlidingWindowCustom generation cubins for a spec-dec tree mask and packs the mask
  (ported from flashinfer PR 3585, closed unmerged). On top of the PR: a
  `spec_dec_packed_mask` argument so the mask is packed once per verify and shared by
  every layer; SlidingWindowCustom takes the persistent scheduler (there is no static
  single-CTA cubin, and without it the runner silently falls back to a folded mask
  with a host sync); the JIT module is renamed `fmha_gen_tree` so the prebuilt
  `fmha_gen` in flashinfer-jit-cache cannot shadow it. sglang's trtllm-mha backend
  uses this path when `flashinfer.decode._pack_trtllm_gen_spec_dec_mask` exists
  and head_dim is 64 or 128; otherwise it verifies trees through XQA.
- `xqa-tree-depth-window.patch`: XQA's spec-dec sliding window starts each draft row
  at its tree depth (popcount of its ancestor mask row - 1) instead of its row index,
  so tree verify is exact past the window. Identical for linear chains.
