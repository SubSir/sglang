"""Build-time patch: make the DFLASH verify CUDA graph mask-capable for tree verify.

Upstream's DFLASH verify graph is captured chain-style (custom_mask=None for
flashinfer), so a tree-verify batch (which carries a tree mask) can't be replayed
and the verify runs maskless -> wrong target predictions -> tiny accept length.

The v1 fork (which runs tree verify correctly under CUDA graph) forces
build_custom_mask=True for DFLASH tree in get_spec_info, so the captured graph has
a custom-mask buffer and mask-capable flashinfer wrappers. Replicate that here.
"""
import os as _os
import sglang

p = _os.path.join(
    _os.path.dirname(sglang.__file__),
    "srt/model_executor/runner/decode_cuda_graph_runner.py",
)
s = open(p).read()

# Ensure `os` is importable inside the module (insert after the first stdlib import,
# which sits after `from __future__`).
if "\nimport os\n" not in s:
    assert "\nimport contextlib\n" in s, "expected `import contextlib` anchor"
    s = s.replace("\nimport contextlib\n", "\nimport contextlib\nimport os\n", 1)

needle = (
    "            _, build_custom_mask = resolve_dflash_verify_mask_policy(\n"
    "                self.model_runner.attn_backend\n"
    "            )\n"
)
patch = needle + (
    "            # DFLASH tree verify needs a mask-capable verify graph (mirror v1 fork).\n"
    "            if (\n"
    "                not self.model_runner.is_draft_worker\n"
    "                and os.environ.get(\"SGLANG_DFLASH_TREE_VERIFY\", \"0\") == \"1\"\n"
    "            ):\n"
    "                build_custom_mask = True\n"
)
assert needle in s, "DFLASH get_spec_info mask-policy block not found; upstream changed"
if "DFLASH tree verify needs a mask-capable verify graph" not in s:
    s = s.replace(needle, patch, 1)

# Also carry the tree topk into the capture spec_info so the verify graph metadata
# matches tree mode (the chain default topk=1 differs). The DFLASH constructor in
# get_spec_info passes custom_mask via a ternary; add topk right after it.
mask_ctor = (
    "                custom_mask=(\n"
    "                    None\n"
    "                    if (self.model_runner.is_draft_worker or not build_custom_mask)\n"
    "                    else self.buffers.custom_mask\n"
    "                ),\n"
)
mask_ctor_with_topk = mask_ctor + (
    "                topk=(\n"
    "                    self.model_runner.server_args.speculative_eagle_topk\n"
    "                    if build_custom_mask\n"
    "                    else 1\n"
    "                ),\n"
)
if mask_ctor in s and "speculative_eagle_topk\n                    if build_custom_mask" not in s:
    s = s.replace(mask_ctor, mask_ctor_with_topk, 1)

open(p, "w").write(s)
print("PATCHED get_spec_info: DFLASH tree verify graph is mask-capable")
