"""Build-time patch: force DFLASH tree verify to run eager.

The upstream DFLASH decode cuda graph is captured chain-style (custom_mask=None),
so it cannot replay a tree-verify batch (which carries a custom tree mask) — the
copy path hits an unset `raw_num_token`. Make `can_run_graph` return False for
DFLASH batches that carry a custom tree mask so they take the eager flashinfer
path (which consumes spec_info.custom_mask via generate_attn_arg_prefill).

EAGLE is unaffected (gated on is_dflash()). Base decode / draft / prefill still
use cuda graphs; only the tree-verify step runs eager.
"""
import os
import sglang

p = os.path.join(
    os.path.dirname(sglang.__file__),
    "srt/model_executor/runner/decode_cuda_graph_runner.py",
)
s = open(p).read()
needle = "def can_run_graph(self, forward_batch: ForwardBatch):"
assert needle in s, "can_run_graph signature not found; upstream changed"
guard = (
    needle
    + "\n        # DFLASH tree verify carries a custom tree mask the chain graph"
    + "\n        # cannot replay; force eager (base decode/draft graphs unaffected)."
    + "\n        if self.model_runner.spec_algorithm.is_dflash() and (\n"
    + '            getattr(forward_batch.spec_info, "custom_mask", None) is not None\n'
    + "        ):\n            return False"
)
if "DFLASH tree verify carries a custom tree mask" not in s:
    s = s.replace(needle, guard, 1)
    open(p, "w").write(s)
    print("PATCHED can_run_graph for DFLASH tree eager")
else:
    print("can_run_graph already patched")
