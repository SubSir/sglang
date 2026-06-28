"""vLLM-DFlash (linear) vs vLLM-JetSpec (tree) in the SAME vLLM engine, on B200.

vllm-jetspec's spec method="dflash" is unified: --tree-width 1 == linear DFlash (vLLM's
DFlash), --tree-width >1 == JetSpec parallel tree. Same draft head (JetSpec/jetspec-qwen3-8b),
same engine, so the delta isolates the tree contribution inside vLLM (parallel to the HF-reference
Task 1). The fork adds NO new .cu kernels (tree attn is triton/optimus-cutedsl, both JIT), so the
documented VLLM_USE_PRECOMPILED install fetches a prebuilt vLLM wheel + overlays the fork's python.

  modal run modal_vllm_jetspec.py::bench       # ar + dflash(tw1) + jetspec(tw7) on gsm8k
  modal run modal_vllm_jetspec.py::pull
"""

import os
import modal

app = modal.App("vllm-jetspec-bench")
vol = modal.Volume.from_name("vllm-jetspec-results", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "curl")
    .run_commands(
        "echo vj4 > /tmp/bt",
        # Overlay approach (same as our SGLang v2 port): install STOCK vllm to get a working
        # vllm._C (+ all B200 runtime deps), then copy the fork's python vllm/ over it. The
        # fork adds NO .cu kernels, so stock's compiled _C.so stays valid; only the fork's
        # .py (spec_decode/dflash, tree_attn backend) replace stock. Avoids VLLM_USE_PRECOMPILED's
        # commit-resolution which silently failed to extract _C on a shallow clone.
        "pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly || pip install -U vllm",
        "python -c 'import vllm._C; print(\"STOCK_C_OK\", __import__(\"vllm\").__version__)'",
        "pip install datasets",
        "git clone --depth 1 https://github.com/JetSpec-project/vllm-jetspec /root/vllm-jetspec",
        # overlay the fork's python over the installed stock vllm package (keep stock .so files)
        "SITE=$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))') && "
        "echo \"stock vllm at $SITE\" && "
        "cp -rf /root/vllm-jetspec/vllm/. \"$SITE/\" && "
        "python -c 'import vllm._C; from vllm import LLM; print(\"OVERLAY_OK\")'",
    )
)


@app.function(gpu="B200", timeout=10800, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def run(mode, tree_width, max_tree_budget, tag, prompt_set="gsm8k", max_samples=16,
        target="Qwen/Qwen3-8B", draft="JetSpec/jetspec-qwen3-8b"):
    import subprocess
    env = dict(os.environ)
    env["VLLM_USE_V1"] = "1"
    args = [
        "python", "/root/vllm-jetspec/examples/offline_inference/dflash_profiling.py",
        "--prompt-set", prompt_set, "--mode", mode, "--head-type", "causal",
        "--model", target, "--draft-model", draft,
        "--max-tokens", "1024", "--block-size", "16",
        "--attention-backend", "FLASH_ATTN",
        "--tree-width", str(tree_width), "--max-tree-budget", str(max_tree_budget),
        "--tree-draft", "accum_logp", "--tree-attn-kernel", "triton",
        "--tree-kv-layout", "physical",
        "--tp-sizes", "1", "--batch-sizes", "1", "--gpu-memory-utilization", "0.85",
        "--max-num-batched-tokens", "51200", "--max-samples", str(max_samples),
        "--max-num-seqs", "1", "--num-runs", "2", "--num-warmup-runs", "1",
        "--profiler", "none",
    ]
    print(">>>", " ".join(args), flush=True)
    p = subprocess.run(args, cwd="/root", env=env, capture_output=True, text=True)
    out = p.stdout + "\n===STDERR(tail)===\n" + "\n".join(p.stderr.splitlines()[-50:])
    with open(f"/results/{tag}.txt", "w") as f:
        f.write(out)
    vol.commit()
    print(out[-5000:], flush=True)
    return out[-1500:]


@app.local_entrypoint()
def bench(prompt_set: str = "gsm8k", max_samples: int = 16):
    # Only linear-vs-tree (no AR): vLLM-DFlash (linear, tw1) vs vLLM-JetSpec (tree, tw7/budget128)
    run.spawn("dflash", 1, 16, f"dflash_linear_{prompt_set}", prompt_set, max_samples)
    run.spawn("dflash", 7, 128, f"jetspec_tree_{prompt_set}", prompt_set, max_samples)
    print("launched: dflash(tw1 linear) + jetspec(tw7 tree) on", prompt_set)


@app.function(image=image, volumes={"/results": vol})
def _list():
    import os as o
    return {fn: open(f"/results/{fn}").read()
            for fn in sorted(o.listdir("/results")) if fn.endswith(".txt")}


@app.local_entrypoint()
def pull():
    os.makedirs("vllm_jetspec_results", exist_ok=True)
    for fn, c in _list.remote().items():
        with open(f"vllm_jetspec_results/{fn}", "w") as f:
            f.write(c)
        print(f"\n===== {fn} =====\n" + "\n".join(c.splitlines()[-25:]))
