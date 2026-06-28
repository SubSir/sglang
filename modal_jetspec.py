"""Benchmark JetSpec (hao-ai-lab parallel tree drafting) on B200 vs DFlash.

JetSpec ships its own DFlash baseline (bench/reference/benchmark.py --include-dflash-baseline)
and a wall-clock engine benchmark (bench/engine/tps_walltime.py). Both run Qwen3-8B with the
trained JetSpec draft head (JetSpec/jetspec-qwen3-8b).

  modal run modal_jetspec.py::reference   # JetSpec tree vs DFlash: accept len + speedup
  modal run modal_jetspec.py::engine      # JetSpec engine tok/s (cuda-graph drafter)
  modal run modal_jetspec.py::pull        # fetch persisted results
"""

import os
import modal

app = modal.App("jetspec-bench")
vol = modal.Volume.from_name("jetspec-results", create_if_missing=True)

# B200-ready torch (cu12.8) + triton 3.6 + flashinfer already prebuilt; just add JetSpec.
image = (
    modal.Image.from_registry("lmsysorg/sglang:nightly-dev-cu12-20260627-13b5bd96")
    .run_commands(
        "echo js1 > /tmp/bt",
        "git clone --depth 1 https://github.com/hao-ai-lab/JetSpec /root/JetSpec",
        # bench+kernel extras (datasets, triton); skip flash-attn (sdpa base + triton tree
        # avoids the slow/fragile sm_100 flash-attn source build).
        "cd /root/JetSpec && pip install -e '.[bench,kernel]' --no-build-isolation || "
        "pip install -e '.[bench,kernel]'",
        # JetSpec's install pulls a newer `kernels` than the image's huggingface_hub
        # supports (strict-dataclass `str | None` crash at transformers import). It's an
        # optional fused-kernel helper; removing it lets transformers degrade gracefully.
        "pip uninstall -y kernels || true",
    )
)


@app.function(gpu="B200", timeout=10800, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def run(script, extra_args, tag, env_extra=None):
    """Run a JetSpec bench script, capture stdout+stderr, persist to the volume."""
    import subprocess
    env = dict(os.environ)
    env["JETSPEC_DRAFT_HEAD"] = "JetSpec/jetspec-qwen3-8b"
    env["PYTHONPATH"] = "/root/JetSpec"
    if env_extra:
        env.update(env_extra)
    cmd = ["python", script] + extra_args
    print(">>>", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd="/root/JetSpec", env=env,
                       capture_output=True, text=True)
    out = p.stdout + "\n===STDERR(tail)===\n" + "\n".join(p.stderr.splitlines()[-40:])
    with open(f"/results/{tag}.txt", "w") as f:
        f.write(out)
    vol.commit()
    print(out[-4000:], flush=True)
    return out[-2000:]


@app.local_entrypoint()
def reference(datasets: str = "gsm8k", samples: int = 32):
    """JetSpec tree vs DFlash baseline (HF reference): accept length + speedup, one run each."""
    for ds in datasets.split(","):
        run.remote(
            "bench/reference/benchmark.py",
            ["--model", "Qwen/Qwen3-8B", "--dataset", ds, "--samples", str(samples),
             "--algos", "accum_logp", "--width", "7", "--depth", "20", "--budget", "256",
             "--max-new", "1024", "--tree-attn-implementation", "triton",
             "--attn-implementation", "sdpa", "--include-dflash-baseline"],
            f"reference_{ds}")


@app.local_entrypoint()
def engine(prompt_sets: str = "gsm8k,mt_bench", samples: int = 32):
    """JetSpec engine wall-clock tok/s with cuda-graph drafter (headline numbers)."""
    env = {"JETSPEC_FUSE_GEMMS": "1",
           "JETSPEC_BACKEND": "triton_paged_tree_cudagraph_nogather"}
    for ps in prompt_sets.split(","):
        run.remote(
            "bench/engine/tps_walltime.py",
            ["--model", "Qwen/Qwen3-8B", "--prompt-set", ps, "--samples", str(samples),
             "--max-tokens", "2048", "--tree-depth", "15", "--tree-width", "7",
             "--budget", "127", "--algo", "accum_logp", "--drafter", "graphed",
             "--warm-all", "--session"],
            f"engine_{ps}", env_extra=env)


@app.function(image=image, volumes={"/results": vol})
def _list():
    import os as o
    return {fn: open(f"/results/{fn}").read()
            for fn in sorted(o.listdir("/results")) if fn.endswith(".txt")}


@app.local_entrypoint()
def pull():
    os.makedirs("jetspec_results", exist_ok=True)
    for fn, c in _list.remote().items():
        with open(f"jetspec_results/{fn}", "w") as f:
            f.write(c)
        print(f"\n===== {fn} =====\n" + "\n".join(c.splitlines()[-30:]))
