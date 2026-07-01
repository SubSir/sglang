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
        "echo js2 > /tmp/bt",
        "git clone --depth 1 https://github.com/hao-ai-lab/JetSpec /root/JetSpec",
        # DATA CONTROL: drop shuffle(seed=0) -> first-N (matches sglang/DDTree first-N).
        "sed -i 's/\\.shuffle(seed=0)\\.select/.select/' /root/JetSpec/bench/reference/benchmark.py",
        # bench+kernel extras (datasets, triton); skip flash-attn (sdpa base + triton tree
        # avoids the slow/fragile sm_100 flash-attn source build).
        "cd /root/JetSpec && pip install -e '.[bench,kernel]' --no-build-isolation || "
        "pip install -e '.[bench,kernel]'",
        # The image's `kernels` 0.14 uses `str | None` in a strict dataclass that the
        # image's huggingface_hub (0.36, <1.0 as JetSpec requires) can't validate, crashing
        # `import transformers.models.qwen3`. `kernels` is an optional fused-kernel helper;
        # removing it makes transformers degrade gracefully (verified: QWEN3_OK). Keep
        # hf_hub <1.0 (JetSpec asserts it).
        "pip install 'huggingface_hub>=0.34,<1.0' && pip uninstall -y kernels",
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
def reference(dataset: str = "gsm8k", samples: int = 32,
              budgets: str = "16,32,64,128,256", width: str = "2"):
    """Acc-len vs budget sweep (first-N data control). accum_logp tree.
    First budget run also emits the DFlash chain baseline for reference."""
    handles = []
    for i, b in enumerate(budgets.split(",")):
        extra = ["--include-dflash-baseline"] if i == 0 else []
        handles.append(run.spawn(
            "bench/reference/benchmark.py",
            ["--model", "Qwen/Qwen3-8B", "--dataset", dataset, "--samples", str(samples),
             "--algos", "accum_logp", "--width", width, "--depth", "20", "--budget", b,
             "--max-new", "1024", "--tree-attn-implementation", "triton",
             "--attn-implementation", "sdpa", *extra],
            f"ref_{dataset}_w{width}_b{b}"))
    print(f"launched JetSpec ref sweep: {dataset} budgets={budgets} width={width} — waiting")
    for h in handles:
        print(h.get()[-300:])


@app.local_entrypoint()
def engine(prompt_sets: str = "gsm8k,mt_bench", samples: int = 32,
           budgets: str = "16,32,64,128,256", widths: str = "7", tree_depth: int = 20,
           draft_head: str = "JetSpec/jetspec-qwen3-8b", tag_suffix: str = ""):
    """JetSpec engine wall-clock tok/s (conc=1) with cuda-graph drafter — budget x width
    sweep. This is JetSpec's BEST/headline path (own engine, only conc=1). tree_depth=20
    matches the official README config (b128/w7/depth20 -> their reported numbers).
    draft_head: swap in a different DFlash head (e.g. z-lab/Qwen3-8B-DFlash-b16 = OUR
    bidirectional head, which loads via the same DFlashDraftModel.from_pretrained and
    runs bidirectional automatically since its config has no causal_head)."""
    env = {"JETSPEC_FUSE_GEMMS": "1",
           "JETSPEC_BACKEND": "triton_paged_tree_cudagraph_nogather"}
    handles = []
    for ps in prompt_sets.split(","):
        for b in budgets.split(","):
            for w in widths.split(","):
                handles.append(run.spawn(
                    "bench/engine/tps_walltime.py",
                    ["--model", "Qwen/Qwen3-8B", "--draft-head", draft_head,
                     "--prompt-set", ps, "--samples", str(samples),
                     "--max-tokens", "2048", "--tree-depth", str(tree_depth), "--tree-width", w,
                     "--budget", b, "--algo", "accum_logp", "--drafter", "graphed",
                     "--warm-all", "--session"],
                    f"engine_{ps}_b{b}_w{w}_d{tree_depth}_s{samples}{tag_suffix}", env_extra=env))
    print(f"launched JetSpec engine: {prompt_sets} b={budgets} w={widths} d={tree_depth} head={draft_head} — waiting")
    for h in handles:
        print(h.get()[-200:])


@app.function(gpu="B200", timeout=10800, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def head_ab(prompt_set: str = "gsm8k", samples: int = 80, budget: int = 128,
            width: int = 7, tree_depth: int = 20):
    """SAME-CARD A/B: jetspec head vs OUR DFlash head, both in JetSpec's fast engine,
    back-to-back in ONE container (kills host variance). Compute per step is identical
    (same DFlash arch, configs differ only in causal_head -> draft mask); so if our
    head's acc (higher) holds, it should give >= throughput on the same card."""
    import subprocess
    env = dict(os.environ)
    env["JETSPEC_FUSE_GEMMS"] = "1"
    env["JETSPEC_BACKEND"] = "triton_paged_tree_cudagraph_nogather"
    env["PYTHONPATH"] = "/root/JetSpec"
    out = {}
    for label, head in [("jetspec", "JetSpec/jetspec-qwen3-8b"),
                        ("ourhead", "z-lab/Qwen3-8B-DFlash-b16")]:
        cmd = ["python", "bench/engine/tps_walltime.py", "--model", "Qwen/Qwen3-8B",
               "--draft-head", head, "--prompt-set", prompt_set, "--samples", str(samples),
               "--max-tokens", "2048", "--tree-depth", str(tree_depth), "--tree-width", str(width),
               "--budget", str(budget), "--algo", "accum_logp", "--drafter", "graphed",
               "--warm-all", "--session"]
        p = subprocess.run(cmd, cwd="/root/JetSpec", env=env, capture_output=True, text=True)
        lines = [l for l in p.stdout.splitlines() if l.startswith("tree ")]
        out[label] = lines[-1] if lines else ("ERR " + (p.stdout[-200:] + p.stderr[-300:]))
        with open(f"/results/headab_{label}_{prompt_set}.txt", "w") as f:
            f.write(p.stdout + "\n==ERR==\n" + p.stderr[-2000:])
    vol.commit()
    for k, v in out.items():
        print(f">>> {k}: {v}", flush=True)
    return out


@app.function(gpu="B200", timeout=10800, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def config_ab(prompt_set: str = "gsm8k", samples: int = 128, tree_depth: int = 20):
    """SAME-CARD A/B: jetspec b128/w7 vs b32/w2, back-to-back in ONE container (kills
    host variance) -> definitive answer to which jetspec config is faster."""
    import subprocess
    env = dict(os.environ)
    env["JETSPEC_FUSE_GEMMS"] = "1"
    env["JETSPEC_BACKEND"] = "triton_paged_tree_cudagraph_nogather"
    env["PYTHONPATH"] = "/root/JetSpec"
    out = {}
    for label, budget, width in [("b128_w7", 128, 7), ("b32_w2", 32, 2)]:
        cmd = ["python", "bench/engine/tps_walltime.py", "--model", "Qwen/Qwen3-8B",
               "--draft-head", "JetSpec/jetspec-qwen3-8b", "--prompt-set", prompt_set,
               "--samples", str(samples), "--max-tokens", "1024",
               "--tree-depth", str(tree_depth), "--tree-width", str(width),
               "--budget", str(budget), "--algo", "accum_logp", "--drafter", "graphed",
               "--warm-all", "--session"]
        p = subprocess.run(cmd, cwd="/root/JetSpec", env=env, capture_output=True, text=True)
        lines = [l for l in p.stdout.splitlines() if l.startswith("tree ")]
        out[label] = lines[-1] if lines else ("ERR " + (p.stdout[-200:] + p.stderr[-300:]))
    for k, v in out.items():
        print(f">>> {k}: {v}", flush=True)
    return out


@app.local_entrypoint()
def cfgab(prompt_sets: str = "gsm8k,math500", samples: int = 128):
    hs = [(ps, config_ab.spawn(ps, samples)) for ps in prompt_sets.split(",")]
    for ps, h in hs:
        print(f"=== {ps} ==="); print(h.get())


@app.local_entrypoint()
def headab(prompt_set: str = "gsm8k", samples: int = 80, budget: int = 128):
    print(head_ab.remote(prompt_set, samples, budget))


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


@app.function(image=image, timeout=600)
def diag():
    """No-GPU: find the kernels/hf_hub combo that imports qwen3 cleanly."""
    import subprocess as sp
    def sh(c): return sp.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    def imp():
        r = sp.run(["python","-c","import transformers; print(transformers.__version__);"
                    "import transformers.models.qwen3.modeling_qwen3; print('QWEN3_OK')"],
                   capture_output=True, text=True)
        return (r.stdout + r.stderr).splitlines()[-4:]
    print("hf_hub:", sh("pip show huggingface_hub | grep -i version"))
    print("kernels:", sh("pip show kernels | grep -i version"))
    print("transformers:", sh("pip show transformers | grep -i version"))
    print("--- import as-is ---"); print(imp())
    sh("pip uninstall -y kernels")
    print("--- after uninstall kernels ---"); print(imp())
