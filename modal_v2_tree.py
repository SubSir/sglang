"""Validate + benchmark the v2-worker tree-verify port on upstream SGLang.

Base image = upstream sgl-project/sglang @ tip (sglang-kernel 0.4.4, flashinfer 0.6.12),
overlaid with:
  - the tree-verify port from the worktree ../sglang-v2-tree (speculative/ dir)
  - the fork's patched bench (benchmark/dflash) for the sweep harness

Validate:  modal run modal_v2_tree.py
Sweep:     modal run modal_v2_tree.py::sweep --target-model ... --draft-model ... --block-size N
"""

import os
import modal

app = modal.App("dflash-v2-tree")

# Persist results so they survive local session death (`modal volume get
# dflash-v2-results ./pulled` to retrieve, or the `pull` entrypoint below).
results_vol = modal.Volume.from_name("dflash-v2-results", create_if_missing=True)

WT = "/Users/subsir/Desktop/Studio/Python/sglang-v2-tree"
UPSTREAM_COMMIT = "e0c0c0a45"

# Official same-day (2026-06-27) nightly: deep_gemm / nvrtc / sgl-kernel / flashinfer
# / rust all prebuilt and consistent. We only overlay our tree-verify port.
base_image = (
    modal.Image.from_registry(
        "lmsysorg/sglang:nightly-dev-cu12-20260627-13b5bd96"
    )
    .run_commands("echo v2tree3 > /tmp/build_time")
    # Overlay the tree-verify port (the speculative dir from the worktree).
    .add_local_dir(
        f"{WT}/python/sglang/srt/speculative",
        remote_path="/tmp/port_speculative",
        copy=True,
    )
    .add_local_dir(
        f"{WT}/python/sglang/srt/arg_groups",
        remote_path="/tmp/port_arg_groups",
        copy=True,
    )
    .add_local_dir("./benchmark", remote_path="/root/benchmark", copy=True)
    .run_commands(
        "SGLANG_DIR=$(python3 -c 'import sglang,os;print(os.path.dirname(sglang.__file__))') && "
        "echo \"sglang at $SGLANG_DIR\" && "
        "cp -rf /tmp/port_speculative/. \"$SGLANG_DIR/srt/speculative/\" && "
        "cp -rf /tmp/port_arg_groups/. \"$SGLANG_DIR/srt/arg_groups/\" && "
        "echo overlaid tree-verify port"
    )
)


@app.function(gpu="B200", timeout=10800, image=base_image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results_vol})
def run(target_model, draft_model, data_names, tree_verify, topk, block_size,
        num_draft_tokens, samples_base, concurrencies, backend="flashinfer",
        mem_fraction=0.8, tp=1, gpu_count=1, skip_baseline=True, result_tag=None):
    """One server launch; bench internally sweeps all concurrencies x datasets.
    When skip_baseline is False, the bench runs the target-only baseline on the
    SAME GPU (serially) before DFLASH, yielding the speedup denominator."""
    import sys, importlib.util

    args = [
        "bench_dflash_sweep.py",
        "--data-names", data_names,
        "--target-model", target_model,
        "--draft-model", draft_model,
        "--tp-sizes", str(tp),
        "--concurrencies", str(concurrencies),
        "--samples-per-concurrency-base", str(samples_base),
        "--max-samples-per-config", str(samples_base * 64),
        "--max-new-tokens", "1024",
        "--attention-backends", backend,
        "--mem-fraction-static", str(mem_fraction),
        "--speculative-eagle-topk", str(topk),
        "--speculative-dflash-block-size", str(block_size),
        # cap running requests = max swept concurrency, so the bench only captures
        # cuda graphs up to that bs (128 graphs OOMs tree verify).
        "--max-running-requests", str(max(int(c) for c in str(concurrencies).split(","))),
    ]
    if skip_baseline:
        args.append("--skip-baseline")
    if num_draft_tokens is not None:
        args += ["--speculative-num-draft-tokens", str(num_draft_tokens)]

    out_path = f"/root/out_{'tree' if tree_verify else 'chain'}_b{block_size}_t{topk}.md"
    args += ["--output-md", out_path]

    env = os.environ
    env["SGLANG_DFLASH_TREE_VERIFY"] = "1" if tree_verify else "0"
    env["DFLASH_DRAFT_ATTN_BACKEND"] = backend  # avoid broken fa4

    os.chdir("/root")
    sys.argv = [a for a in args if a]
    spec = importlib.util.spec_from_file_location(
        "bench", "/root/benchmark/dflash/bench_dflash_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    spec.loader.exec_module(mod)
    mod.main()
    with open(out_path) as f:
        res = f.read()
    if result_tag:
        # Persist to the volume so results survive local session death.
        with open(f"/results/{result_tag}.md", "w") as f:
            f.write(res)
        results_vol.commit()
    return res


@app.function(gpu="B200", timeout=1200, image=base_image,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def debug(target_model="Qwen/Qwen3-8B", draft_model="z-lab/Qwen3-8B-DFlash-b16",
          block_size=16, num_draft_tokens=32, topk=4, tree=True):
    """Launch the DFLASH tree server directly, capture stdout+stderr, print the tail."""
    import subprocess, os, time
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", target_model, "--trust-remote-code",
        "--attention-backend", "flashinfer",
        "--speculative-draft-attention-backend", "flashinfer",
        "--tp-size", "1", "--dtype", "bfloat16",
        "--mem-fraction-static", "0.8", "--max-running-requests", "32",
        "--page-size", "1", "--cuda-graph-max-bs", "32",
        "--speculative-algorithm", "DFLASH",
        "--speculative-draft-model-path", draft_model,
        "--speculative-dflash-block-size", str(block_size),
        "--speculative-num-draft-tokens", str(num_draft_tokens),
        "--speculative-eagle-topk", str(topk),
        "--port", "30000",
    ]
    env = dict(os.environ)
    env["SGLANG_DFLASH_TREE_VERIFY"] = "1" if tree else "0"
    env["DFLASH_DRAFT_ATTN_BACKEND"] = "flashinfer"
    print("CMD:", " ".join(cmd), "TREE=", env["SGLANG_DFLASH_TREE_VERIFY"])
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    t0 = time.time()
    for line in p.stdout:
        lines.append(line.rstrip())
        if p.poll() is not None or time.time() - t0 > 600:
            break
        if "The server is fired up" in line or "Uvicorn running" in line:
            lines.append(">>> SERVER UP OK")
            p.terminate(); break
    try:
        p.wait(timeout=10)
    except Exception:
        p.kill()
    return "\n".join(lines[-120:])


def _tables(res):
    """Pull the per-conc accept-length and tok/s table rows (tp=1) for a quick echo."""
    out = []
    lines = res.splitlines()
    header = None
    for i, line in enumerate(lines):
        if line.strip().startswith("| tp\\conc"):
            header = line
        if ("acceptance length" in line.lower() or "DFLASH output tok/s" in line):
            for j in range(i + 1, min(i + 7, len(lines))):
                if lines[j].startswith("| 1 |"):
                    label = "accept" if "accept" in line.lower() else "tok/s"
                    out.append(f"  {label:7s} {header}  ->  {lines[j]}")
                    break
    return "\n".join(out)


@app.local_entrypoint()
def dbg(tree: bool = True):
    print(debug.remote(tree=tree))


@app.local_entrypoint()
def main(target_model: str = "Qwen/Qwen3-8B",
         draft_model: str = "z-lab/Qwen3-8B-DFlash-b16",
         block_size: int = 16):
    """Quick validation: chain vs tree(budget32) at conc=1, gsm8k, cuda graph ON."""
    configs = [("chain", False, 1, block_size), ("tree_b32", True, 4, 32)]
    os.makedirs("v2_tree_results", exist_ok=True)
    for label, tv, topk, ndt in configs:
        print(f">>> {label}")
        try:
            res = run.remote(target_model, draft_model, "gsm8k", tv, topk,
                             block_size, ndt, 24, "1")
            with open(f"v2_tree_results/validate_{label}_{target_model.split('/')[-1]}.md", "w") as f:
                f.write(res)
            print(_tables(res))
        except Exception as e:
            print(f">>> FAILED {label}: {e}")


"""Target models for the three-model study (z-lab internal drafts via HF secret)."""
THREE_MODELS = [
    # (target, draft, block_size, tp, mem_fraction)
    ("openai/gpt-oss-120b", "z-lab/gpt-oss-120b-DFlash", 10, 1, 0.85),
    ("google/gemma-4-31b-it", "z-lab/gemma-4-31B-it-DFlash", 16, 1, 0.8),
    ("Qwen/Qwen3.6-27B", "z-lab/Qwen3.6-27B-DFlash", 16, 1, 0.8),
]


@app.function(image=base_image, volumes={"/results": results_vol})
def _list_results():
    import os
    out = {}
    for fn in sorted(os.listdir("/results")):
        if fn.endswith(".md"):
            with open(f"/results/{fn}") as f:
                out[fn] = f.read()
    return out


@app.local_entrypoint()
def pull():
    """Retrieve all persisted results from the volume (survives session death)."""
    os.makedirs("v2_tree_results", exist_ok=True)
    out = _list_results.remote()
    for fn, content in out.items():
        with open(f"v2_tree_results/{fn}", "w") as f:
            f.write(content)
        print(f"\n===== {fn} =====")
        print(_tables(content))
    print(f"\npulled {len(out)} result files to v2_tree_results/")


def _budgets_for(block_size):
    return [8, 16, 32, 64] if block_size <= 8 else (
        [10, 16, 32, 64] if block_size == 10 else [16, 32, 64])


@app.local_entrypoint()
def three(concurrencies: str = "1,8,32",
          data_names: str = "gsm8k,mt-bench",
          samples_base: int = 8,
          models: str = ""):
    """Run all 3 target models concurrently (one container/card each spawn).

    Per model: chain@blocksize (WITH baseline, same card) + tree at each budget.
    cuda graph stays ON (not disabled). Tree verify step runs eager internally
    (port TODO), base decode + draft + chain verify are graphed.
    """
    sel = set(m.strip() for m in models.split(",") if m.strip())
    os.makedirs("v2_tree_results", exist_ok=True)
    calls = []
    for target, draft, bsz, tp, memf in THREE_MODELS:
        if sel and target.split("/")[-1] not in sel and target not in sel:
            continue
        mtag = target.split("/")[-1]
        configs = [("chain", False, 1, bsz, False)]
        for b in _budgets_for(bsz):
            configs.append((f"tree_b{b}", True, 4, b, True))
        for label, tv, topk, ndt, skip_bl in configs:
            print(f">>> spawn {mtag}/{label} budget={ndt} baseline={not skip_bl}")
            calls.append((mtag, label, ndt, run.spawn(
                target, draft, data_names, tv, topk, bsz, ndt, samples_base,
                concurrencies, "flashinfer", memf, tp, tp, skip_bl, f"{mtag}_{label}")))
    summary = {}
    for mtag, label, ndt, c in calls:
        try:
            res = c.get()
            with open(f"v2_tree_results/three_{mtag}_{label}.md", "w") as f:
                f.write(res)
            summary.setdefault(mtag, []).append((label, ndt, _tables(res)))
            print(f"\n===== {mtag} / {label} (budget={ndt}) =====\n{_tables(res)}")
        except Exception as e:
            print(f">>> FAILED {mtag}/{label}: {e}")
    print("\n#################### DONE ####################")
    for mtag, items in summary.items():
        print(f"\n## {mtag}")
        for label, ndt, t in items:
            print(f"  {label} (budget={ndt})")


@app.local_entrypoint()
def sweep(target_model: str = "Qwen/Qwen3-8B",
          draft_model: str = "z-lab/Qwen3-8B-DFlash-b16",
          block_size: int = 16,
          budgets: str = "",          # comma list; default derived from block_size
          concurrencies: str = "1,8,32",
          data_names: str = "gsm8k,mt-bench",
          samples_base: int = 16,
          tp: int = 1,
          mem_fraction: float = 0.8):
    """Sweep one model: chain + tree at each budget. ONE server per config sweeps
    all concurrencies x datasets internally. budget rule: blocksize 8 -> 8,16,32,64;
    else -> 16,32,64 (starting at block_size)."""
    if budgets.strip():
        budget_list = [int(b) for b in budgets.split(",") if b.strip()]
    elif block_size <= 8:
        budget_list = [8, 16, 32, 64]
    else:
        budget_list = [16, 32, 64]

    # chain at budget=block_size ALSO runs the no-spec baseline (same GPU, serial),
    # giving the speedup denominator. Tree budgets skip baseline (same value).
    configs = [("chain", False, 1, block_size, False)]
    for b in budget_list:
        configs.append((f"tree_b{b}", True, 4, b, True))

    os.makedirs("v2_tree_results", exist_ok=True)
    mtag = target_model.split("/")[-1]
    # Spawn all configs in parallel (each is its own container/GPU).
    calls = []
    for label, tv, topk, ndt, skip_bl in configs:
        print(f">>> spawning {label} (budget={ndt}, conc={concurrencies}, data={data_names}, baseline={not skip_bl})")
        calls.append((label, ndt, run.spawn(
            target_model, draft_model, data_names, tv, topk, block_size,
            ndt, samples_base, concurrencies, "flashinfer", mem_fraction, tp, tp, skip_bl)))
    for label, ndt, c in calls:
        try:
            res = c.get()
            with open(f"v2_tree_results/sweep_{mtag}_{label}.md", "w") as f:
                f.write(res)
            print(f"\n========== {mtag} / {label} (budget={ndt}) ==========")
            print(_tables(res))
        except Exception as e:
            print(f">>> FAILED {label}: {e}")
