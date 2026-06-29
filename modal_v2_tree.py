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
    .add_local_file("./patch_cudagraph.py", remote_path="/tmp/patch_cudagraph.py", copy=True)
    .run_commands(
        "SGLANG_DIR=$(python3 -c 'import sglang,os;print(os.path.dirname(sglang.__file__))') && "
        "echo \"sglang at $SGLANG_DIR\" && "
        "cp -rf /tmp/port_speculative/. \"$SGLANG_DIR/srt/speculative/\" && "
        "cp -rf /tmp/port_arg_groups/. \"$SGLANG_DIR/srt/arg_groups/\" && "
        "echo overlaid tree-verify port",
        "python3 /tmp/patch_cudagraph.py",
    )
)


@app.function(gpu="B200", timeout=10800, image=base_image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results_vol})
def run(target_model, draft_model, data_names, tree_verify, topk, block_size,
        num_draft_tokens, samples_base, concurrencies, backend="flashinfer",
        mem_fraction=0.8, tp=1, gpu_count=1, skip_baseline=True, result_tag=None,
        disable_cuda_graph=False, tree_algo="fused",
        depth_bonus=None, force_width=None):
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
    if disable_cuda_graph:
        args.append("--disable-cuda-graph")
    if num_draft_tokens is not None:
        args += ["--speculative-num-draft-tokens", str(num_draft_tokens)]

    out_path = f"/root/out_{'tree' if tree_verify else 'chain'}_b{block_size}_t{topk}.md"
    args += ["--output-md", out_path]

    env = os.environ
    env["SGLANG_DFLASH_TREE_VERIFY"] = "1" if tree_verify else "0"
    env["SGLANG_DFLASH_TREE_ALGO"] = tree_algo  # "fused" | "ddtree"
    # ddtree ablation knobs (default None = builder defaults)
    if depth_bonus is not None:
        env["SGLANG_DDTREE_DEPTH_BONUS"] = str(depth_bonus)
    if force_width is not None:
        env["SGLANG_DDTREE_FORCE_WIDTH"] = str(force_width)
    env["DFLASH_DRAFT_ATTN_BACKEND"] = backend  # avoid broken fa4
    # VL models (Qwen3.6-VL) need a non-cute vision-attention backend.
    if any(k in target_model.lower() for k in ("qwen3.6", "qwen3.5", "-vl")):
        env["DFLASH_MM_ATTN_BACKEND"] = "triton_attn"

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
          block_size=16, num_draft_tokens=32, topk=4, tree=True, tree_algo="fused"):
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
    env["SGLANG_DFLASH_TREE_ALGO"] = tree_algo
    env["DFLASH_DRAFT_ATTN_BACKEND"] = "flashinfer"
    env["CUDA_LAUNCH_BLOCKING"] = "1"  # precise CUDA error location
    print("CMD:", " ".join(cmd), "TREE=", env["SGLANG_DFLASH_TREE_VERIFY"], flush=True)
    import threading, urllib.request, json as _json

    def _send():
        # poll health, then send a real generate to trigger the tree-verify forward
        for _ in range(180):
            try:
                urllib.request.urlopen("http://127.0.0.1:30000/health_generate", timeout=3).read()
                break
            except Exception:
                time.sleep(2)
        print(">>> SERVER HEALTHY, sending generate", flush=True)
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:30000/generate",
                data=_json.dumps({"text": "Solve 2+2 step by step.",
                                  "sampling_params": {"temperature": 0, "max_new_tokens": 64}}).encode(),
                headers={"Content-Type": "application/json"})
            r = urllib.request.urlopen(req, timeout=180).read().decode()
            print(">>> GENERATE OK: " + r[:300], flush=True)
        except Exception as e:
            print(f">>> GENERATE ERR: {e}", flush=True)

    threading.Thread(target=_send, daemon=True).start()
    # Inherit stdout/stderr so the scheduler subprocess crash lands in container logs.
    p = subprocess.Popen(cmd, env=env)
    try:
        p.wait(timeout=480)
    except Exception:
        pass
    try:
        p.terminate(); p.wait(timeout=10)
    except Exception:
        p.kill()
    return ">>> debug done (see container logs above)"


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
def dbg(tree: bool = True, tree_algo: str = "fused"):
    print(debug.remote(tree=tree, tree_algo=tree_algo))


@app.local_entrypoint()
def throughput(target_model: str = "Qwen/Qwen3-8B",
               draft_model: str = "z-lab/Qwen3-8B-DFlash-b16",
               data_names: str = "gsm8k,mt-bench",
               block_size: int = 16, budget: int = 32, topk: int = 4,
               concurrencies: str = "1,8,32", with_chain: bool = True,
               backend: str = "flashinfer"):
    """v2-port (fast upstream nightly + working tree) throughput+accept: chain vs tree,
    cuda graph ON. topk should scale with budget (topk4 too narrow at big budget -> acc drops)."""
    configs = [(f"tree_b{budget}_k{topk}", True, topk, budget)]
    if with_chain:
        configs = [("chain", False, 1, None)] + configs
    os.makedirs("v2_tree_results", exist_ok=True)
    for label, tv, topk, ndt in configs:
        print(f">>> {label} conc={concurrencies} backend={backend}")
        try:
            res = run.remote(target_model, draft_model, data_names, tv, topk,
                             block_size, ndt, 32, concurrencies, backend,
                             0.85, 1, 1, True, f"tp_{label}", False)  # cuda graph ON
            with open(f"v2_tree_results/tp_{label}_{data_names.replace(',','-')}.md", "w") as f:
                f.write(res)
            print(_tables(res))
        except Exception as e:
            print(f">>> FAILED {label}: {e}")


@app.local_entrypoint()
def ddtree_bench(target_model: str = "Qwen/Qwen3-8B",
                 draft_model: str = "z-lab/Qwen3-8B-DFlash-b16",
                 data_names: str = "gsm8k",
                 block_size: int = 16, budgets: str = "16,32,64", topk: int = 4,
                 concurrencies: str = "1", backend: str = "triton"):
    """Paired chain vs tree-fused vs tree-ddtree across budgets. triton verify
    (our throughput win). Prints accept + tok/s per config."""
    os.makedirs("v2_tree_results", exist_ok=True)
    budget_list = [int(b) for b in str(budgets).split(",")]
    # chain once (budget-independent); then fused+ddtree per budget.
    configs = [("chain", False, 1, None, "fused")]
    for b in budget_list:
        configs.append((f"tree_fused_b{b}", True, topk, b, "fused"))
        configs.append((f"tree_ddtree_b{b}", True, topk, b, "ddtree"))
    for label, tv, tk, ndt, algo in configs:
        print(f">>> {label} algo={algo} conc={concurrencies} backend={backend}")
        try:
            res = run.remote(target_model, draft_model, data_names, tv, tk,
                             block_size, ndt, 32, concurrencies, backend,
                             0.85, 1, 1, True, f"ddt_{label}", False, algo)
            with open(f"v2_tree_results/ddt_{label}_{data_names.replace(',','-')}.md", "w") as f:
                f.write(res)
            print(_tables(res))
        except Exception as e:
            print(f">>> FAILED {label}: {e}")


@app.local_entrypoint()
def ddtree_ablate(target_model: str = "Qwen/Qwen3-8B",
                  draft_model: str = "z-lab/Qwen3-8B-DFlash-b16",
                  block_size: int = 16, budget: int = 32, topk: int = 4,
                  backend: str = "triton"):
    """Isolate the ddtree accept gap: default vs bonus=0 vs bonus=0+width=4."""
    os.makedirs("v2_tree_results", exist_ok=True)
    runs = [
        ("ablate_default", None, None),
        ("ablate_bonus0", 0.0, None),
        ("ablate_bonus0_w4", 0.0, topk),
    ]
    for label, db, fw in runs:
        print(f">>> {label} bonus={db} width={fw}")
        try:
            res = run.remote(target_model, draft_model, "gsm8k", True, topk,
                             block_size, budget, 32, "1", backend,
                             0.85, 1, 1, True, label, False, "ddtree", db, fw)
            with open(f"v2_tree_results/{label}_gsm8k.md", "w") as f:
                f.write(res)
            print(_tables(res))
        except Exception as e:
            print(f">>> FAILED {label}: {e}")


@app.function(gpu="B200", timeout=10800, image=base_image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results_vol})
def profile_decode(tree_verify: bool, topk: int, budget: int, tag: str,
                   target="Qwen/Qwen3-8B", draft="z-lab/Qwen3-8B-DFlash-b16",
                   block_size: int = 16, n_prof_reqs: int = 6, disable_cuda_graph: bool = False,
                   overlap: bool = False):
    """Launch v2-port server (DFLASH chain or tree), drive sglang's torch profiler
    over a few gsm8k decodes, then analyze the trace: GPU-busy vs wall (util),
    top GPU kernels, top CPU ops. This catches launch/CPU-bound overhead that
    cuda-event timing would mis-attribute to slow kernels."""
    import subprocess, time, signal, json, glob, gzip, os, urllib.request
    from collections import defaultdict
    from datasets import load_dataset
    from transformers import AutoTokenizer

    profdir = f"/results/prof_{tag}"
    os.makedirs(profdir, exist_ok=True)
    env = dict(os.environ); env["SGLANG_TORCH_PROFILER_DIR"] = profdir
    # tree-verify is gated by env (bypasses the block_size==num_draft_tokens check)
    env["SGLANG_DFLASH_TREE_VERIFY"] = "1" if tree_verify else "0"
    env["SGLANG_DFLASH_BLOCK_VERIFY"] = "0" if tree_verify else "1"
    env["DFLASH_DRAFT_ATTN_BACKEND"] = "flashinfer"
    spec = ["--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", draft,
            "--speculative-dflash-block-size", str(block_size),
            "--speculative-eagle-topk", str(topk if tree_verify else 1)]
    if tree_verify:
        spec += ["--speculative-num-draft-tokens", str(budget)]
        if not overlap:
            spec += ["--disable-overlap-schedule"]
    server = ["python", "-m", "sglang.launch_server", "--model-path", target,
              "--trust-remote-code", "--mem-fraction-static", "0.85",
              "--max-running-requests", "1", "--attention-backend", "flashinfer",
              *spec, "--port", "30000"]
    if disable_cuda_graph:
        server.append("--disable-cuda-graph")
    print(">>>", " ".join(server), flush=True)
    srv = subprocess.Popen(server, env=env)
    ok = False
    for _ in range(240):
        try:
            urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=2); ok = True; break
        except Exception:
            time.sleep(2)
    if not ok:
        srv.send_signal(signal.SIGINT); return f"{tag}: server failed to start"

    tok = AutoTokenizer.from_pretrained(target)
    ds = None
    for _ in range(5):
        try:
            ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n_prof_reqs + 2)); break
        except Exception as e:
            print("gsm8k load retry:", e); time.sleep(10)
    if ds is None:
        srv.send_signal(signal.SIGINT); return f"{tag}: dataset load failed"
    fmt = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    def prompts(k0, k1):
        return [tok.apply_chat_template([{"role": "user", "content": fmt.format(q=ds[i]["question"])}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False) for i in range(k0, k1)]
    def send(p, mx=256):
        req = urllib.request.Request("http://127.0.0.1:30000/generate",
            data=json.dumps({"text": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": mx}}).encode(),
            headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=600).read())

    for p in prompts(0, 2): send(p, 64)                          # warmup
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:30000/start_profile",
        data=b"{}", headers={"Content-Type": "application/json"}), timeout=30)
    accs = []
    for p in prompts(2, 2 + n_prof_reqs):
        r = send(p, 256); accs.append(r["meta_info"].get("spec_verify_ct"))
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:30000/stop_profile",
        data=b"{}", headers={"Content-Type": "application/json"}), timeout=120)
    time.sleep(8)
    srv.send_signal(signal.SIGINT); time.sleep(3)

    # ---- analyze the chrome trace ----
    files = sorted(glob.glob(f"{profdir}/*.trace.json*") + glob.glob(f"{profdir}/*.json*"))
    summary = {"tag": tag, "tree": tree_verify, "topk": topk, "budget": budget, "trace_files": files}
    if not files:
        summary["error"] = "no trace file";
        json.dump(summary, open(f"/results/profsum_{tag}.json", "w"), indent=2); results_vol.commit()
        return json.dumps(summary)
    f = files[-1]
    raw = gzip.open(f).read() if f.endswith(".gz") else open(f, "rb").read()
    ev = json.loads(raw)["traceEvents"]
    gpu_pids = set()
    for e in ev:
        if e.get("ph") == "M" and e.get("name") == "process_labels" and "GPU" in str(e.get("args", {}).get("labels", "")):
            gpu_pids.add(e.get("pid"))
    ksum = defaultdict(float); csum = defaultdict(float); gpu_busy = 0.0; cpu_busy = 0.0
    tmin = float("inf"); tmax = 0.0
    for e in ev:
        if e.get("ph") != "X" or "dur" not in e: continue
        dur = e["dur"]; cat = e.get("cat", "")
        ts = e.get("ts", 0); tmin = min(tmin, ts); tmax = max(tmax, ts + dur)
        if cat in ("kernel", "gpu_memcpy", "gpu_memset") or e.get("pid") in gpu_pids:
            ksum[e["name"]] += dur; gpu_busy += dur
        elif cat in ("cpu_op", "user_annotation", "cuda_runtime", "python_function"):
            csum[e["name"]] += dur
            if cat == "cpu_op": cpu_busy += dur
    wall = (tmax - tmin) if tmax > tmin else 1.0
    summary["wall_us"] = round(wall, 1)
    summary["gpu_busy_us"] = round(gpu_busy, 1)
    summary["gpu_util_pct"] = round(100 * gpu_busy / wall, 1)
    summary["accept_verify_ct"] = accs
    summary["top_gpu_kernels"] = sorted(({"name": k[:70], "us": round(v, 1)} for k, v in ksum.items()),
                                        key=lambda x: -x["us"])[:18]
    summary["top_cpu_ops"] = sorted(({"name": k[:70], "us": round(v, 1)} for k, v in csum.items()),
                                    key=lambda x: -x["us"])[:18]
    json.dump(summary, open(f"/results/profsum_{tag}.json", "w"), indent=2); results_vol.commit()
    print(json.dumps({k: summary[k] for k in ("tag", "wall_us", "gpu_busy_us", "gpu_util_pct")}), flush=True)
    return json.dumps(summary)[:2500]


@app.function(gpu="B200", timeout=10800, image=base_image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results_vol})
def quickbench(tree_verify: bool, topk: int, budget: int, tag: str, overlap: bool = False,
               target="Qwen/Qwen3-8B", draft="z-lab/Qwen3-8B-DFlash-b16",
               block_size: int = 16, n: int = 24):
    """Lean conc=1 gsm8k throughput+accept for the v2-port tree, with an OVERLAP toggle
    (test if tree can run with overlap scheduling on = hide exposed CPU)."""
    import subprocess, time, signal, json, urllib.request
    env = dict(os.environ)
    env["SGLANG_DFLASH_TREE_VERIFY"] = "1" if tree_verify else "0"
    env["SGLANG_DFLASH_BLOCK_VERIFY"] = "0" if tree_verify else "1"
    env["DFLASH_DRAFT_ATTN_BACKEND"] = "flashinfer"
    spec = ["--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", draft,
            "--speculative-dflash-block-size", str(block_size),
            "--speculative-eagle-topk", str(topk if tree_verify else 1)]
    if tree_verify:
        spec += ["--speculative-num-draft-tokens", str(budget)]
        if not overlap:
            spec += ["--disable-overlap-schedule"]
    server = ["python", "-m", "sglang.launch_server", "--model-path", target, "--trust-remote-code",
              "--mem-fraction-static", "0.85", "--max-running-requests", "1",
              "--attention-backend", "flashinfer", *spec, "--port", "30000"]
    print(">>>", " ".join(server), flush=True)
    srv = subprocess.Popen(server, env=env)
    ok = False
    for _ in range(240):
        try: urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=2); ok = True; break
        except Exception: time.sleep(2)
    out = {"tag": tag, "ready": ok, "overlap": overlap}
    if ok:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(target)
        ds = None
        for _ in range(5):
            try: ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n)); break
            except Exception: time.sleep(10)
        fmt = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        ps = [tok.apply_chat_template([{"role": "user", "content": fmt.format(q=ds[i]["question"])}],
              tokenize=False, add_generation_prompt=True, enable_thinking=False) for i in range(n)]
        def send(p):
            req = urllib.request.Request("http://127.0.0.1:30000/generate",
                data=json.dumps({"text": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": 512}}).encode(),
                headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=600).read())
        send(ps[0]); send(ps[1])  # warmup
        t0 = time.perf_counter(); toks = 0; accs = []
        for p in ps:
            r = send(p); m = r["meta_info"]
            toks += m["completion_tokens"]
            if m.get("spec_verify_ct"): accs.append(m["completion_tokens"] / m["spec_verify_ct"])
        dt = time.perf_counter() - t0
        out["tok_s"] = round(toks / dt, 1); out["accept_len"] = round(sum(accs)/len(accs), 2) if accs else None
        out["tokens"] = toks; out["secs"] = round(dt, 2)
    srv.send_signal(signal.SIGINT); time.sleep(2)
    json.dump(out, open(f"/results/qb_{tag}.json", "w")); results_vol.commit()
    print(json.dumps(out), flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def qb():
    """Test: chain, tree(overlap off, current), tree(overlap ON, experiment)."""
    hs = [
        ("chain", quickbench.spawn(False, 1, 16, "chain", False)),
        ("tree_b32_noov", quickbench.spawn(True, 4, 32, "tree_b32_noov", False)),
        ("tree_b32_overlap", quickbench.spawn(True, 4, 32, "tree_b32_overlap", True)),
    ]
    for name, h in hs:
        try: print(name, "->", h.get())
        except Exception as e: print(name, "FAILED", e)


@app.local_entrypoint()
def qbfull(n: int = 48):
    """Robust conc=1 single-request throughput: chain (overlap on) vs tree (overlap on)
    at budgets 16/32/64. Establishes tree > chain+10%."""
    hs = [("chain", quickbench.spawn(False, 1, 16, "qbf_chain", True, n=n))]
    for b in (16, 32, 64):
        hs.append((f"tree_b{b}", quickbench.spawn(True, 4, b, f"qbf_tree_b{b}", True, n=n)))
    rows = {}
    for name, h in hs:
        try:
            r = h.get(); rows[name] = r; print(name, "->", r)
        except Exception as e:
            print(name, "FAILED", e)
    import json
    ch = json.loads(rows.get("chain", "{}")).get("tok_s")
    if ch:
        for name, r in rows.items():
            d = json.loads(r)
            if d.get("tok_s"): print(f"{name}: {d['tok_s']} tok/s (acc {d.get('accept_len')})  vs chain = {100*(d['tok_s']/ch-1):+.1f}%")


@app.local_entrypoint()
def prof(budget: int = 32, topk: int = 4):
    """Profile chain vs tree to localize the per-forward overhead (sglang torch profiler)."""
    h_chain = profile_decode.spawn(False, 1, 16, "chain")
    h_tree = profile_decode.spawn(True, topk, budget, f"tree_b{budget}_k{topk}")
    for h in (h_chain, h_tree):
        print(h.get()[:1500])


@app.local_entrypoint()
def profov(budget: int = 32, topk: int = 4):
    """Profile chain vs tree with OVERLAP ON (matches the real bench config)."""
    h_chain = profile_decode.spawn(False, 1, 16, "chain_ov", overlap=True)
    h_tree = profile_decode.spawn(True, topk, budget, f"tree_ov_b{budget}_k{topk}", overlap=True)
    for h in (h_chain, h_tree):
        print(h.get()[:1500])


@app.function(image=base_image, volumes={"/results": results_vol})
def reanalyze(tags: list):
    """Proper per-step trace analysis (reuses saved traces, no GPU). For each tag:
    GPU busy (merged intervals per stream), wall, #steps (run_batch annotations),
    per-step GPU-busy vs per-step wall (gap = launch/CPU-bound), phase annotations."""
    import glob, gzip, json
    from collections import defaultdict
    res = {}
    for tag in tags:
        files = sorted(set(glob.glob(f"/results/prof_{tag}/*.trace.json*")))
        if not files: res[tag] = {"error": "no trace"}; continue
        raw = gzip.open(files[-1]).read() if files[-1].endswith(".gz") else open(files[-1], "rb").read()
        ev = json.loads(raw)["traceEvents"]
        streams = defaultdict(list); ann = defaultdict(lambda: [0.0, 0]); kname = defaultdict(float)
        for e in ev:
            if e.get("ph") != "X" or "dur" not in e: continue
            cat = e.get("cat", "")
            if cat == "kernel":
                streams[(e.get("pid"), e.get("tid"))].append((e["ts"], e["ts"] + e["dur"]))
                kname[e["name"][:55]] += e["dur"]
            elif cat in ("user_annotation", "cpu_op"):
                ann[e["name"][:45]][0] += e["dur"]; ann[e["name"][:45]][1] += 1
        # merge intervals on the busiest stream
        def merged(iv):
            iv = sorted(iv); tot = 0.0; cs = ce = None
            for s, e2 in iv:
                if cs is None: cs, ce = s, e2
                elif s <= ce: ce = max(ce, e2)
                else: tot += ce - cs; cs, ce = s, e2
            if cs is not None: tot += ce - cs
            return tot, (iv[0][0] if iv else 0), (max(e2 for _, e2 in iv) if iv else 0)
        best = max(streams.values(), key=lambda iv: sum(b - a for a, b in iv)) if streams else []
        gpu_busy, t0, t1 = merged(best)
        wall = t1 - t0 if t1 > t0 else 1.0
        steps = ann.get("scheduler.run_batch", [0, 0])[1] or ann.get("TARGET_VERIFY", [0, 1])[1] or 1
        res[tag] = {
            "wall_ms": round(wall / 1e3, 1), "gpu_busy_ms": round(gpu_busy / 1e3, 1),
            "gpu_util_pct": round(100 * gpu_busy / wall, 1), "steps": steps,
            "per_step_wall_us": round(wall / steps, 1), "per_step_gpu_us": round(gpu_busy / steps, 1),
            "per_step_idle_us": round((wall - gpu_busy) / steps, 1),
            "top_kernels": sorted(({"k": k, "ms": round(v / 1e3, 1), "us_per_step": round(v / steps, 1)}
                                   for k, v in kname.items()), key=lambda x: -x["ms"])[:14],
            "top_annotations": sorted(({"a": k, "ms": round(v[0] / 1e3, 1), "n": v[1],
                                        "us_per_step": round(v[0] / steps, 1)} for k, v in ann.items()),
                                      key=lambda x: -x["ms"])[:14],
        }
    return res


@app.function(image=base_image, volumes={"/results": results_vol})
def launchcount(tags: list):
    """Count GPU kernel launches per decode step (tree vs chain), plus the
    longest dependent launch chain inside one TARGET_VERIFY window. This is the
    real conc=1 lever: # of serial launches, not individual kernel durations."""
    import glob, gzip, json
    from collections import defaultdict
    res = {}
    for tag in tags:
        files = sorted(set(glob.glob(f"/results/prof_{tag}/*.trace.json*")))
        if not files: res[tag] = {"error": "no trace"}; continue
        raw = gzip.open(files[-1]).read() if files[-1].endswith(".gz") else open(files[-1], "rb").read()
        ev = json.loads(raw)["traceEvents"]
        n_kernels = 0; n_runtime = 0; steps = 0
        kcount = defaultdict(int)
        for e in ev:
            if e.get("ph") != "X": continue
            cat = e.get("cat", "")
            if cat == "kernel":
                n_kernels += 1; kcount[e["name"][:55]] += 1
            elif cat == "cuda_runtime":
                # launch-issuing runtime calls (cudaLaunchKernel etc.)
                if "Launch" in e.get("name", "") or "launch" in e.get("name", ""):
                    n_runtime += 1
            elif cat == "user_annotation" and e.get("name") == "scheduler.run_batch":
                steps += 1
        steps = steps or 1
        res[tag] = {
            "steps": steps,
            "kernels_per_step": round(n_kernels / steps, 1),
            "launches_per_step": round(n_runtime / steps, 1),
            "top_kernel_counts": sorted(({"k": k, "n_per_step": round(v / steps, 1)}
                                         for k, v in kcount.items()),
                                        key=lambda x: -x["n_per_step"])[:25],
        }
    return res


@app.local_entrypoint()
def lc(tags: str = "chain,tree_b32_k4"):
    import json
    r = launchcount.remote(tags.split(","))
    for tag, d in r.items():
        if "error" in d: print(f"\n##### {tag}: {d['error']}"); continue
        print(f"\n##### {tag}: steps={d['steps']} kernels/step={d['kernels_per_step']} "
              f"launches/step={d['launches_per_step']}")
        for k in d["top_kernel_counts"]: print(f"    {k['n_per_step']:>6.1f}/step  {k['k']}")


@app.local_entrypoint()
def reana(tags: str = "chain,tree_b32_k4"):
    import json
    r = reanalyze.remote(tags.split(","))
    for tag, d in r.items():
        if "error" in d: print(f"\n##### {tag}: {d['error']}"); continue
        print(f"\n##### {tag}: util={d['gpu_util_pct']}% steps={d['steps']} "
              f"per-step wall={d['per_step_wall_us']}us gpu={d['per_step_gpu_us']}us idle={d['per_step_idle_us']}us")
        print("  -- top GPU kernels (us/step) --")
        for k in d["top_kernels"][:10]: print(f"    {k['us_per_step']:>8.1f}us/step  {k['k']}")
        print("  -- top annotations (us/step) --")
        for a in d["top_annotations"][:8]: print(f"    {a['us_per_step']:>8.1f}us/step (n={a['n']})  {a['a']}")


@app.function(image=base_image, volumes={"/results": results_vol})
def _list_profsum():
    import os, json
    out = {}
    for fn in sorted(os.listdir("/results")):
        if fn.startswith("profsum_") and fn.endswith(".json"):
            out[fn] = open(f"/results/{fn}").read()
    return out


@app.local_entrypoint()
def profpull():
    import os, json
    os.makedirs("v2_prof", exist_ok=True)
    for fn, c in _list_profsum.remote().items():
        open(f"v2_prof/{fn}", "w").write(c)
        d = json.loads(c)
        print(f"\n===== {fn}: util={d.get('gpu_util_pct')}% wall={d.get('wall_us')}us gpu_busy={d.get('gpu_busy_us')}us =====")
        for k in d.get("top_gpu_kernels", [])[:8]: print(f"  GPU {k['us']:>9.1f}us  {k['name']}")
        for k in d.get("top_cpu_ops", [])[:8]: print(f"  CPU {k['us']:>9.1f}us  {k['name']}")


@app.local_entrypoint()
def grid(target_model: str = "Qwen/Qwen3-8B",
         draft_model: str = "z-lab/Qwen3-8B-DFlash-b16",
         data_names: str = "gsm8k,mt-bench", concurrencies: str = "1"):
    """Sweep (budget x topk) at conc=1 to find best tree config + show topk-scaling.
    Each config = its own GPU container (parallel via spawn)."""
    GRID = [(16, 4), (32, 4), (32, 8), (64, 4), (64, 8),
            (128, 4), (128, 8), (128, 16), (256, 8), (256, 16)]
    handles = [("chain", run.spawn(target_model, draft_model, data_names, False, 1, 16,
                                   None, 32, concurrencies, "flashinfer", 0.85, 1, 1, True,
                                   "grid_chain", False))]
    for b, k in GRID:
        handles.append((f"b{b}_k{k}", run.spawn(
            target_model, draft_model, data_names, True, k, 16, b, 32, concurrencies,
            "flashinfer", 0.85, 1, 1, True, f"grid_b{b}_k{k}", False)))
    os.makedirs("v2_tree_results", exist_ok=True)
    for name, h in handles:
        try:
            res = h.get()
            with open(f"v2_tree_results/grid_{name}.md", "w") as f:
                f.write(res)
            print(f"\n>>> {name}\n{_tables(res)}")
        except Exception as e:
            print(f">>> FAILED {name}: {e}")


@app.local_entrypoint()
def main(target_model: str = "Qwen/Qwen3-8B",
         draft_model: str = "z-lab/Qwen3-8B-DFlash-b16",
         block_size: int = 16):
    """Quick validation: chain vs tree(budget32) at conc=1, gsm8k, cuda graph ON."""
    configs = [("tree_b16", True, 4, 16)]  # budget == block_size: isolate tree verify from decoupling
    os.makedirs("v2_tree_results", exist_ok=True)
    for label, tv, topk, ndt in configs:
        print(f">>> {label}")
        try:
            res = run.remote(target_model, draft_model, "gsm8k", tv, topk,
                             block_size, ndt, 24, "1", "flashinfer", 0.8, 1, 1, True,
                             f"validate_{label}", True)  # disable_cuda_graph=True (decisive mask test)
            with open(f"v2_tree_results/validate_{label}_{target_model.split('/')[-1]}.md", "w") as f:
                f.write(res)
            print(_tables(res))
        except Exception as e:
            print(f">>> FAILED {label}: {e}")


"""Target models for the three-model study (z-lab internal drafts via HF secret)."""
THREE_MODELS = [
    # (target, draft, block_size, tp, mem_fraction, backend)
    # backend: gemma rejects flashinfer (triton supports tree mask); Qwen3.6 is hybrid
    # GDN -> Blackwell full-attn layers need triton/trtllm_mha/fa4 (triton has tree mask).
    # gpt-oss-120b dropped (per request: fa4-cute broken, rejects flashinfer).
    ("Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16", 16, 1, 0.85, "flashinfer"),
    ("google/gemma-4-31b-it", "z-lab/gemma-4-31B-it-DFlash", 16, 1, 0.6, "triton"),
    ("Qwen/Qwen3.6-27B", "z-lab/Qwen3.6-27B-DFlash", 16, 1, 0.6, "triton"),
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
          models: str = "",
          chain_only: bool = False):
    """Run all 3 target models concurrently (one container/card each spawn).

    Per model: chain@blocksize (WITH baseline, same card) + tree at each budget.
    cuda graph stays ON (not disabled). Tree verify step runs eager internally
    (port TODO), base decode + draft + chain verify are graphed.
    """
    sel = set(m.strip() for m in models.split(",") if m.strip())
    os.makedirs("v2_tree_results", exist_ok=True)
    calls = []
    for target, draft, bsz, tp, memf, backend in THREE_MODELS:
        if sel and target.split("/")[-1] not in sel and target not in sel:
            continue
        mtag = target.split("/")[-1]
        # chain + tree budgets; no no-spec baseline (chain is the reference).
        configs = [("chain", False, 1, bsz, True)]
        if not chain_only:
            for b in _budgets_for(bsz):
                configs.append((f"tree_b{b}", True, 4, b, True))
        for label, tv, topk, ndt, skip_bl in configs:
            print(f">>> spawn {mtag}/{label} budget={ndt} backend={backend}")
            calls.append((mtag, label, ndt, run.spawn(
                target, draft, data_names, tv, topk, bsz, ndt, samples_base,
                concurrencies, backend, memf, tp, tp, skip_bl, f"{mtag}_{label}")))
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
