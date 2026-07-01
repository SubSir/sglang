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
    .run_commands("echo v2tree7-reusetreebuf > /tmp/build_time")
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
    # Domino port edits models/dflash.py (GRU projector) -> overlay it too.
    .add_local_file(
        f"{WT}/python/sglang/srt/models/dflash.py",
        remote_path="/tmp/port_models_dflash.py",
        copy=True,
    )
    .add_local_dir("./benchmark", remote_path="/root/benchmark", copy=True)
    .add_local_file("./patch_cudagraph.py", remote_path="/tmp/patch_cudagraph.py", copy=True)
    .run_commands(
        "SGLANG_DIR=$(python3 -c 'import sglang,os;print(os.path.dirname(sglang.__file__))') && "
        "echo \"sglang at $SGLANG_DIR\" && "
        "cp -rf /tmp/port_speculative/. \"$SGLANG_DIR/srt/speculative/\" && "
        "cp -rf /tmp/port_arg_groups/. \"$SGLANG_DIR/srt/arg_groups/\" && "
        "cp -f /tmp/port_models_dflash.py \"$SGLANG_DIR/srt/models/dflash.py\" && "
        "echo overlaid tree-verify + domino port",
        "python3 /tmp/patch_cudagraph.py",
    )
)


# --- fa4 experiment image: cu13 (native cuda-13 for fa4's cutlass DSL) -----------
# Isolated from base_image (cu12) so the running comparison is undisturbed.
#  - libnuma-dev: sgl_kernel's compiled `common_ops` links libnuma; the cu13 image
#    omits the runtime lib -> dlopen fails -> "No module named 'common_ops'".
#  - cutlass payload via --no-deps so numpy/cuda-python aren't upgraded (those
#    upgrades also break sgl_kernel's common_ops).
fa4_image = (
    modal.Image.from_registry(
        "lmsysorg/sglang:nightly-dev-cu13-20260627-13b5bd96"
    )
    .apt_install("git", "wget", "libnuma-dev")
    .run_commands("echo fa4img1 > /tmp/build_time")
    .run_commands(
        "pip install --force-reinstall --no-cache-dir --no-deps "
        "nvidia-cutlass-dsl==4.5.2 nvidia-cutlass-dsl-libs-base==4.5.2"
    )
    .add_local_dir(f"{WT}/python/sglang/srt/speculative",
                   remote_path="/tmp/port_speculative", copy=True)
    .add_local_dir(f"{WT}/python/sglang/srt/arg_groups",
                   remote_path="/tmp/port_arg_groups", copy=True)
    .add_local_dir("./benchmark", remote_path="/root/benchmark", copy=True)
    .add_local_file("./patch_cudagraph.py", remote_path="/tmp/patch_cudagraph.py", copy=True)
    .run_commands(
        "SGLANG_DIR=$(python3 -c 'import sglang,os;print(os.path.dirname(sglang.__file__))') && "
        "cp -rf /tmp/port_speculative/. \"$SGLANG_DIR/srt/speculative/\" && "
        "cp -rf /tmp/port_arg_groups/. \"$SGLANG_DIR/srt/arg_groups/\" && echo overlaid",
        "python3 /tmp/patch_cudagraph.py",
    )
)


@app.function(image=fa4_image, gpu="B200", cloud="aws", timeout=600)
def fa4check():
    """GPU check (needs libcuda.so.1): cutlass + sgl_kernel + sglang import on fa4_image."""
    import subprocess
    checks = [
        ("import cutlass", "import cutlass; print(cutlass.__file__)"),
        ("import cutlass.cute", "import cutlass.cute as cute; print('cute ok')"),
        ("import sglang", "import sglang; print('sglang', sglang.__version__)"),
        ("import sgl_kernel", "import sgl_kernel; print('sgl_kernel ok')"),
        ("dlopen common_ops (real error)",
         "import ctypes; ctypes.CDLL('/usr/local/lib/python3.12/dist-packages/sgl_kernel/sm100/common_ops.abi3.so'); print('dlopen ok')"),
        ("import flashinfer", "import flashinfer; print('flashinfer ok')"),
    ]
    for label, code in checks:
        r = subprocess.run(["python3", "-c", code], capture_output=True, text=True)
        print(f"### {label}\nrc={r.returncode}\nout={r.stdout}\nerr={r.stderr[-1500:]}")
    # what top-level modules does the DSL package actually install?
    r = subprocess.run(["bash", "-c",
        "python3 -c \"import importlib.metadata as m; "
        "print([f for f in m.files('nvidia-cutlass-dsl') if str(f).count('/')<=1][:40])\"; "
        "echo '--- libs-base ---'; "
        "python3 -c \"import importlib.metadata as m; "
        "print([f for f in m.files('nvidia-cutlass-dsl-libs-base') if str(f).count('/')<=1][:40])\"; "
        "echo '--- how sglang fa4 imports cutlass ---'; "
        "grep -rnE 'import cutlass|from cutlass' /usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/ 2>/dev/null | head; "
        "echo '--- how flash_attn_4 / flashinfer import it (the working consumers) ---'; "
        "python3 -c \"import flash_attn_interface\" 2>&1 | tail -2; "
        "grep -rlE 'import cutlass' /usr/local/lib/python3.12/dist-packages/flashinfer/ 2>/dev/null | head -3"],
        capture_output=True, text=True)
    print(f"### module files + import sites\n{r.stdout}\nERR:{r.stderr[-800:]}")
    return "done"


@app.local_entrypoint()
def fa4dbg():
    fa4check.remote()


def _bench_impl(target_model, draft_model, data_names, tree_verify, topk, block_size,
                num_draft_tokens, samples_base, concurrencies, backend,
                mem_fraction, tp, gpu_count, skip_baseline, result_tag,
                disable_cuda_graph, tree_algo, depth_bonus, force_width):
    """Shared bench body (image-agnostic) used by run() [cu12] and fa4_run() [cu13]."""
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


@app.function(gpu="B200", timeout=10800, image=base_image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results_vol})
def run(target_model, draft_model, data_names, tree_verify, topk, block_size,
        num_draft_tokens, samples_base, concurrencies, backend="flashinfer",
        mem_fraction=0.8, tp=1, gpu_count=1, skip_baseline=True, result_tag=None,
        disable_cuda_graph=False, tree_algo="fused", depth_bonus=None, force_width=None):
    """cu12 (validated triton) bench entrypoint. One server launch; sweeps datasets."""
    return _bench_impl(target_model, draft_model, data_names, tree_verify, topk, block_size,
                       num_draft_tokens, samples_base, concurrencies, backend,
                       mem_fraction, tp, gpu_count, skip_baseline, result_tag,
                       disable_cuda_graph, tree_algo, depth_bonus, force_width)


@app.function(gpu="B200", cloud="aws", timeout=10800, image=fa4_image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results_vol})
def fa4_run(target_model, draft_model, data_names, tree_verify, topk, block_size,
            num_draft_tokens, samples_base, concurrencies, backend="fa4",
            mem_fraction=0.8, tp=1, gpu_count=1, skip_baseline=True, result_tag=None,
            disable_cuda_graph=False, tree_algo="fused", depth_bonus=None, force_width=None):
    """cu13 + cutlass image: same bench, for testing the fa4 attention backend."""
    return _bench_impl(target_model, draft_model, data_names, tree_verify, topk, block_size,
                       num_draft_tokens, samples_base, concurrencies, backend,
                       mem_fraction, tp, gpu_count, skip_baseline, result_tag,
                       disable_cuda_graph, tree_algo, depth_bonus, force_width)


@app.local_entrypoint()
def fa4bench(datasets: str = "gsm8k", budget: int = 32, n: int = 80, topk: int = 4,
             block_size: int = 16):
    """Test the fa4 attention backend on the tree verify (cu13 image). Runs chain +
    ours-fused at budget on fa4, conc=1. Compare to the triton/flashinfer numbers."""
    os.makedirs("v2_tree_results", exist_ok=True)
    configs = [("chain", False, 1, None), ("ours_fused", True, topk, budget)]
    handles = []
    for ds in [d.strip() for d in datasets.split(",")]:
        for label, tv, tk, ndt in configs:
            tag = f"fa4_{label}_{ds}"
            print(f">>> spawn {tag} (backend=fa4)")
            h = fa4_run.spawn("Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16", ds, tv, tk,
                              block_size, ndt, n, "1", "fa4", 0.85, 1, 1, True, tag, False, "fused")
            handles.append((tag, h))
    for tag, h in handles:
        try:
            res = h.get()
            with open(f"v2_tree_results/{tag}.md", "w") as f:
                f.write(res)
            print(f">>> DONE {tag}")
            print(_tables(res))
        except Exception as e:
            print(f">>> FAILED {tag}: {e}")


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


@app.local_entrypoint()
def domino(datasets: str = "gsm8k,math500", n: int = 80, block_size: int = 16,
           backend: str = "triton"):
    """Domino GRU-rollout chain vs z-lab DFlash block-parallel chain, OUR v2 engine.

    conc=1, CHAIN verify. Reports accept_len + tok/s for both drafts on each
    dataset. gsm8k accept is CONTAMINATED for Domino (training overlap) -> read
    math500 for the fair ~0.2 acc signal. Net tok/s captures GRU rollout overhead.
    """
    os.makedirs("v2_tree_results", exist_ok=True)
    # (label, draft_model). Both chain (tree_verify=False, topk=1, ndt=None).
    drafts = [
        ("zlab_dflash", "z-lab/Qwen3-8B-DFlash-b16"),
        ("domino", "Huang2020/Qwen3-8B-Domino-b16"),
    ]
    handles = []
    for ds in [d.strip() for d in datasets.split(",")]:
        for label, draft in drafts:
            tag = f"domino_{label}_{ds}"
            print(f">>> spawn {tag} (draft={draft}, backend={backend}, chain)")
            h = run.spawn("Qwen/Qwen3-8B", draft, ds, False, 1, block_size,
                          None, n, "1", backend, 0.8, 1, 1, True, tag, False, "fused")
            handles.append((tag, h))
    for tag, h in handles:
        try:
            res = h.get()
            with open(f"v2_tree_results/{tag}.md", "w") as f:
                f.write(res)
            print(f">>> DONE {tag}")
            print(_tables(res))
        except Exception as e:
            print(f">>> FAILED {tag}: {e}")


@app.local_entrypoint()
def domino_time(dataset: str = "gsm8k", n: int = 40, block_size: int = 16,
                backend: str = "triton"):
    """Isolate the draft-step cost: CUDA-event us/step for GRU rollout vs DFlash
    sample (SGLANG_DOMINO_TIME_DRAFT=1). Look for [DRAFT-TIME] lines in logs."""
    os.makedirs("v2_tree_results", exist_ok=True)
    drafts = [
        ("zlab_dflash", "z-lab/Qwen3-8B-DFlash-b16"),
        ("domino", "Huang2020/Qwen3-8B-Domino-b16"),
    ]
    handles = []
    for label, draft in drafts:
        tag = f"dtime_{label}_{dataset}"
        print(f">>> spawn {tag} (time-draft, draft={draft})")
        h = time_run.spawn("Qwen/Qwen3-8B", draft, dataset, block_size, n, backend, tag)
        handles.append((tag, h))
    for tag, h in handles:
        try:
            print(f">>> DONE {tag}\n{h.get()}")
        except Exception as e:
            print(f">>> FAILED {tag}: {e}")


@app.function(gpu="B200", timeout=3600, image=base_image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results_vol})
def time_run(target_model, draft_model, data_name, block_size, n, backend, result_tag):
    """Run the chain bench with SGLANG_DOMINO_TIME_DRAFT=1 so the worker logs
    mean us/step for the draft generation. Returns the captured [DRAFT-TIME] lines."""
    import sys, importlib.util, io, contextlib
    os.environ["SGLANG_DFLASH_TREE_VERIFY"] = "0"
    os.environ["SGLANG_DFLASH_TREE_ALGO"] = "fused"
    os.environ["DFLASH_DRAFT_ATTN_BACKEND"] = backend
    os.environ["SGLANG_DOMINO_TIME_DRAFT"] = "1"
    args = [
        "bench_dflash_sweep.py", "--data-names", data_name,
        "--target-model", target_model, "--draft-model", draft_model,
        "--tp-sizes", "1", "--concurrencies", "1",
        "--samples-per-concurrency-base", str(n),
        "--max-samples-per-config", str(n * 4), "--max-new-tokens", "512",
        "--attention-backends", backend, "--mem-fraction-static", "0.8",
        "--speculative-eagle-topk", "1", "--speculative-dflash-block-size", str(block_size),
        "--max-running-requests", "1", "--skip-baseline",
        "--output-md", "/root/out_time.md",
    ]
    os.chdir("/root")
    sys.argv = args
    spec = importlib.util.spec_from_file_location(
        "bench", "/root/benchmark/dflash/bench_dflash_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    spec.loader.exec_module(mod)
    mod.main()
    with open("/root/out_time.md") as f:
        table = _tables(f.read())
    return f"(see [DRAFT-TIME] lines in container logs)\n{table}"


@app.local_entrypoint()
def domino_opt(datasets: str = "math500", n: int = 80, backend: str = "triton"):
    """Optimization sweep: Domino chain with candidate_pool in {2048(default),0(full-vocab
    fused, skips per-step eager topk)}. Lossless check = accept_len must match. Reports
    tok/s + accept for each pool on each dataset, vs the z-lab DFlash baseline."""
    os.makedirs("v2_tree_results", exist_ok=True)
    DOM = "Huang2020/Qwen3-8B-Domino-b16"
    handles = []
    for ds in [d.strip() for d in datasets.split(",")]:
        for pool in ("2048", "0"):
            tag = f"dopt_pool{pool}_{ds}"
            print(f">>> spawn {tag} (cand_pool={pool})")
            h = run_env.spawn("Qwen/Qwen3-8B", DOM, ds, n, backend, tag,
                              {"DFLASH_DOMINO_CANDIDATE_POOL": pool})
            handles.append((tag, h))
    for tag, h in handles:
        try:
            res = h.get()
            with open(f"v2_tree_results/{tag}.md", "w") as f:
                f.write(res)
            print(f">>> DONE {tag}")
            print(_tables(res))
        except Exception as e:
            print(f">>> FAILED {tag}: {e}")


@app.function(gpu="B200", timeout=10800, image=base_image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results_vol})
def ab_samecard(datasets: str = "math500,mbpp", n: int = 128, backend: str = "triton",
                concs: str = "1"):
    """SAME-CARD A/B across CONCURRENCY: one container, sequential server launches —
    z-lab DFlash chain then optimized Domino chain (cand_pool=0). For each (dataset,
    conc) reports tok/s, accept, per-FORWARD ms. Same B200. concs = "1,8,32"."""
    import subprocess, time, signal, json, urllib.request, os as _os
    from concurrent.futures import ThreadPoolExecutor
    from datasets import load_dataset
    from transformers import AutoTokenizer
    target = "Qwen/Qwen3-8B"
    DOM = "Huang2020/Qwen3-8B-Domino-b16"; ZLAB = "z-lab/Qwen3-8B-DFlash-b16"
    conc_list = [int(c) for c in str(concs).split(",")]
    max_conc = max(conc_list)
    print("GPU:", subprocess.run(["nvidia-smi","--query-gpu=name,uuid","--format=csv,noheader"],
          capture_output=True, text=True).stdout.strip(), flush=True)
    tok = AutoTokenizer.from_pretrained(target)
    fmt = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    def load(ds_name):
        # Return the FULL split; bench() cycles it to reach n=max(1024,conc*32).
        if ds_name == "math500":
            d = load_dataset("HuggingFaceH4/MATH-500", split="test")
            return [x["problem"] for x in d]
        if ds_name == "mbpp":
            d = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
            return [x["prompt"] for x in d]
        d = load_dataset("openai/gsm8k","main",split="test")
        return [x["question"] for x in d]

    def launch(draft, pool):
        env = dict(_os.environ)
        env["SGLANG_DFLASH_TREE_VERIFY"] = "0"; env["SGLANG_DFLASH_BLOCK_VERIFY"] = "1"
        env["DFLASH_DRAFT_ATTN_BACKEND"] = backend
        if pool is not None: env["DFLASH_DOMINO_CANDIDATE_POOL"] = str(pool)
        server = ["python","-m","sglang.launch_server","--model-path",target,"--trust-remote-code",
                  "--mem-fraction-static","0.85","--max-running-requests",str(max_conc),
                  "--cuda-graph-max-bs",str(max_conc),
                  "--attention-backend",backend,"--speculative-algorithm","DFLASH",
                  "--speculative-draft-model-path",draft,"--speculative-dflash-block-size","16",
                  "--speculative-eagle-topk","1","--port","30000"]
        srv = subprocess.Popen(server, env=env)
        for _ in range(300):
            try: urllib.request.urlopen("http://127.0.0.1:30000/health",timeout=2); return srv
            except Exception: time.sleep(2)
        srv.send_signal(signal.SIGINT); return None

    def send(p):
        req=urllib.request.Request("http://127.0.0.1:30000/generate",
            data=json.dumps({"text":p,"sampling_params":{"temperature":0.0,"max_new_tokens":512}}).encode(),
            headers={"Content-Type":"application/json"})
        return json.loads(urllib.request.urlopen(req,timeout=900).read())

    def bench(base_prompts, conc):
        # n requests = max(1024, conc*32) so the batch stays saturated in steady state;
        # cycle the dataset prompts to reach that count.
        nreq = max(1024, conc * 32)
        prompts = [base_prompts[i % len(base_prompts)] for i in range(nreq)]
        send(base_prompts[0]); send(base_prompts[1])  # warmup
        t0=time.perf_counter(); toks=0; fwd=0; accs=[]
        with ThreadPoolExecutor(max_workers=conc) as ex:
            for r in ex.map(send, prompts):
                m=r["meta_info"]; toks+=m["completion_tokens"]; vc=m.get("spec_verify_ct")
                if vc: fwd+=vc; accs.append(m["completion_tokens"]/vc)
        dt=time.perf_counter()-t0
        # batched_step_ms ~= wall of one batched decode step. With a ~full batch the
        # number of batched decode steps ~= total_forward_ct / conc, so
        # batched_step_ms = conc * wall / total_forward_ct.
        batched_step_ms = round(conc * dt / fwd * 1000, 3) if fwd else None
        return {"tok_s":round(toks/dt,1),"accept":round(sum(accs)/len(accs),3) if accs else None,
                "per_forward_ms":round(dt/fwd*1000,3) if fwd else None,
                "batched_step_ms":batched_step_ms,"forward_ct":fwd,"nreq":nreq,"secs":round(dt,1)}

    out={}
    for ds_name in [d.strip() for d in datasets.split(",")]:
        prompts=[tok.apply_chat_template([{"role":"user","content":fmt.format(q=q)}],
                 tokenize=False,add_generation_prompt=True,enable_thinking=False) for q in load(ds_name)]
        out[ds_name]={}
        for label, draft, pool in (("dflash",ZLAB,None),("domino_opt",DOM,0)):
            srv = launch(draft, pool)
            if srv is None:
                for c in conc_list: out[ds_name].setdefault(str(c),{})[label]={"error":"server down"}
                continue
            for c in conc_list:
                res = bench(prompts, c)
                out[ds_name].setdefault(str(c),{})[label]=res
                print(f"  [{ds_name} conc={c}] {label}: {res}", flush=True)
            srv.send_signal(signal.SIGINT); time.sleep(4)
            try: srv.wait(timeout=10)
            except Exception: srv.kill()
            time.sleep(3)
        print(f"\n### {ds_name} SAME-CARD conc sweep (n=max(1024,conc*32))", flush=True)
        print(f"  {'conc':>4} {'DF step ms':>10} {'DO step ms':>10} {'abs ovh ms':>10} "
              f"{'rel ovh %':>9} {'acc gain %':>10} {'net tok/s %':>11}", flush=True)
        for c in conc_list:
            df=out[ds_name][str(c)].get("dflash",{}); do=out[ds_name][str(c)].get("domino_opt",{})
            if df.get("tok_s") and do.get("tok_s"):
                dfs=df["batched_step_ms"]; dos=do["batched_step_ms"]
                abs_ovh=dos-dfs; rel_ovh=abs_ovh/dfs*100
                acc_gain=(do["accept"]/df["accept"]-1)*100
                net=(do["tok_s"]/df["tok_s"]-1)*100
                print(f"  {c:>4} {dfs:>10.3f} {dos:>10.3f} {abs_ovh:>+10.3f} "
                      f"{rel_ovh:>+8.1f}% {acc_gain:>+9.1f}% {net:>+10.1f}%", flush=True)
    json.dump(out, open("/results/ab_concsweep.json","w"), indent=2); results_vol.commit()
    return json.dumps(out, indent=2)


@app.local_entrypoint()
def absame(datasets: str = "math500,mbpp", concs: str = "1,8,32,64"):
    print(ab_samecard.remote(datasets, 1, "triton", concs))


@app.local_entrypoint()
def domino_final(datasets: str = "math500,mt-bench,mbpp", n: int = 80, backend: str = "triton"):
    """FINAL: optimized Domino chain (cand_pool=0 + out= graph-buffer writes) vs z-lab
    DFlash chain, conc=1, n>=80, per clean dataset. Reports tok/s + accept for both."""
    os.makedirs("v2_tree_results", exist_ok=True)
    DOM = "Huang2020/Qwen3-8B-Domino-b16"; ZLAB = "z-lab/Qwen3-8B-DFlash-b16"
    handles = []
    for ds in [d.strip() for d in datasets.split(",")]:
        h_df = run.spawn("Qwen/Qwen3-8B", ZLAB, ds, False, 1, 16, None, n, "1",
                         backend, 0.8, 1, 1, True, f"fin_dflash_{ds}", False, "fused")
        h_do = run_env.spawn("Qwen/Qwen3-8B", DOM, ds, n, backend, f"fin_domino_{ds}",
                             {"DFLASH_DOMINO_CANDIDATE_POOL": "0"})
        handles += [(f"fin_dflash_{ds}", h_df), (f"fin_domino_{ds}", h_do)]
    for tag, h in handles:
        try:
            res = h.get()
            with open(f"v2_tree_results/{tag}.md", "w") as f:
                f.write(res)
            print(f">>> DONE {tag}")
            print(_tables(res))
        except Exception as e:
            print(f">>> FAILED {tag}: {e}")


@app.function(gpu="B200", timeout=10800, image=base_image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results_vol})
def run_env(target_model, draft_model, data_names, n, backend, result_tag, extra_env):
    """Chain bench (conc=1) with extra env vars applied (for opt sweeps)."""
    for k, v in (extra_env or {}).items():
        os.environ[k] = str(v)
    return _bench_impl(target_model, draft_model, data_names, False, 1, 16,
                       None, n, "1", backend, 0.8, 1, 1, True, result_tag,
                       False, "fused", None, None)


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


@app.function(gpu="B200", timeout=21600, image=base_image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results_vol})
def treecmp_run(dataset, samples):
    """SAME-CARD (one container per dataset): chain / ours-b32-t4 / ours-b128-t8 /
    ours-b128-t16 / ddtree-b32-w16, sequentially, with SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1.
    1024 samples, conc=1, triton verify. Each config = its own server launch in THIS
    container, so all methods share one physical B200."""
    import subprocess
    BENCH = "/root/benchmark/dflash/bench_dflash_sweep.py"
    # (label, tree_verify, topk, budget, algo, force_width)
    configs = [
        ("chain",          "0", 1,  None, "fused",  None),
        ("ours_b32_t4",    "1", 4,  32,   "fused",  None),
        ("ours_b128_t8",   "1", 8,  128,  "fused",  None),
        ("ours_b128_t16",  "1", 16, 128,  "fused",  None),
        ("ddtree_b32_w16", "1", 4,  32,   "ddtree", 16),
    ]
    out = {}
    for label, tv, topk, budget, algo, fw in configs:
        env = dict(os.environ)
        env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"   # <-- requested
        env["SGLANG_DFLASH_TREE_VERIFY"] = tv
        env["SGLANG_DFLASH_TREE_ALGO"] = algo
        env["DFLASH_DRAFT_ATTN_BACKEND"] = "triton"
        if fw is not None:
            env["SGLANG_DDTREE_FORCE_WIDTH"] = str(fw)
        else:
            env.pop("SGLANG_DDTREE_FORCE_WIDTH", None)
        md = f"/results/tcmp_{label}_{dataset}.md"
        args = ["python3", BENCH, "--data-names", dataset,
                "--target-model", "Qwen/Qwen3-8B",
                "--draft-model", "z-lab/Qwen3-8B-DFlash-b16",
                "--tp-sizes", "1", "--concurrencies", "1",
                "--samples-per-concurrency-base", str(samples),
                "--max-samples-per-config", str(samples * 64),
                "--max-new-tokens", "1024", "--attention-backends", "triton",
                "--mem-fraction-static", "0.85",
                "--speculative-dflash-block-size", "16",
                "--max-running-requests", "1", "--skip-baseline",
                "--speculative-eagle-topk", str(topk), "--output-md", md]
        if budget is not None:
            args += ["--speculative-num-draft-tokens", str(budget)]
        print(f">>> [{dataset}] {label}  OVERLAP_PLAN_STREAM=1 topk={topk} budget={budget} algo={algo} fw={fw}", flush=True)
        p = subprocess.run(args, cwd="/root", env=env, text=True,
                           capture_output=True)
        tail = "\n".join((p.stdout + p.stderr).splitlines()[-8:])
        print(f">>> [{dataset}] {label} rc={p.returncode}\n{tail}", flush=True)
        try:
            out[label] = open(md).read()
        except Exception as e:
            out[label] = f"ERR rc={p.returncode}: {tail}"
        results_vol.commit()
    return {"dataset": dataset, "labels": list(out.keys())}


@app.local_entrypoint()
def treecmp(datasets: str = "gsm8k,math500,mt-bench", samples: int = 1024):
    """Same-card tree compare with OVERLAP_PLAN_STREAM + ours-b128. One container per
    dataset (all methods share that card). Detach-friendly: `modal run --detach`."""
    hs = [(ds, treecmp_run.spawn(ds, samples)) for ds in datasets.split(",")]
    for ds, h in hs:
        print(f"=== {ds} ==="); print(h.get())


@app.local_entrypoint()
def compare(datasets: str = "gsm8k,math500,mt-bench", budget: int = 32, n: int = 128,
            backend: str = "triton", topk: int = 4, block_size: int = 16,
            target_model: str = "Qwen/Qwen3-8B",
            draft_model: str = "z-lab/Qwen3-8B-DFlash-b16"):
    """Four-way comparison (sglang side: chain / ours-fused / ddtree) at conc=1,
    best budget, triton verify, across datasets. n samples per dataset (mt-bench
    caps at its 80). Throughput + accept. Detach-friendly: spawns all 3 in parallel.
    JetSpec side runs separately via modal_vllm_jetspec.py::compare."""
    os.makedirs("v2_tree_results", exist_ok=True)
    configs = [
        ("chain",      False, 1,    None,   "fused"),
        ("ours_fused", True,  topk, budget, "fused"),
        ("ddtree",     True,  topk, budget, "ddtree"),
    ]
    # Spawn EVERY (method x dataset) as its own AWS B200 container -> max parallelism.
    # Same GPU model + same cloud (aws) controls hardware; conc=1 greedy is deterministic.
    handles = []
    for ds in [d.strip() for d in datasets.split(",")]:
        for label, tv, tk, ndt, algo in configs:
            tag = f"cmp_{label}_{ds}"
            print(f">>> spawn {tag} budget={budget} n={n} backend={backend}")
            h = run.spawn(target_model, draft_model, ds, tv, tk, block_size,
                          ndt, n, "1", backend, 0.85, 1, 1, True,
                          tag, False, algo)
            handles.append((tag, h))
    for tag, h in handles:
        try:
            res = h.get()
            with open(f"v2_tree_results/{tag}.md", "w") as f:
                f.write(res)
            print(f">>> DONE {tag}")
        except Exception as e:
            print(f">>> FAILED {tag}: {e}")


@app.local_entrypoint()
def kbench(dataset: str = "gsm8k", n: int = 80, topk: int = 4, block_size: int = 16):
    """Verify-kernel x budget: does fa4 handle the big (b128) tree better than triton/
    flashinfer? If a kernel keeps b128 fast, b128's higher acc -> higher throughput
    (jetspec-style). ours-fused, conc=1. fa4 runs on cu13 fa4_image; others on cu12."""
    os.makedirs("v2_tree_results", exist_ok=True)
    combos = [  # (backend, budget, fn)
        ("triton", 32, run), ("triton", 128, run),
        ("flashinfer", 128, run),
        ("fa4", 32, fa4_run), ("fa4", 128, fa4_run),
    ]
    handles = []
    for bk, bud, fn in combos:
        tag = f"kb_{dataset}_{bk}_b{bud}"
        h = fn.spawn("Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16", dataset, True, topk,
                     block_size, bud, n, "1", bk, 0.85, 1, 1, True, tag, False, "fused")
        handles.append((tag, h))
    for tag, h in handles:
        try:
            res = h.get()
            with open(f"v2_tree_results/{tag}.md", "w") as f:
                f.write(res)
            print(f">>> DONE {tag}"); print(_tables(res))
        except Exception as e:
            print(f">>> FAILED {tag}: {e}")


@app.local_entrypoint()
def bsweep(dataset: str = "gsm8k", budgets: str = "32,48,64", n: int = 128,
           topk: int = 4, block_size: int = 16):
    """ours-fused budget sweep at conc=1 (triton). Tests whether a bigger verify tree
    raises acc ~for free (memory-bound 8B verify) -> higher throughput. Each its own
    B200 (unpinned). budget = num_draft_tokens."""
    os.makedirs("v2_tree_results", exist_ok=True)
    handles = []
    for b in [int(x) for x in budgets.split(",")]:
        tag = f"bsw_{dataset}_b{b}"
        h = run.spawn("Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16", dataset, True, topk,
                      block_size, b, n, "1", "triton", 0.85, 1, 1, True, tag, False, "fused")
        handles.append((b, h))
    for b, h in handles:
        try:
            res = h.get()
            with open(f"v2_tree_results/bsw_{dataset}_b{b}.md", "w") as f:
                f.write(res)
            print(f">>> DONE b{b}"); print(_tables(res))
        except Exception as e:
            print(f">>> FAILED b{b}: {e}")


@app.local_entrypoint()
def ddtree_width(dataset: str = "gsm8k", budget: int = 32, n: int = 80,
                 widths: str = "6,16,32", topk: int = 4, block_size: int = 16):
    """Sweep ddtree's build/sample width (force_width). Official DDTree uses
    topk=min(budget,V) i.e. width≈budget; our default caps at ~3-6. Tests whether a
    faithful wide DDTree heap gets higher accept than ours-fused. triton verify, conc=1."""
    os.makedirs("v2_tree_results", exist_ok=True)
    handles = []
    for w in [x.strip() for x in widths.split(",")]:
        tag = f"ddw_{dataset}_fw{w}"
        print(f">>> spawn {tag} (ddtree force_width={w})")
        h = run.spawn("Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16", dataset, True, topk,
                      block_size, budget, n, "1", "triton", 0.85, 1, 1, True, tag, False,
                      "ddtree", None, int(w))
        handles.append((w, h))
    for w, h in handles:
        try:
            res = h.get()
            with open(f"v2_tree_results/ddw_{dataset}_fw{w}.md", "w") as f:
                f.write(res)
            print(f">>> DONE fw{w}")
            print(_tables(res))
        except Exception as e:
            print(f">>> FAILED fw{w}: {e}")


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
    # Spawn all configs concurrently (each its own GPU container) instead of serial
    # .remote() — wall-clock ~= one config's boot+run, not the sum of all seven.
    handles = []
    for label, tv, tk, ndt, algo in configs:
        print(f">>> spawn {label} algo={algo} conc={concurrencies} backend={backend}")
        h = run.spawn(target_model, draft_model, data_names, tv, tk,
                      block_size, ndt, 32, concurrencies, backend,
                      0.85, 1, 1, True, f"ddt_{label}", False, algo)
        handles.append((label, h))
    for label, h in handles:
        try:
            res = h.get()
            with open(f"v2_tree_results/ddt_{label}_{data_names.replace(',','-')}.md", "w") as f:
                f.write(res)
            print(f">>> DONE {label}")
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
                   overlap: bool = False, backend: str = "flashinfer"):
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
    env["DFLASH_DRAFT_ATTN_BACKEND"] = backend
    spec = ["--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", draft,
            "--speculative-dflash-block-size", str(block_size),
            "--speculative-eagle-topk", str(topk if tree_verify else 1)]
    if tree_verify:
        spec += ["--speculative-num-draft-tokens", str(budget)]
        if not overlap:
            spec += ["--disable-overlap-schedule"]
    server = ["python", "-m", "sglang.launch_server", "--model-path", target,
              "--trust-remote-code", "--mem-fraction-static", "0.85",
              "--max-running-requests", "1", "--attention-backend", backend,
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


@app.function(gpu="B200", timeout=10800, image=base_image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results_vol})
def cgbench(cudagraph: bool, tag: str,
            target="Qwen/Qwen3-8B", draft="z-lab/Qwen3-8B-DFlash-b16",
            block_size: int = 16, topk: int = 4, budget: int = 32, n: int = 128):
    """conc=1 tree bench for the SGLANG_DFLASH_TREE_CUDAGRAPH opt (triton verify,
    overlap-plan-stream on, per the tree-cudagraph task). Captures full output TEXT
    per request so the flag-on run can be diffed against flag-off for LOSSLESS
    confirmation (accept + output tokens identical). Also reports tok/s."""
    import subprocess, time, signal, json, urllib.request
    env = dict(os.environ)
    env["SGLANG_DFLASH_TREE_VERIFY"] = "1"
    env["SGLANG_DFLASH_TREE_ALGO"] = "fused"
    env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"
    env["DFLASH_DRAFT_ATTN_BACKEND"] = "triton"
    env["SGLANG_DFLASH_TREE_CUDAGRAPH"] = "1" if cudagraph else "0"
    spec = ["--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", draft,
            "--speculative-dflash-block-size", str(block_size),
            "--speculative-eagle-topk", str(topk),
            "--speculative-num-draft-tokens", str(budget)]
    server = ["python", "-m", "sglang.launch_server", "--model-path", target,
              "--trust-remote-code", "--mem-fraction-static", "0.85",
              "--max-running-requests", "1", "--attention-backend", "triton",
              *spec, "--port", "30000"]
    print(">>> CUDAGRAPH=", cudagraph, " ".join(server), flush=True)
    srv = subprocess.Popen(server, env=env)
    ok = False
    for _ in range(240):
        try: urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=2); ok = True; break
        except Exception: time.sleep(2)
    out = {"tag": tag, "ready": ok, "cudagraph": cudagraph}
    if ok:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(target)
        ds = None
        for _ in range(5):
            try: ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n)); break
            except Exception: time.sleep(10)
        fmt = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        ps = [tk.apply_chat_template([{"role": "user", "content": fmt.format(q=ds[i]["question"])}],
              tokenize=False, add_generation_prompt=True, enable_thinking=False) for i in range(n)]
        def send(p):
            req = urllib.request.Request("http://127.0.0.1:30000/generate",
                data=json.dumps({"text": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": 512}}).encode(),
                headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=600).read())
        send(ps[0]); send(ps[1])  # warmup
        t0 = time.perf_counter(); toks = 0; accs = []; texts = []
        for p in ps:
            r = send(p); m = r["meta_info"]
            toks += m["completion_tokens"]
            texts.append(r["text"])
            if m.get("spec_verify_ct"): accs.append(m["completion_tokens"] / m["spec_verify_ct"])
        dt = time.perf_counter() - t0
        out["tok_s"] = round(toks / dt, 1); out["accept_len"] = round(sum(accs)/len(accs), 4) if accs else None
        out["tokens"] = toks; out["secs"] = round(dt, 2)
        # Hash of the concatenated outputs = exact lossless fingerprint.
        import hashlib
        out["out_hash"] = hashlib.sha256("\x1e".join(texts).encode()).hexdigest()[:16]
    srv.send_signal(signal.SIGINT); time.sleep(2)
    json.dump(out, open(f"/results/cg_{tag}.json", "w")); results_vol.commit()
    print(json.dumps({k: out.get(k) for k in ("tag","ready","cudagraph","tok_s","accept_len","tokens","out_hash")}), flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def cg(n: int = 128, budget: int = 32, topk: int = 4):
    """Tree-cudagraph opt: run ours-b{budget} triton+overlap with the flag OFF and ON,
    same card sequence. LOSSLESS = accept_len + out_hash must match. Then compare tok/s."""
    import json
    hoff = cgbench.spawn(False, f"off_b{budget}", topk=topk, budget=budget, n=n)
    hon = cgbench.spawn(True, f"on_b{budget}", topk=topk, budget=budget, n=n)
    off = json.loads(hoff.get()); on = json.loads(hon.get())
    print("OFF:", {k: off.get(k) for k in ("tok_s","accept_len","tokens","out_hash")})
    print("ON :", {k: on.get(k) for k in ("tok_s","accept_len","tokens","out_hash")})
    lossless = (off.get("out_hash") == on.get("out_hash")
                and off.get("accept_len") == on.get("accept_len")
                and off.get("tokens") == on.get("tokens"))
    print("LOSSLESS:", lossless, "(out_hash + accept_len + tokens identical)")
    if off.get("tok_s") and on.get("tok_s"):
        print(f"tok/s  OFF {off['tok_s']} -> ON {on['tok_s']}  = {100*(on['tok_s']/off['tok_s']-1):+.1f}%")


@app.function(gpu="B200", timeout=10800, image=base_image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results_vol})
def cgbench_same(target="Qwen/Qwen3-8B", draft="z-lab/Qwen3-8B-DFlash-b16",
                 block_size: int = 16, topk: int = 4, budget: int = 32, n: int = 256,
                 overlap: bool = True):
    """SAME-CARD: in ONE container, run flag OFF then ON back-to-back -> clean
    cuda-graph tok/s delta (no cross-container noise). Lossless via out_hash."""
    import subprocess, time, signal, json, urllib.request, hashlib
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(target)
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    fmt = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    ps = [tk.apply_chat_template([{"role": "user", "content": fmt.format(q=ds[i]["question"])}],
          tokenize=False, add_generation_prompt=True, enable_thinking=False) for i in range(n)]
    def run(cudagraph):
        env = dict(os.environ)
        env["SGLANG_DFLASH_TREE_VERIFY"] = "1"; env["SGLANG_DFLASH_TREE_ALGO"] = "fused"
        env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1" if overlap else "0"
        env["DFLASH_DRAFT_ATTN_BACKEND"] = "triton"
        # "on" = fold-in (topk head + tree build in one graph fed by draft-graph
        # hidden buffer); "off" = fully eager. Separate-graph Option A stays off.
        env["SGLANG_DFLASH_TREE_IN_DRAFT_GRAPH"] = "1" if cudagraph else "0"
        env["SGLANG_DFLASH_TREE_CUDAGRAPH"] = "0"
        server = ["python", "-m", "sglang.launch_server", "--model-path", target,
                  "--trust-remote-code", "--mem-fraction-static", "0.85",
                  "--max-running-requests", "1", "--attention-backend", "triton",
                  "--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", draft,
                  "--speculative-dflash-block-size", str(block_size),
                  "--speculative-eagle-topk", str(topk), "--speculative-num-draft-tokens", str(budget),
                  "--port", "30000"]
        srv = subprocess.Popen(server, env=env); ok = False
        for _ in range(240):
            try: urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=2); ok = True; break
            except Exception: time.sleep(2)
        def send(p):
            req = urllib.request.Request("http://127.0.0.1:30000/generate",
                data=json.dumps({"text": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": 512}}).encode(),
                headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=600).read())
        send(ps[0]); send(ps[1])
        t0 = time.perf_counter(); toks = 0; accs = []; texts = []
        for p in ps:
            r = send(p); m = r["meta_info"]; toks += m["completion_tokens"]; texts.append(r["text"])
            if m.get("spec_verify_ct"): accs.append(m["completion_tokens"]/m["spec_verify_ct"])
        dt = time.perf_counter() - t0
        srv.send_signal(signal.SIGINT); time.sleep(5)
        try: srv.wait(timeout=15)
        except Exception: srv.kill()
        time.sleep(3)
        return {"cudagraph": cudagraph, "tok_s": round(toks/dt, 1),
                "accept_len": round(sum(accs)/len(accs), 4) if accs else None, "tokens": toks,
                "out_hash": hashlib.sha256("\x1e".join(texts).encode()).hexdigest()[:16]}
    off = run(False); on = run(True)
    lossless = (off["out_hash"] == on["out_hash"] and off["accept_len"] == on["accept_len"] and off["tokens"] == on["tokens"])
    delta = 100*(on["tok_s"]/off["tok_s"]-1) if off["tok_s"] else None
    res = {"overlap": overlap, "off": off, "on": on, "lossless": lossless, "delta_pct": round(delta, 1) if delta else None}
    print("SAME-CARD RESULT:", json.dumps(res), flush=True)
    json.dump(res, open(f"/results/cgsame_ov{int(overlap)}_b{budget}.json", "w")); results_vol.commit()
    return json.dumps(res)


@app.function(gpu="B200", timeout=10800, image=base_image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results_vol})
def tcsame(target="Qwen/Qwen3-8B", draft="z-lab/Qwen3-8B-DFlash-b16",
           block_size: int = 16, budget: int = 32, topk: int = 4, n: int = 128):
    """SAME-CARD chain vs tree(ours-b{budget}): one container back-to-back, triton +
    overlap on, conc=1. Reports tok/s + accept + PER-STEP wall ms (dt/total_verify_ct)."""
    import subprocess, time, signal, json, urllib.request
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(target)
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    fmt = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    ps = [tk.apply_chat_template([{"role": "user", "content": fmt.format(q=ds[i]["question"])}],
          tokenize=False, add_generation_prompt=True, enable_thinking=False) for i in range(n)]
    def run(is_tree):
        env = dict(os.environ)
        env["SGLANG_DFLASH_TREE_VERIFY"] = "1" if is_tree else "0"
        env["SGLANG_DFLASH_TREE_ALGO"] = "fused"
        env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"
        env["DFLASH_DRAFT_ATTN_BACKEND"] = "triton"
        spec = ["--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", draft,
                "--speculative-dflash-block-size", str(block_size),
                "--speculative-eagle-topk", str(topk if is_tree else 1)]
        if is_tree: spec += ["--speculative-num-draft-tokens", str(budget)]
        server = ["python", "-m", "sglang.launch_server", "--model-path", target,
                  "--trust-remote-code", "--mem-fraction-static", "0.85",
                  "--max-running-requests", "1", "--attention-backend", "triton", *spec, "--port", "30000"]
        srv = subprocess.Popen(server, env=env)
        for _ in range(240):
            try: urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=2); break
            except Exception: time.sleep(2)
        def send(p):
            req = urllib.request.Request("http://127.0.0.1:30000/generate",
                data=json.dumps({"text": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": 512}}).encode(),
                headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=600).read())
        send(ps[0]); send(ps[1])
        t0 = time.perf_counter(); toks = 0; vct = 0; accs = []
        for p in ps:
            r = send(p); m = r["meta_info"]; toks += m["completion_tokens"]
            v = m.get("spec_verify_ct")
            if v: vct += v; accs.append(m["completion_tokens"]/v)
        dt = time.perf_counter() - t0
        srv.send_signal(signal.SIGINT); time.sleep(5)
        try: srv.wait(timeout=15)
        except Exception: srv.kill()
        time.sleep(3)
        return {"is_tree": is_tree, "tok_s": round(toks/dt, 1),
                "accept_len": round(sum(accs)/len(accs), 4) if accs else None,
                "step_ms": round(dt*1000/vct, 4) if vct else None, "verify_ct": vct, "tokens": toks}
    ch = run(False); tr = run(True)
    res = {"chain": ch, "tree": tr,
           "tok_s_gain_pct": round(100*(tr["tok_s"]/ch["tok_s"]-1), 1),
           "accept_gain_pct": round(100*(tr["accept_len"]/ch["accept_len"]-1), 1),
           "step_ms_overhead_pct": round(100*(tr["step_ms"]/ch["step_ms"]-1), 1)}
    print("TCSAME:", json.dumps(res), flush=True)
    json.dump(res, open(f"/results/tcsame_b{budget}.json", "w")); results_vol.commit()
    return json.dumps(res)


@app.local_entrypoint()
def tc(n: int = 128, budget: int = 32, topk: int = 4):
    import json
    r = json.loads(tcsame.remote(budget=budget, topk=topk, n=n))
    c, t = r["chain"], r["tree"]
    print(f"CHAIN: tok/s={c['tok_s']} accept={c['accept_len']} step_ms={c['step_ms']}")
    print(f"TREE : tok/s={t['tok_s']} accept={t['accept_len']} step_ms={t['step_ms']}")
    print(f"tok/s gain {r['tok_s_gain_pct']:+}%  |  accept gain {r['accept_gain_pct']:+}%  |  step_ms overhead {r['step_ms_overhead_pct']:+}%")


@app.function(gpu="B200", timeout=10800, image=base_image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results_vol})
def reusebuf_same(target="Qwen/Qwen3-8B", draft="z-lab/Qwen3-8B-DFlash-b16",
                  block_size: int = 16, topk: int = 4, budget: int = 32, n: int = 128):
    """SAME-CARD ours-b{budget} tree: SGLANG_DFLASH_REUSE_TREE_BUF OFF then ON,
    back-to-back in one container. triton verify, overlap on, conc=1. Reports
    per-step wall ms + tok/s (reuse on vs off) and LOSSLESS via out_hash."""
    import subprocess, time, signal, json, urllib.request, hashlib
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(target)
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    fmt = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    ps = [tk.apply_chat_template([{"role": "user", "content": fmt.format(q=ds[i]["question"])}],
          tokenize=False, add_generation_prompt=True, enable_thinking=False) for i in range(n)]
    def run(reuse):
        env = dict(os.environ)
        env["SGLANG_DFLASH_TREE_VERIFY"] = "1"; env["SGLANG_DFLASH_TREE_ALGO"] = "fused"
        env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"
        env["DFLASH_DRAFT_ATTN_BACKEND"] = "triton"
        env["SGLANG_DFLASH_REUSE_TREE_BUF"] = "1" if reuse else "0"
        server = ["python", "-m", "sglang.launch_server", "--model-path", target,
                  "--trust-remote-code", "--mem-fraction-static", "0.85",
                  "--max-running-requests", "1", "--attention-backend", "triton",
                  "--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", draft,
                  "--speculative-dflash-block-size", str(block_size),
                  "--speculative-eagle-topk", str(topk), "--speculative-num-draft-tokens", str(budget),
                  "--port", "30000"]
        srv = subprocess.Popen(server, env=env)
        for _ in range(240):
            try: urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=2); break
            except Exception: time.sleep(2)
        def send(p):
            req = urllib.request.Request("http://127.0.0.1:30000/generate",
                data=json.dumps({"text": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": 512}}).encode(),
                headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=600).read())
        send(ps[0]); send(ps[1])
        t0 = time.perf_counter(); toks = 0; vct = 0; accs = []; texts = []
        for p in ps:
            r = send(p); m = r["meta_info"]; toks += m["completion_tokens"]; texts.append(r["text"])
            v = m.get("spec_verify_ct")
            if v: vct += v; accs.append(m["completion_tokens"]/v)
        dt = time.perf_counter() - t0
        srv.send_signal(signal.SIGINT); time.sleep(5)
        try: srv.wait(timeout=15)
        except Exception: srv.kill()
        time.sleep(3)
        return {"reuse": reuse, "tok_s": round(toks/dt, 1),
                "accept_len": round(sum(accs)/len(accs), 4) if accs else None,
                "step_ms": round(dt*1000/vct, 4) if vct else None, "verify_ct": vct, "tokens": toks,
                "out_hash": hashlib.sha256("\x1e".join(texts).encode()).hexdigest()[:16]}
    off = run(False); on = run(True)
    lossless = (off["out_hash"] == on["out_hash"] and off["accept_len"] == on["accept_len"]
                and off["tokens"] == on["tokens"])
    res = {"off": off, "on": on, "lossless": lossless,
           "tok_s_gain_pct": round(100*(on["tok_s"]/off["tok_s"]-1), 1) if off["tok_s"] else None,
           "step_ms_delta_pct": round(100*(on["step_ms"]/off["step_ms"]-1), 1) if off["step_ms"] else None}
    print("REUSEBUF:", json.dumps(res), flush=True)
    json.dump(res, open(f"/results/reusebuf_b{budget}.json", "w")); results_vol.commit()
    return json.dumps(res)


@app.local_entrypoint()
def reusebuf(n: int = 128, budget: int = 32, topk: int = 4):
    import json
    r = json.loads(reusebuf_same.remote(budget=budget, topk=topk, n=n))
    off, on = r["off"], r["on"]
    print(f"OFF: tok/s={off['tok_s']} accept={off['accept_len']} step_ms={off['step_ms']} hash={off['out_hash']}")
    print(f"ON : tok/s={on['tok_s']} accept={on['accept_len']} step_ms={on['step_ms']} hash={on['out_hash']}")
    print(f"LOSSLESS={r['lossless']}  |  tok/s gain {r['tok_s_gain_pct']:+}%  |  step_ms delta {r['step_ms_delta_pct']:+}%")


@app.local_entrypoint()
def cgsame(n: int = 256, budget: int = 32, topk: int = 4, overlap: bool = True):
    """SAME-CARD cuda-graph off vs on (one container). Also pass --overlap False to
    test whether the cuda-graph helps when overlap-plan-stream is NOT hiding launches."""
    import json
    r = json.loads(cgbench_same.remote(topk=topk, budget=budget, n=n, overlap=overlap))
    print(f"overlap={r['overlap']}  LOSSLESS={r['lossless']}")
    print(f"  OFF tok/s={r['off']['tok_s']} acc={r['off']['accept_len']}")
    print(f"  ON  tok/s={r['on']['tok_s']} acc={r['on']['accept_len']}")
    print(f"  cuda-graph delta = {r['delta_pct']:+}%")


@app.local_entrypoint()
def profdomino():
    """Profile Domino chain vs z-lab DFlash chain (both triton, conc=1) to break the
    per-step into backbone / lm_head / GRU-rollout. Look at top_gpu_kernels +
    top_cpu_ops (domino_* record_function spans) in the printed/persisted summary."""
    DOM = "Huang2020/Qwen3-8B-Domino-b16"
    ZLAB = "z-lab/Qwen3-8B-DFlash-b16"
    h_dflash = profile_decode.spawn(False, 1, 16, "pd_dflash_chain", draft=ZLAB, backend="triton")
    h_domino = profile_decode.spawn(False, 1, 16, "pd_domino_chain", draft=DOM, backend="triton")
    for name, h in (("dflash", h_dflash), ("domino", h_domino)):
        try:
            import json
            s = json.loads(h.get())
            print(f"\n===== {name} chain (triton) =====")
            print(f"wall_us={s.get('wall_us')} gpu_busy_us={s.get('gpu_busy_us')} util={s.get('gpu_util_pct')}%")
            print("-- top_gpu_kernels --")
            for k in s.get("top_gpu_kernels", [])[:12]:
                print(f"   {k['us']:>9.1f}us  {k['name']}")
            print("-- top_cpu_ops (incl domino_* spans) --")
            for k in s.get("top_cpu_ops", [])[:14]:
                print(f"   {k['us']:>9.1f}us  {k['name']}")
        except Exception as e:
            print(f"{name} FAILED: {e}")


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


# =====================================================================================
# TREE vs CHAIN per-step GPU bucketing (why +22% accept -> only +7% throughput).
# Runs BOTH configs back-to-back in ONE container (same physical B200), exact finding
# config: triton verify, fused tree, overlap-plan-stream ON. Profiles each with the
# server torch profiler (num_steps auto-stop), then buckets per-step GPU-busy us into
# draft-forward / tree-build / target-verify / sampling+overhead by span-window +
# kernel-name, and reports per-step GPU-busy vs wall.
# =====================================================================================

# kernel-name -> bucket for kernels OUTSIDE the DRAFT_LOOP / TARGET_VERIFY spans.
# (Inside-span kernels are attributed to draft / verify by the span they land in.)
_TREE_BUILD_KERNELS = (
    "_dflash_expand_topk4_kernel", "_dflash_tree_verify_steps_topk4_kernel",
    "build_tree_kernel", "expand", "cumsum", "sort", "topk", "gather",
    "argsort", "index_", "arange", "_scatter", "cat_", "nonzero", "unique",
    "assign_draft_cache_locs", "generate_draft_decode_kv_indices",
    "create_flashinfer_kv_indices", "assign_req_to_token",
)
_ACCEPT_KERNELS = (
    "_dflash_tree_accept_compact_kernel", "_dflash_accept_bonus", "fill_bonus_tokens",
    "fill_accept_out_cache_loc", "argmax", "sample", "Memcpy", "Memset", "copy_",
    "fused_kv_materialize", "_fused_norm_rope",
)


def _bucket_of(name, in_draft, in_verify):
    if in_draft:
        return "draft_fwd"
    if in_verify:
        return "target_verify"
    if any(t in name for t in _TREE_BUILD_KERNELS):
        return "tree_build"
    return "accept_overhead"  # loose accept/sample/memcpy + unclassified between-phase


@app.function(gpu="B200", timeout=10800, image=base_image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results_vol})
def profile_tree_vs_chain(budget: int = 32, topk: int = 4, num_steps: int = 40,
                          target="Qwen/Qwen3-8B", draft="z-lab/Qwen3-8B-DFlash-b16",
                          block_size: int = 16):
    """Same-card: launch tree then chain (each its own server in THIS container), profile
    ~num_steps decode steps of each, bucket per-step GPU-busy. Exact finding config:
    triton verify, fused tree, overlap-plan-stream ON, max-running-requests 1."""
    import subprocess, time, signal, json, glob, gzip, os, urllib.request
    from collections import defaultdict
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(target)
    ds = None
    for _ in range(6):
        try:
            ds = load_dataset("openai/gsm8k", "main", split="test").select(range(12)); break
        except Exception as e:
            print("gsm8k retry:", e); time.sleep(10)
    fmt = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    def prompt(i):
        return tok.apply_chat_template(
            [{"role": "user", "content": fmt.format(q=ds[i]["question"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def launch(tree_verify, tag):
        profdir = f"/results/prof2_{tag}"
        os.makedirs(profdir, exist_ok=True)
        # clean stale traces so we analyze THIS run
        for old in glob.glob(f"{profdir}/*"):
            try: os.remove(old)
            except Exception: pass
        env = dict(os.environ)
        env["SGLANG_TORCH_PROFILER_DIR"] = profdir
        env["SGLANG_DFLASH_TREE_VERIFY"] = "1" if tree_verify else "0"
        env["SGLANG_DFLASH_BLOCK_VERIFY"] = "0" if tree_verify else "1"
        env["SGLANG_DFLASH_TREE_ALGO"] = "fused"
        env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"
        env["DFLASH_DRAFT_ATTN_BACKEND"] = "triton"
        spec = ["--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", draft,
                "--speculative-dflash-block-size", str(block_size),
                "--speculative-eagle-topk", str(topk if tree_verify else 1)]
        if tree_verify:
            spec += ["--speculative-num-draft-tokens", str(budget)]
        server = ["python", "-m", "sglang.launch_server", "--model-path", target,
                  "--trust-remote-code", "--mem-fraction-static", "0.85",
                  "--max-running-requests", "1", "--attention-backend", "triton",
                  *spec, "--port", "30000"]
        print(">>>", " ".join(server), "TREE=", env["SGLANG_DFLASH_TREE_VERIFY"], flush=True)
        srv = subprocess.Popen(server, env=env)
        for _ in range(240):
            try:
                urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=2)
                return srv, profdir
            except Exception:
                time.sleep(2)
        srv.send_signal(signal.SIGINT)
        return None, profdir

    def send(p, mx=256):
        req = urllib.request.Request("http://127.0.0.1:30000/generate",
            data=json.dumps({"text": p, "sampling_params":
                {"temperature": 0.0, "max_new_tokens": mx}}).encode(),
            headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=600).read())

    def profile_one(tree_verify, tag):
        srv, profdir = launch(tree_verify, tag)
        if srv is None:
            return {"tag": tag, "error": "server failed to start"}
        try:
            send(prompt(0), 64); send(prompt(1), 64)  # warmup
            # start_profile with num_steps -> auto-stops after num_steps decode steps
            body = json.dumps({"num_steps": num_steps, "activities": ["CPU", "GPU"]}).encode()
            urllib.request.urlopen(urllib.request.Request(
                "http://127.0.0.1:30000/start_profile", data=body,
                headers={"Content-Type": "application/json"}), timeout=30)
            # one long generate produces the num_steps decode steps
            r = send(prompt(2), 320)
            acc = r["meta_info"].get("spec_verify_ct")
            comp = r["meta_info"].get("completion_tokens")
            time.sleep(10)  # let trace flush
        finally:
            srv.send_signal(signal.SIGINT); time.sleep(4)
            try: srv.wait(timeout=10)
            except Exception: srv.kill()
            time.sleep(3)
        summary = _bucketize_trace(profdir, tag, tree_verify)
        summary["accept_len"] = round(comp / acc, 3) if acc else None
        summary["verify_ct"] = acc
        return summary

    out = {}
    for tv, tag in ((True, f"tree_b{budget}_k{topk}"), (False, "chain")):
        out[tag] = profile_one(tv, tag)
        print(f">>> DONE {tag}: {json.dumps({k: out[tag].get(k) for k in ('per_step_gpu_us','per_step_wall_us','buckets_us','accept_len')})}", flush=True)
    json.dump(out, open("/results/tree_vs_chain_buckets.json", "w"), indent=2)
    results_vol.commit()
    return json.dumps(out, indent=2)


def _bucketize_trace(profdir, tag, tree_verify):
    """Merge GPU kernel intervals, count decode steps from run_batch spans, attribute
    each GPU interval to a bucket by the span window it lands in (DRAFT_LOOP/TARGET_VERIFY)
    else by kernel name. Returns per-step GPU-busy us per bucket + wall."""
    import glob, gzip, json
    from collections import defaultdict
    files = sorted(set(glob.glob(f"{profdir}/*.trace.json*") + glob.glob(f"{profdir}/*.json*")))
    if not files:
        return {"tag": tag, "error": "no trace", "profdir": profdir}
    f = files[-1]
    raw = gzip.open(f).read() if f.endswith(".gz") else open(f, "rb").read()
    ev = json.loads(raw)["traceEvents"]

    # span windows (user_annotation), per (pid,tid) doesn't matter — use global ts window
    draft_spans, verify_spans = [], []
    step_spans = []
    kernels = []  # (ts, ts+dur, name)  on GPU streams
    kname_tot = defaultdict(float)
    for e in ev:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat = e.get("cat", ""); name = e.get("name", ""); ts = e.get("ts", 0); dur = e["dur"]
        if cat == "kernel" or cat in ("gpu_memcpy", "gpu_memset"):
            kernels.append((ts, ts + dur, name)); kname_tot[name] += dur
        elif cat == "user_annotation":
            if name.startswith("step[DRAFT_LOOP"):
                draft_spans.append((ts, ts + dur))
            elif name.startswith("step[TARGET_VERIFY") or "TARGET_VERIFY" in name:
                verify_spans.append((ts, ts + dur))
            elif name == "scheduler.run_batch":
                step_spans.append((ts, ts + dur))
    if not kernels:
        return {"tag": tag, "error": "no gpu kernels", "profdir": profdir}

    n_steps = len(step_spans) or len(verify_spans) or 1
    draft_spans.sort(); verify_spans.sort()

    def _in(spans, ts):  # midpoint-in-span check (spans are sorted, disjoint enough)
        lo, hi = 0, len(spans)
        while lo < hi:
            mid = (lo + hi) // 2
            if spans[mid][1] < ts: lo = mid + 1
            else: hi = mid
        return lo < len(spans) and spans[lo][0] <= ts <= spans[lo][1]

    # merge all GPU intervals for true GPU-busy (union, de-overlap across streams)
    ivs = sorted((a, b) for a, b, _ in kernels)
    gpu_busy = 0.0; cs = ce = None; t0 = ivs[0][0]; t1 = 0
    for a, b in ivs:
        t1 = max(t1, b)
        if cs is None: cs, ce = a, b
        elif a <= ce: ce = max(ce, b)
        else: gpu_busy += ce - cs; cs, ce = a, b
    if cs is not None: gpu_busy += ce - cs
    wall = t1 - t0

    # bucket by span-window (mid) then name; use raw dur sum (double-counts cross-stream
    # overlap, but tree-build/accept are tiny single-stream kernels so this is a fair
    # per-bucket split — we renormalize buckets to the merged gpu_busy).
    buckets_raw = defaultdict(float)
    for a, b, name in kernels:
        mid = (a + b) / 2
        bkt = _bucket_of(name, _in(draft_spans, mid), _in(verify_spans, mid))
        buckets_raw[bkt] += (b - a)
    raw_tot = sum(buckets_raw.values()) or 1.0
    # scale raw bucket shares onto the de-overlapped gpu_busy
    buckets_us = {k: round(v / raw_tot * gpu_busy / n_steps, 1) for k, v in buckets_raw.items()}

    top_k = sorted(({"name": k[:60], "us_step": round(v / n_steps, 1)}
                    for k, v in kname_tot.items()), key=lambda x: -x["us_step"])[:16]
    return {
        "tag": tag, "tree": tree_verify, "steps": n_steps,
        "trace_file": f,
        "per_step_gpu_us": round(gpu_busy / n_steps, 1),
        "per_step_wall_us": round(wall / n_steps, 1),
        "per_step_idle_us": round((wall - gpu_busy) / n_steps, 1),
        "gpu_util_pct": round(100 * gpu_busy / wall, 1) if wall else None,
        "n_draft_spans": len(draft_spans), "n_verify_spans": len(verify_spans),
        "buckets_us": buckets_us,
        "top_kernels_us_step": top_k,
    }


@app.function(image=base_image, volumes={"/results": results_vol})
def rebucket(tags: list, tree_flags: list):
    """Re-run the bucketizer on already-saved prof2_<tag> traces (no GPU)."""
    return {tag: _bucketize_trace(f"/results/prof2_{tag}", tag, bool(tv))
            for tag, tv in zip(tags, tree_flags)}


def _print_bucket_report(out):
    order = ["draft_fwd", "tree_build", "target_verify", "accept_overhead"]
    tree = next((v for k, v in out.items() if v.get("tree")), None)
    chain = next((v for k, v in out.items() if v.get("tree") is False), None)
    print("\n================ PER-STEP GPU BUCKET TABLE (us/step) ================")
    print(f"{'bucket':>16} {'chain':>10} {'tree':>10} {'tree-chain':>12}")
    for b in order:
        c = (chain or {}).get("buckets_us", {}).get(b, 0.0)
        t = (tree or {}).get("buckets_us", {}).get(b, 0.0)
        print(f"{b:>16} {c:>10.1f} {t:>10.1f} {t - c:>+12.1f}")
    for lbl, d in (("chain", chain), ("tree", tree)):
        if not d: continue
        print(f"\n[{lbl}] steps={d.get('steps')} accept={d.get('accept_len')} "
              f"per-step GPU={d.get('per_step_gpu_us')}us wall={d.get('per_step_wall_us')}us "
              f"idle={d.get('per_step_idle_us')}us util={d.get('gpu_util_pct')}% "
              f"draft_spans={d.get('n_draft_spans')} verify_spans={d.get('n_verify_spans')}")
        print(f"      trace: {d.get('trace_file')}")
        for k in d.get("top_kernels_us_step", [])[:12]:
            print(f"      {k['us_step']:>8.1f}us/step  {k['name']}")


@app.local_entrypoint()
def treeprofile(budget: int = 32, topk: int = 4, num_steps: int = 40):
    """Same-card tree-vs-chain per-step GPU bucketing (the WHY-only-+7% deliverable)."""
    import json
    out = json.loads(profile_tree_vs_chain.remote(budget, topk, num_steps))
    _print_bucket_report(out)


@app.local_entrypoint()
def treerebucket(tags: str = "tree_b32_k4,chain", tree_flags: str = "1,0"):
    """Re-analyze saved prof2 traces without a GPU run."""
    out = rebucket.remote(tags.split(","), [int(x) for x in tree_flags.split(",")])
    _print_bucket_report(out)
