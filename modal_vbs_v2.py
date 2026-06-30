"""Modal B200 harness for DFlash dynamic-VBS on sglang SPEC-V2 (overlap worker).

Base: upstream sglang with official Spec-V2 DFlash (DFlashWorkerV2) + the dynamic-VBS
truncation feature ported on top. dynamic-VBS truncates each request's draft tokens
before verify (by estimated accept length) so verify wastes less compute. Goal: dyn >=
no-dyn even at low concurrency.

dynamic arg semantics:
  None  -> pass no vbs flag (pure upstream baseline; use before the flag exists)
  True  -> --speculative-dflash-dynamic-vbs
  False -> --no-speculative-dflash-dynamic-vbs

Dev loop:
  modal run modal_vbs_v2.py::one --dynamic-mode none --concurrency 1   # upstream baseline
  modal run modal_vbs_v2.py::one --dynamic-mode on   --concurrency 1
  modal run modal_vbs_v2.py::sweep --concurrencies 1,8,32,64,128
"""
import json
import os
import modal

app = modal.App("dflash-vbs-v2")
WT = "/Users/subsir/Desktop/Studio/Python/sglang-dynamic-verify-v2"

MODEL = "Qwen/Qwen3-8B"
DRAFT = "z-lab/Qwen3-8B-DFlash-b16"
NUM_DRAFT_TOKENS = 16

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "wget", "libnuma-dev")
    .run_commands("pip install --upgrade pip setuptools wheel")
    .add_local_dir(f"{WT}/python", remote_path="/root/sglang/python", copy=True)
    .run_commands(
        # The optional gRPC Rust extension (rust/ dir not copied; unused here) would
        # force a Rust build. Strip its setuptools-rust ext-module so the editable
        # install is pure-Python.
        "python -c \"import re,pathlib; p=pathlib.Path('/root/sglang/python/pyproject.toml'); "
        "s=p.read_text(); "
        "s=re.sub(r'\\[\\[tool\\.setuptools-rust\\.ext-modules\\]\\].*?(?=\\n\\[)', '', s, flags=re.S); "
        "p.write_text(s)\"",
        "cd /root/sglang && pip install -e python",
        "pip install datasets requests rich tqdm transformers numpy safetensors loguru",
    )
)


def _write_mt_bench_jsonl(path: str) -> int:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    with open(path, "w") as f:
        for row in ds:
            turns = row.get("prompt") or row.get("turns")
            if isinstance(turns, str):
                turns = [turns]
            f.write(json.dumps({"turns": list(turns)}) + "\n")
    return len(ds)


@app.function(gpu="B200", cloud="aws", timeout=3600, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def bench(dynamic, concurrency: int, num_prompts: int = 512,
          max_new_tokens: int = 1024, mem_fraction: float = 0.75,
          vbs_margin: float = 2.0, vbs_stat: str = "mean",
          vbs_min_bs: int = 48, vbs_probe: int = 0):
    """dynamic: None|True|False -> see module docstring."""
    import subprocess, signal, time, statistics
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from transformers import AutoTokenizer
    import requests

    # Scale measured prompts with concurrency so low-conc configs (e.g. conc=1,
    # which runs prompts ~sequentially) don't dominate sweep wall-clock. Still
    # enough requests for a stable tok/s.
    num_prompts = min(num_prompts, max(16, concurrency * 6))

    _write_mt_bench_jsonl("/root/mt-bench.jsonl")
    with open("/root/mt-bench.jsonl") as f:
        dataset = [json.loads(l) for l in f]

    cmd = [
        "python", "-m", "sglang.launch_server", "--model-path", MODEL,
        "--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", DRAFT,
        "--speculative-num-draft-tokens", str(NUM_DRAFT_TOKENS),
        "--tp-size", "1", "--attention-backend", "flashinfer", "--page-size", "1",
        "--mem-fraction-static", str(mem_fraction),
        "--max-running-requests", str(max(concurrency, 1)),
        "--port", "30000", "--trust-remote-code",
    ]
    # overlap (spec-v2) is the default; do NOT pass --disable-overlap-schedule.
    if dynamic is True:
        cmd.append("--speculative-dflash-dynamic-vbs")
    elif dynamic is False:
        cmd.append("--no-speculative-dflash-dynamic-vbs")
    env = dict(os.environ); env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    env["SGLANG_DFLASH_VBS_MARGIN"] = str(vbs_margin)
    env["SGLANG_DFLASH_VBS_STAT"] = str(vbs_stat)
    env["SGLANG_DFLASH_VBS_MIN_BS"] = str(vbs_min_bs)
    env["SGLANG_DFLASH_VBS_PROBE"] = str(vbs_probe)
    print(">>> SERVER:", " ".join(cmd),
          f"(margin={vbs_margin} stat={vbs_stat} min_bs={vbs_min_bs} probe={vbs_probe})", flush=True)
    srv = subprocess.Popen(cmd, cwd="/root/sglang", env=env)

    base = "http://127.0.0.1:30000"
    ready = False
    for _ in range(300):
        if srv.poll() is not None:
            break
        try:
            # During startup (model load + cuda graph capture) the server answers
            # /health with 503, which is NOT an exception -> always sleep between
            # checks so we don't burn all iterations before it's warm.
            if requests.get(base + "/health", timeout=5).status_code == 200:
                ready = True; break
        except Exception:
            pass
        time.sleep(3)
    if not ready:
        srv.send_signal(signal.SIGINT)
        return {"dynamic": dynamic, "concurrency": concurrency, "error": "server not ready"}

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = []
    for i in range(num_prompts + concurrency):
        item = dataset[i % len(dataset)]
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": item["turns"][0]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False))

    def send(p):
        r = requests.post(base + "/generate", json={"text": p, "sampling_params":
            {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": max_new_tokens}}, timeout=3600)
        r.raise_for_status(); o = r.json(); return o if isinstance(o, dict) else o[0]

    try:
        requests.get(base + "/flush_cache", timeout=60)
    except Exception:
        pass
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(send, prompts[:concurrency]))
    prompts = prompts[concurrency:]

    t0 = time.perf_counter(); total = 0; accs = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(send, p) for p in prompts]
        for fut in as_completed(futs):
            m = (fut.result().get("meta_info") or {})
            total += int(m.get("completion_tokens", 0))
            if "spec_accept_length" in m:
                try: accs.append(float(m["spec_accept_length"]))
                except Exception: pass
    dt = time.perf_counter() - t0
    srv.send_signal(signal.SIGINT)
    res = {"dynamic": dynamic, "concurrency": concurrency,
           "tok_s": round(total / max(dt, 1e-6), 2),
           "accept_len": round(statistics.mean(accs), 3) if accs else None,
           "secs": round(dt, 1)}
    print(f"\n>>> dynamic={dynamic} conc={concurrency}  Throughput: {res['tok_s']} tok/s  "
          f"Accept length: {res['accept_len']}", flush=True)
    return res


@app.function(gpu="B200", cloud="aws", timeout=7200, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def final_bench(concurrency: int, num_prompts: int = 1024,
                max_new_tokens: int = 1024, mem_fraction: float = 0.75,
                vbs_margin: float = 1.0, vbs_stat: str = "mean", vbs_min_bs: int = 48):
    """Final fair comparison: run no-dynamic then dynamic SERIALLY on the SAME card,
    each over `num_prompts` samples (dataset looped). Returns both + ratio."""
    import subprocess, signal, time, statistics
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from transformers import AutoTokenizer
    import requests

    _write_mt_bench_jsonl("/root/mt-bench.jsonl")
    with open("/root/mt-bench.jsonl") as f:
        dataset = [json.loads(l) for l in f]
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = []
    for i in range(num_prompts + concurrency):
        item = dataset[i % len(dataset)]
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": item["turns"][0]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False))

    def run(dynamic):
        cmd = [
            "python", "-m", "sglang.launch_server", "--model-path", MODEL,
            "--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", DRAFT,
            "--speculative-num-draft-tokens", str(NUM_DRAFT_TOKENS),
            "--tp-size", "1", "--attention-backend", "flashinfer", "--page-size", "1",
            "--mem-fraction-static", str(mem_fraction),
            "--max-running-requests", str(max(concurrency, 1)),
            "--port", "30000", "--trust-remote-code",
        ]
        cmd.append("--speculative-dflash-dynamic-vbs" if dynamic
                   else "--no-speculative-dflash-dynamic-vbs")
        env = dict(os.environ); env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        env["SGLANG_DFLASH_VBS_MARGIN"] = str(vbs_margin)
        env["SGLANG_DFLASH_VBS_STAT"] = str(vbs_stat)
        env["SGLANG_DFLASH_VBS_MIN_BS"] = str(vbs_min_bs)
        srv = subprocess.Popen(cmd, cwd="/root/sglang", env=env)
        base = "http://127.0.0.1:30000"; ready = False
        for _ in range(300):
            if srv.poll() is not None: break
            try:
                if requests.get(base + "/health", timeout=5).status_code == 200:
                    ready = True; break
            except Exception: pass
            time.sleep(3)
        if not ready:
            srv.send_signal(signal.SIGINT); return {"error": "server not ready"}

        def send(p):
            r = requests.post(base + "/generate", json={"text": p, "sampling_params":
                {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": max_new_tokens}}, timeout=3600)
            r.raise_for_status(); o = r.json(); return o if isinstance(o, dict) else o[0]
        try: requests.get(base + "/flush_cache", timeout=60)
        except Exception: pass
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(send, prompts[:concurrency]))
        t0 = time.perf_counter(); total = 0; accs = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(send, p) for p in prompts[concurrency:]]
            for fut in as_completed(futs):
                m = (fut.result().get("meta_info") or {})
                total += int(m.get("completion_tokens", 0))
                if "spec_accept_length" in m:
                    try: accs.append(float(m["spec_accept_length"]))
                    except Exception: pass
        dt = time.perf_counter() - t0
        srv.send_signal(signal.SIGINT); time.sleep(8)
        return {"tok_s": round(total / max(dt, 1e-6), 2),
                "accept_len": round(statistics.mean(accs), 3) if accs else None,
                "secs": round(dt, 1)}

    off = run(False)
    on = run(True)
    ratio = (on["tok_s"] / off["tok_s"]) if (off.get("tok_s") and on.get("tok_s")) else None
    res = {"concurrency": concurrency, "num_prompts": num_prompts, "off": off, "on": on,
           "ratio": round(ratio, 4) if ratio else None}
    print(f"\n>>> FINAL conc={concurrency} n={num_prompts}: "
          f"no-dyn={off.get('tok_s')} (acc {off.get('accept_len')}) | "
          f"dyn={on.get('tok_s')} (acc {on.get('accept_len')}) | ratio={res['ratio']}", flush=True)
    return res


@app.local_entrypoint()
def final(concurrencies: str = "1,32,64,128", num_prompts: int = 1024,
          low_prompts: int = 256, vbs_margin: float = 1.0, vbs_stat: str = "mean",
          vbs_min_bs: int = 48):
    """Same-card serial no-dyn vs dyn across concurrencies (the deliverable). Low conc
    uses fewer prompts (slow, ~sequential); high conc uses the full num_prompts."""
    concs = [int(c) for c in concurrencies.split(",")]
    handles = []
    for c in concs:
        n = num_prompts if c >= 32 else low_prompts
        handles.append((c, final_bench.spawn(c, n, 1024, 0.75, vbs_margin, vbs_stat, vbs_min_bs)))
    print(f"\n==== DFlash dynamic-VBS FINAL (same card, serial; margin={vbs_margin} "
          f"stat={vbs_stat} min_bs={vbs_min_bs}) ====")
    print(f"{'conc':>6} {'n':>6} | {'no-dyn':>9} {'acc':>6} | {'dyn':>9} {'acc':>6} | ratio")
    for c, h in handles:
        try:
            r = h.get()
            print(f"{c:>6} {r['num_prompts']:>6} | {str(r['off'].get('tok_s')):>9} "
                  f"{str(r['off'].get('accept_len')):>6} | {str(r['on'].get('tok_s')):>9} "
                  f"{str(r['on'].get('accept_len')):>6} | {r['ratio']}")
        except Exception as e:
            print(f"{c:>6}: ERROR {e}")


@app.function(gpu="B200", cloud="aws", timeout=3600, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def profile(dynamic, concurrency: int = 32, capture_secs: float = 4.0,
            mem_fraction: float = 0.75, vbs_margin: float = 0.0,
            vbs_stat: str = "mean", vbs_min_bs: int = 2):
    """Torch-profiler (NOT cuda-event) trace of steady-state decode at `concurrency`.
    Returns GPU-busy vs wall (idle% = CPU-bound bubbles) + top GPU kernels, so we can
    see what fraction is the target verify and whether the dynamic path stalls the GPU.
    """
    import subprocess, signal, time, glob, json as _json, os as _os
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoTokenizer
    import requests, threading

    prof_dir = "/root/prof"
    _os.makedirs(prof_dir, exist_ok=True)
    _write_mt_bench_jsonl("/root/mt-bench.jsonl")
    with open("/root/mt-bench.jsonl") as f:
        dataset = [_json.loads(l) for l in f]

    cmd = [
        "python", "-m", "sglang.launch_server", "--model-path", MODEL,
        "--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", DRAFT,
        "--speculative-num-draft-tokens", str(NUM_DRAFT_TOKENS),
        "--tp-size", "1", "--attention-backend", "flashinfer", "--page-size", "1",
        "--mem-fraction-static", str(mem_fraction),
        "--max-running-requests", str(max(concurrency, 1)),
        "--port", "30000", "--trust-remote-code",
    ]
    if dynamic is True:
        cmd.append("--speculative-dflash-dynamic-vbs")
    elif dynamic is False:
        cmd.append("--no-speculative-dflash-dynamic-vbs")
    env = dict(os.environ)
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    env["SGLANG_TORCH_PROFILER_DIR"] = prof_dir
    env["SGLANG_DFLASH_VBS_MARGIN"] = str(vbs_margin)
    env["SGLANG_DFLASH_VBS_STAT"] = str(vbs_stat)
    env["SGLANG_DFLASH_VBS_MIN_BS"] = str(vbs_min_bs)
    print(">>> SERVER:", " ".join(cmd), flush=True)
    srv = subprocess.Popen(cmd, cwd="/root/sglang", env=env)
    base = "http://127.0.0.1:30000"
    ready = False
    for _ in range(300):
        if srv.poll() is not None:
            break
        try:
            if requests.get(base + "/health", timeout=5).status_code == 200:
                ready = True; break
        except Exception:
            pass
        time.sleep(3)
    if not ready:
        srv.send_signal(signal.SIGINT)
        return {"error": "server not ready"}

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = []
    for i in range(concurrency * 4):
        item = dataset[i % len(dataset)]
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": item["turns"][0]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False))

    def send(p):
        try:
            requests.post(base + "/generate", json={"text": p, "sampling_params":
                {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512}}, timeout=600)
        except Exception:
            pass

    # Maintain a constant `concurrency` in-flight (replenish on completion) so the
    # GPU stays saturated — a batch-and-wait loop sags between batches and inflates
    # measured GPU-idle. Use many workers + a semaphore to cap concurrency.
    stop = threading.Event()
    sem = threading.Semaphore(concurrency)
    def one(i):
        sem.acquire()
        try:
            send(prompts[i % len(prompts)])
        finally:
            sem.release()
    def load_loop():
        with ThreadPoolExecutor(max_workers=concurrency * 2) as pool:
            i = 0
            while not stop.is_set():
                sem.acquire(); sem.release()  # throttle submission to inflight cap
                pool.submit(one, i); i += 1
    th = threading.Thread(target=load_loop, daemon=True); th.start()
    time.sleep(8)  # reach steady-state decode at full concurrency

    requests.post(base + "/start_profile",
                  json={"activities": ["CPU", "GPU"]}, timeout=60)
    time.sleep(capture_secs)
    requests.post(base + "/stop_profile", timeout=120)
    time.sleep(3)
    stop.set(); srv.send_signal(signal.SIGINT); time.sleep(2)

    traces = sorted(glob.glob(prof_dir + "/*.json*") + glob.glob(prof_dir + "/*.pt.trace.json*"),
                    key=lambda p: _os.path.getmtime(p))
    if not traces:
        return {"error": "no trace produced", "dir": _os.listdir(prof_dir)}
    summary = _summarize_trace(traces[-1])
    summary["dynamic"] = dynamic; summary["concurrency"] = concurrency
    print(">>> PROFILE SUMMARY:", _json.dumps(summary, indent=2), flush=True)
    return summary


def _summarize_trace(path: str):
    """GPU busy/idle + top GPU kernels from a torch-profiler chrome trace."""
    import json as _json, gzip
    opn = gzip.open if path.endswith(".gz") else open
    with opn(path, "rt") as f:
        data = _json.load(f)
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data
    kernels = [e for e in evs if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_op")]
    if not kernels:
        # fall back: cat may be "Kernel" or use pid/tid lanes; treat cuda category
        kernels = [e for e in evs if e.get("ph") == "X" and "kernel" in str(e.get("cat", "")).lower()]
    cpu_ops = [e for e in evs if e.get("ph") == "X" and e.get("cat") == "cpu_op"]
    runtime = [e for e in evs if e.get("ph") == "X"
               and str(e.get("cat", "")).lower() in ("cuda_runtime", "runtime")]
    if not kernels:
        return {"error": "no GPU kernels in trace", "n_events": len(evs)}
    gpu_busy = sum(e.get("dur", 0) for e in kernels)
    t0 = min(e["ts"] for e in kernels)
    t1 = max(e["ts"] + e.get("dur", 0) for e in kernels)
    wall = max(t1 - t0, 1)

    def _topn(events, n=16):
        by = {}
        for e in events:
            by[e["name"]] = by.get(e["name"], 0) + e.get("dur", 0)
        return [[round(v, 1), k[:74]] for k, v in
                sorted(by.items(), key=lambda kv: -kv[1])[:n]]

    # CPU self-time per op-name: subtract immediate GPU-launch children isn't tractable
    # from chrome json cheaply, so report total dur (inclusive) per name + launch count.
    launch = [e for e in runtime if "launch" in e["name"].lower()
              or "Launch" in e["name"]]
    return {
        "wall_us": round(wall, 1),
        "gpu_busy_us": round(gpu_busy, 1),
        "gpu_busy_frac": round(gpu_busy / wall, 4),
        "gpu_idle_frac": round(1 - gpu_busy / wall, 4),
        "n_kernels": len(kernels),
        "n_cpu_ops": len(cpu_ops),
        "n_runtime": len(runtime),
        "n_launch": len(launch),
        "launch_us_total": round(sum(e.get("dur", 0) for e in launch), 1),
        "cpu_op_us_total": round(sum(e.get("dur", 0) for e in cpu_ops), 1),
        "top_kernels_us": _topn(kernels),
        "top_cpu_ops_us": _topn(cpu_ops),
        "top_runtime_us": _topn(runtime, 10),
    }


@app.local_entrypoint()
def prof(dynamic_mode: str = "off", concurrency: int = 32, vbs_margin: float = 0.0,
         vbs_stat: str = "mean", vbs_min_bs: int = 2):
    import json as _json
    print(_json.dumps(profile.remote(_mode(dynamic_mode), concurrency, 4.0, 0.75,
                                     vbs_margin, vbs_stat, vbs_min_bs), indent=2))


_trace_vol = modal.Volume.from_name("dflash-traces", create_if_missing=True)


@app.function(gpu="B200", cloud="aws", timeout=3600, image=image,
              volumes={"/traces": _trace_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def trace(concurrency: int = 32, num_steps: int = 40, vbs_margin: float = 0.0,
          vbs_min_bs: int = 1, mem_fraction: float = 0.75):
    """Capture sglang torch-profiler traces for no-dyn AND dyn on the SAME card
    (serial), at `concurrency`. Saves chrome traces to the dflash-traces volume.
    Download: modal volume get dflash-traces /<dir> ./"""
    import subprocess, signal, time, glob, os as _os, threading
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoTokenizer
    import requests

    _write_mt_bench_jsonl("/root/mt-bench.jsonl")
    with open("/root/mt-bench.jsonl") as f:
        dataset = [json.loads(l) for l in f]
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": dataset[i % len(dataset)]["turns"][0]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for i in range(concurrency * 4)]

    out = {}
    for dynamic in (False, True):
        tag = "dyn" if dynamic else "nodyn"
        prof_dir = f"/traces/qwen3-8b_conc{concurrency}_{tag}"
        _os.makedirs(prof_dir, exist_ok=True)
        for _f in glob.glob(prof_dir + "/*"):  # clear stale (possibly-corrupt) traces
            try: _os.remove(_f)
            except Exception: pass
        cmd = [
            "python", "-m", "sglang.launch_server", "--model-path", MODEL,
            "--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", DRAFT,
            "--speculative-num-draft-tokens", str(NUM_DRAFT_TOKENS),
            "--tp-size", "1", "--attention-backend", "flashinfer", "--page-size", "1",
            "--mem-fraction-static", str(mem_fraction),
            "--max-running-requests", str(max(concurrency, 1)),
            "--port", "30000", "--trust-remote-code",
            ("--speculative-dflash-dynamic-vbs" if dynamic
             else "--no-speculative-dflash-dynamic-vbs"),
        ]
        env = dict(os.environ)
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        env["SGLANG_TORCH_PROFILER_DIR"] = prof_dir
        env["SGLANG_DFLASH_VBS_MARGIN"] = str(vbs_margin)
        env["SGLANG_DFLASH_VBS_MIN_BS"] = str(vbs_min_bs)
        print(f">>> [{tag}]", " ".join(cmd), flush=True)
        srv = subprocess.Popen(cmd, cwd="/root/sglang", env=env)
        base = "http://127.0.0.1:30000"; ready = False
        for _ in range(300):
            if srv.poll() is not None: break
            try:
                if requests.get(base + "/health", timeout=5).status_code == 200:
                    ready = True; break
            except Exception: pass
            time.sleep(3)
        if not ready:
            srv.send_signal(signal.SIGINT); out[tag] = "server not ready"; continue

        def send(p):
            try:
                requests.post(base + "/generate", json={"text": p, "sampling_params":
                    {"temperature": 0.0, "max_new_tokens": 512}}, timeout=600)
            except Exception: pass
        stop = threading.Event()
        def load():
            with ThreadPoolExecutor(max_workers=concurrency * 2) as pool:
                i = 0
                while not stop.is_set():
                    if pool._work_queue.qsize() < concurrency:
                        pool.submit(send, prompts[i % len(prompts)]); i += 1
                    else:
                        time.sleep(0.001)
        th = threading.Thread(target=load, daemon=True); th.start()
        time.sleep(8)  # steady state
        before = set(glob.glob(prof_dir + "/*"))
        requests.post(base + "/start_profile",
                      json={"num_steps": num_steps, "activities": ["CPU", "GPU"]}, timeout=60)
        # wait for a NEW trace file, then wait until its size is STABLE (the .gz dump
        # is written incrementally; grabbing it mid-write yields a truncated/corrupt gz).
        path = None
        for _ in range(120):
            time.sleep(2)
            cand = [f for f in set(glob.glob(prof_dir + "/*.json*")) - before
                    if _os.path.getsize(f) > 0]
            if cand:
                path = sorted(cand, key=_os.path.getmtime)[-1]; break
        if path is None:
            try: requests.post(base + "/stop_profile", timeout=60)
            except Exception: pass
            for _ in range(30):
                time.sleep(2)
                cand = sorted(glob.glob(prof_dir + "/*.json*"), key=_os.path.getmtime)
                if cand: path = cand[-1]; break
        if path is not None:
            last = -1
            for _ in range(60):  # wait for size to stop growing
                sz = _os.path.getsize(path)
                if sz == last and sz > 0:
                    break
                last = sz; time.sleep(1)
            time.sleep(2)
        stop.set(); srv.send_signal(signal.SIGINT); time.sleep(5)
        sz = _os.path.getsize(path) / 1e6 if path and _os.path.exists(path) else 0
        out[tag] = {"dir": prof_dir, "file": path, "size_mb": round(sz, 1),
                    "all": [_os.path.basename(x) for x in glob.glob(prof_dir + "/*")]}
        print(f">>> [{tag}] trace: {out[tag]}", flush=True)

    _trace_vol.commit()
    print(">>> DOWNLOAD with: modal volume get dflash-traces / ./traces", flush=True)
    print(">>> TRACE RESULT:", json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def trace_main(concurrency: int = 32, num_steps: int = 40, vbs_margin: float = 0.0,
               vbs_min_bs: int = 1):
    print(json.dumps(trace.remote(concurrency, num_steps, vbs_margin, vbs_min_bs), indent=2))


def _mode(s):
    return {"none": None, "on": True, "off": False}[s]


@app.local_entrypoint()
def one(dynamic_mode: str = "none", concurrency: int = 1, num_prompts: int = 512,
        vbs_margin: float = 1.0, vbs_stat: str = "mean", vbs_min_bs: int = 48,
        vbs_probe: int = 0):
    print(bench.remote(_mode(dynamic_mode), concurrency, num_prompts, 1024, 0.75,
                       vbs_margin, vbs_stat, vbs_min_bs, vbs_probe))


@app.local_entrypoint()
def tune(concurrency: int = 128, num_prompts: int = 512,
         configs: str = "off|2|mean,on|1|mean,on|2|mean,on|3|mean,on|2|0.75,on|2|0.9"):
    """Sweep (dynamic, margin, stat) at one concurrency to find the best throughput/
    accept tradeoff. configs: comma-list of 'mode|margin|stat'."""
    specs = []
    for c in configs.split(","):
        mode, margin, stat = c.split("|")
        specs.append((_mode(mode), float(margin), stat))
    handles = [((m, mg, st), bench.spawn(m, concurrency, num_prompts, 1024, 0.75, mg, st))
               for (m, mg, st) in specs]
    print(f"\n==== DFlash VBS tune @conc={concurrency} ====")
    print(f"{'mode':>5} {'margin':>6} {'stat':>6} | {'tok/s':>9} {'accept':>7}")
    for (m, mg, st), h in handles:
        try:
            r = h.get()
            print(f"{str(m):>5} {mg:>6} {st:>6} | {str(r.get('tok_s')):>9} {str(r.get('accept_len')):>7}")
        except Exception as e:
            print(f"{str(m):>5} {mg:>6} {st:>6} | ERROR {str(e)[:50]}")


@app.local_entrypoint()
def sweep(concurrencies: str = "1,8,32,64,128", num_prompts: int = 512):
    concs = [int(c) for c in concurrencies.split(",")]
    handles = []
    for dyn in (False, True):
        for c in concs:
            handles.append(((dyn, c), bench.spawn(dyn, c, num_prompts)))
    rows = []
    for (dyn, c), h in handles:
        try:
            rows.append(h.get())
        except Exception as e:
            rows.append({"dynamic": dyn, "concurrency": c, "error": str(e)})
    print("\n==== DFlash dynamic-VBS v2 sweep (Qwen3-8B, B200, mt-bench) ====")
    print(f"{'conc':>6} | {'no-dyn tok/s':>12} {'acc':>6} | {'dyn tok/s':>12} {'acc':>6} | dyn/no")
    by = {(r.get('dynamic'), r.get('concurrency')): r for r in rows}
    for c in concs:
        a = by.get((False, c), {}); b = by.get((True, c), {})
        ta, tb = a.get('tok_s'), b.get('tok_s')
        ratio = f"{tb/ta:.2f}x" if (ta and tb) else "?"
        print(f"{c:>6} | {str(ta):>12} {str(a.get('accept_len')):>6} | "
              f"{str(tb):>12} {str(b.get('accept_len')):>6} | {ratio}")
