"""DFlash dynamic-VBS benchmark on Kimi-K2.6 (4-bit MoE, ~595GB) + z-lab/Kimi-K2.6-DFlash
draft (block_size=8), reading weights from the pre-populated `hf-cache` modal Volume
(offline, no download). Multi-GPU (4 or 8 x B200), tp=ngpu.

Same recipe as modal_vbs_v2: same-card serial no-dyn vs dyn, margin=0.
  modal run modal_kimi.py::ksmoke --ngpu 4               # just load + 1 request
  modal run modal_kimi.py::kfinal --concurrencies 1,32,128 --ngpu 4 --vbs-margin 0
"""
import json
import os
import modal

app = modal.App("dflash-kimi")
WT = "/Users/subsir/Desktop/Studio/Python/sglang-dynamic-verify-v2"
MODEL = "moonshotai/Kimi-K2.6"
DRAFT = "z-lab/Kimi-K2.6-DFlash"
NUM_DRAFT_TOKENS = 8  # draft block_size

# Same sglang build as modal_vbs_v2 (Modal reuses the cached layer if identical).
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "wget", "libnuma-dev")
    .run_commands("pip install --upgrade pip setuptools wheel")
    .add_local_dir(f"{WT}/python", remote_path="/root/sglang/python", copy=True)
    .run_commands(
        "python -c \"import re,pathlib; p=pathlib.Path('/root/sglang/python/pyproject.toml'); "
        "s=p.read_text(); "
        "s=re.sub(r'\\[\\[tool\\.setuptools-rust\\.ext-modules\\]\\].*?(?=\\n\\[)', '', s, flags=re.S); "
        "p.write_text(s)\"",
        "cd /root/sglang && pip install -e python",
        "pip install datasets requests rich tqdm transformers numpy safetensors loguru blobfile",
    )
    .env({"HF_HOME": "/cache", "HF_HUB_OFFLINE": "1",
          "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"})
)
vol = modal.Volume.from_name("hf-cache")
SECRET = modal.Secret.from_name("huggingface-secret")


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


def _launch_args(ngpu, dynamic, mem_fraction, max_running, vbs_margin, vbs_min_bs):
    cmd = [
        "python", "-m", "sglang.launch_server", "--model-path", MODEL,
        "--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", DRAFT,
        "--speculative-num-draft-tokens", str(NUM_DRAFT_TOKENS),
        "--tp-size", str(ngpu), "--attention-backend", "flashinfer", "--page-size", "1",
        "--mem-fraction-static", str(mem_fraction),
        "--max-running-requests", str(max(max_running, 1)),
        "--trust-remote-code", "--port", "30000",
    ]
    if dynamic is True:
        cmd.append("--speculative-dflash-dynamic-vbs")
    elif dynamic is False:
        cmd.append("--no-speculative-dflash-dynamic-vbs")
    return cmd


@app.function(gpu="B200:4", cloud="aws", timeout=5400, image=image,
              volumes={"/cache": vol}, secrets=[SECRET])
def ksmoke(ngpu: int = 4, mem_fraction: float = 0.9):
    """Just load Kimi-K2.6 + DFlash draft on ngpu B200 and serve one request."""
    import subprocess, signal, time, requests
    _write_mt_bench_jsonl("/root/mt-bench.jsonl")
    cmd = _launch_args(ngpu, None, mem_fraction, 8, 0.0, 48)
    print(">>> SERVER:", " ".join(cmd), flush=True)
    env = dict(os.environ)
    srv = subprocess.Popen(cmd, cwd="/root/sglang", env=env)
    base = "http://127.0.0.1:30000"; ready = False
    for _ in range(600):  # up to 30 min for a 595GB load + graph capture
        if srv.poll() is not None:
            return {"error": f"server exited rc={srv.returncode}"}
        try:
            if requests.get(base + "/health", timeout=5).status_code == 200:
                ready = True; break
        except Exception:
            pass
        time.sleep(3)
    if not ready:
        srv.send_signal(signal.SIGINT); return {"error": "not ready in 30min"}
    print(">>> server READY, sending a request", flush=True)
    t0 = time.perf_counter()
    r = requests.post(base + "/generate", json={"text": "Explain speculative decoding in two sentences.",
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 128}}, timeout=600)
    out = r.json(); m = out.get("meta_info", {})
    res = {"ok": True, "secs_first_req": round(time.perf_counter() - t0, 1),
           "completion_tokens": m.get("completion_tokens"),
           "accept_len": m.get("spec_accept_length"),
           "text_head": (out.get("text") or "")[:200]}
    srv.send_signal(signal.SIGINT); time.sleep(5)
    print(">>> SMOKE RESULT:", json.dumps(res, indent=2), flush=True)
    return res


@app.local_entrypoint()
def ksmoke_main(ngpu: int = 4, mem_fraction: float = 0.9):
    print(ksmoke.remote(ngpu, mem_fraction))


def _summarize_trace(path):
    import json as _json, gzip
    opn = gzip.open if path.endswith(".gz") else open
    with opn(path, "rt") as f:
        data = _json.load(f)
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data
    kernels = [e for e in evs if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_op")]
    runtime = [e for e in evs if e.get("ph") == "X"
               and str(e.get("cat", "")).lower() in ("cuda_runtime", "runtime")]
    cpu_ops = [e for e in evs if e.get("ph") == "X" and e.get("cat") == "cpu_op"]
    if not kernels:
        return {"error": "no kernels", "n": len(evs)}
    busy = sum(e.get("dur", 0) for e in kernels)
    t0 = min(e["ts"] for e in kernels); t1 = max(e["ts"] + e.get("dur", 0) for e in kernels)
    wall = max(t1 - t0, 1)

    def top(events, n=14):
        by = {}
        for e in events:
            by[e["name"]] = by.get(e["name"], 0) + e.get("dur", 0)
        return [[round(v, 1), k[:80]] for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:n]]
    return {"wall_us": round(wall, 1), "gpu_busy_us": round(busy, 1),
            "gpu_busy_frac": round(busy / wall, 4), "gpu_idle_frac": round(1 - busy / wall, 4),
            "n_kernels": len(kernels), "n_cpu_ops": len(cpu_ops),
            "runtime_us_total": round(sum(e.get("dur", 0) for e in runtime), 1),
            "top_kernels_us": top(kernels), "top_runtime_us": top(runtime, 8)}


@app.function(gpu="B200:4", cloud="aws", timeout=9000, image=image,
              volumes={"/cache": vol, "/traces": modal.Volume.from_name("dflash-traces", create_if_missing=True)},
              secrets=[SECRET])
def kprofile(concurrency: int = 128, num_steps: int = 30, vbs_margin: float = 0.0,
             vbs_min_bs: int = 1, mem_fraction: float = 0.9, ngpu: int = 4):
    """Same 4-card group: profile no-dyn then dyn at `concurrency`. Saves chrome traces
    to dflash-traces and returns GPU-busy/idle + top kernels for both."""
    import subprocess, signal, time, glob, threading
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoTokenizer
    import requests
    tvol = modal.Volume.from_name("dflash-traces")
    _write_mt_bench_jsonl("/root/mt-bench.jsonl")
    with open("/root/mt-bench.jsonl") as f:
        dataset = [json.loads(l) for l in f]
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompts = []
    for i in range(concurrency * 4):
        it = dataset[i % len(dataset)]
        try: p = tok.apply_chat_template([{"role": "user", "content": it["turns"][0]}], tokenize=False, add_generation_prompt=True)
        except Exception: p = it["turns"][0]
        prompts.append(p)

    out = {}
    for dynamic in (False, True):
        tag = "dyn" if dynamic else "nodyn"
        pdir = f"/traces/kimi-k2.6_conc{concurrency}_{tag}"
        os.makedirs(pdir, exist_ok=True)
        cmd = _launch_args(ngpu, dynamic, mem_fraction, max(concurrency, 1), vbs_margin, vbs_min_bs)
        env = dict(os.environ)
        env["SGLANG_TORCH_PROFILER_DIR"] = pdir
        env["SGLANG_DFLASH_VBS_MARGIN"] = str(vbs_margin)
        env["SGLANG_DFLASH_VBS_MIN_BS"] = str(vbs_min_bs)
        print(f">>> [{tag}]", " ".join(cmd), flush=True)
        srv = subprocess.Popen(cmd, cwd="/root/sglang", env=env)
        base = "http://127.0.0.1:30000"; ready = False
        for _ in range(700):
            if srv.poll() is not None: break
            try:
                if requests.get(base + "/health", timeout=5).status_code == 200:
                    ready = True; break
            except Exception: pass
            time.sleep(3)
        if not ready:
            srv.send_signal(signal.SIGINT); out[tag] = {"error": "not ready"}; continue

        def send(p):
            try: requests.post(base + "/generate", json={"text": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": 512}}, timeout=600)
            except Exception: pass
        stop = threading.Event()
        def load():
            with ThreadPoolExecutor(max_workers=concurrency * 2) as pool:
                i = 0
                while not stop.is_set():
                    if pool._work_queue.qsize() < concurrency:
                        pool.submit(send, prompts[i % len(prompts)]); i += 1
                    else: time.sleep(0.001)
        th = threading.Thread(target=load, daemon=True); th.start()
        time.sleep(12)
        before = set(glob.glob(pdir + "/*"))
        requests.post(base + "/start_profile", json={"num_steps": num_steps, "activities": ["CPU", "GPU"]}, timeout=60)
        path = None
        for _ in range(150):
            time.sleep(2)
            done = [f for f in set(glob.glob(pdir + "/*.json*")) - before if os.path.getsize(f) > 0]
            if done: path = sorted(done, key=os.path.getmtime)[-1]; break
        if path is None:
            try: requests.post(base + "/stop_profile", timeout=60)
            except Exception: pass
            for _ in range(30):
                time.sleep(2)
                cand = sorted(glob.glob(pdir + "/*.json*"), key=os.path.getmtime)
                if cand: path = cand[-1]; break
        # CRITICAL: wait until ALL per-rank trace files finish writing (size stable)
        # before parsing or killing the server, else the .gz is truncated/corrupt.
        if path is not None:
            stable = 0; last_total = -1
            for _ in range(120):
                files = glob.glob(pdir + "/*.json*")
                total = sum(os.path.getsize(f) for f in files)
                if total == last_total and total > 0:
                    stable += 1
                    if stable >= 3: break
                else:
                    stable = 0; last_total = total
                time.sleep(1)
            time.sleep(3)
        stop.set(); srv.send_signal(signal.SIGINT); time.sleep(10)
        try:
            summ = _summarize_trace(path) if path else {"error": "no trace"}
        except Exception as e:
            summ = {"error": f"parse failed: {e}", "trace": path}
        summ["trace"] = path
        out[tag] = summ
        print(f">>> [{tag}] {json.dumps(summ)[:600]}", flush=True)
    tvol.commit()
    print(">>> KPROFILE RESULT:", json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def kprofile_main(concurrency: int = 128, num_steps: int = 30, vbs_margin: float = 0.0,
                  vbs_min_bs: int = 1):
    print(json.dumps(kprofile.remote(concurrency, num_steps, vbs_margin, vbs_min_bs), indent=2))


@app.function(gpu="B200:4", cloud="aws", timeout=9000, image=image,
              volumes={"/cache": vol}, secrets=[SECRET])
def kfinal_bench(concurrency: int, num_prompts: int = 512, max_new_tokens: int = 1024,
                 mem_fraction: float = 0.9, vbs_margin: float = 0.0, vbs_stat: str = "mean",
                 vbs_min_bs: int = 1, ngpu: int = 4):
    """Same 4-card group: no-dyn then dyn serially, over num_prompts samples."""
    import subprocess, signal, time, statistics
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from transformers import AutoTokenizer
    import requests

    _write_mt_bench_jsonl("/root/mt-bench.jsonl")
    with open("/root/mt-bench.jsonl") as f:
        dataset = [json.loads(l) for l in f]
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompts = []
    for i in range(num_prompts + concurrency):
        item = dataset[i % len(dataset)]
        try:
            p = tok.apply_chat_template([{"role": "user", "content": item["turns"][0]}],
                                        tokenize=False, add_generation_prompt=True)
        except Exception:
            p = item["turns"][0]
        prompts.append(p)

    def run(dynamic):
        cmd = _launch_args(ngpu, dynamic, mem_fraction, max(concurrency, 1),
                           vbs_margin, vbs_min_bs)
        env = dict(os.environ)
        env["SGLANG_DFLASH_VBS_MARGIN"] = str(vbs_margin)
        env["SGLANG_DFLASH_VBS_STAT"] = str(vbs_stat)
        env["SGLANG_DFLASH_VBS_MIN_BS"] = str(vbs_min_bs)
        print(">>> SERVER:", " ".join(cmd), f"(margin={vbs_margin} min_bs={vbs_min_bs})", flush=True)
        srv = subprocess.Popen(cmd, cwd="/root/sglang", env=env)
        base = "http://127.0.0.1:30000"; ready = False
        for _ in range(700):
            if srv.poll() is not None:
                return {"error": f"server exited rc={srv.returncode}"}
            try:
                if requests.get(base + "/health", timeout=5).status_code == 200:
                    ready = True; break
            except Exception:
                pass
            time.sleep(3)
        if not ready:
            srv.send_signal(signal.SIGINT); return {"error": "not ready"}

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
        srv.send_signal(signal.SIGINT); time.sleep(10)
        return {"tok_s": round(total / max(dt, 1e-6), 2),
                "accept_len": round(statistics.mean(accs), 3) if accs else None, "secs": round(dt, 1)}

    off = run(False)
    on = run(True)
    ratio = (on["tok_s"] / off["tok_s"]) if (off.get("tok_s") and on.get("tok_s")) else None
    res = {"concurrency": concurrency, "num_prompts": num_prompts, "off": off, "on": on,
           "ratio": round(ratio, 4) if ratio else None}
    print(f"\n>>> KIMI FINAL conc={concurrency} n={num_prompts}: no-dyn={off.get('tok_s')} "
          f"(acc {off.get('accept_len')}) | dyn={on.get('tok_s')} (acc {on.get('accept_len')}) "
          f"| ratio={res['ratio']}", flush=True)
    return res


@app.local_entrypoint()
def kfinal(concurrencies: str = "1,32,128", num_prompts: int = 512, low_prompts: int = 128,
           vbs_margin: float = 0.0, vbs_min_bs: int = 1, ngpu: int = 4):
    concs = [int(c) for c in concurrencies.split(",")]
    handles = []
    for c in concs:
        n = num_prompts if c >= 32 else low_prompts
        handles.append((c, kfinal_bench.spawn(c, n, 1024, 0.9, vbs_margin, "mean", vbs_min_bs, ngpu)))
    print(f"\n==== KIMI-K2.6 dynamic-VBS FINAL (same {ngpu}xB200, serial; margin={vbs_margin} "
          f"min_bs={vbs_min_bs}) ====")
    print(f"{'conc':>6} {'n':>6} | {'no-dyn':>9} {'acc':>6} | {'dyn':>9} {'acc':>6} | ratio")
    for c, h in handles:
        try:
            r = h.get()
            print(f"{c:>6} {r['num_prompts']:>6} | {str(r['off'].get('tok_s')):>9} "
                  f"{str(r['off'].get('accept_len')):>6} | {str(r['on'].get('tok_s')):>9} "
                  f"{str(r['on'].get('accept_len')):>6} | {r['ratio']}")
        except Exception as e:
            print(f"{c:>6}: ERROR {e}")
