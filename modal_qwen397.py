"""DFlash dynamic-VBS benchmark on Qwen/Qwen3.5-397B-A17B (bf16 MoE, ~794GB, block_size 16)
+ z-lab/Qwen3.5-397B-A17B-DFlash draft, on 8xB200 (tp=8). Weights cached in the hf-cache
modal Volume (download once via `dl`, then run offline).

  modal run modal_qwen397.py::dl                 # download ~794GB into the volume
  modal run modal_qwen397.py::qsmoke_main        # load + 1 request on 8xB200
  modal run modal_qwen397.py::qfinal --concurrencies 1,32,128 --vbs-margin 0
"""
import json
import os
import modal

app = modal.App("dflash-qwen397")
WT = "/Users/subsir/Desktop/Studio/Python/sglang-dynamic-verify-v2"
MODEL = "Qwen/Qwen3.5-397B-A17B"
DRAFT = "z-lab/Qwen3.5-397B-A17B-DFlash"
NUM_DRAFT_TOKENS = 16

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "libnuma-dev")
    .run_commands("pip install --upgrade pip setuptools wheel")
    .add_local_dir(f"{WT}/python", remote_path="/root/sglang/python", copy=True)
    .run_commands(
        "python -c \"import re,pathlib; p=pathlib.Path('/root/sglang/python/pyproject.toml'); "
        "s=p.read_text(); "
        "s=re.sub(r'\\[\\[tool\\.setuptools-rust\\.ext-modules\\]\\].*?(?=\\n\\[)', '', s, flags=re.S); "
        "p.write_text(s)\"",
        "cd /root/sglang && pip install -e python",
        "pip install datasets requests rich tqdm transformers numpy safetensors loguru blobfile hf_transfer",
    )
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"})
)
vol = modal.Volume.from_name("hf-cache")
SECRET = modal.Secret.from_name("huggingface-secret")


@app.function(image=image, volumes={"/cache": vol}, secrets=[SECRET], timeout=10800)
def dl():
    """Download target + draft into the hf-cache volume (~794GB)."""
    from huggingface_hub import snapshot_download
    import time
    for repo in (DRAFT, MODEL):
        t0 = time.time()
        print(f">>> downloading {repo} ...", flush=True)
        p = snapshot_download(repo, cache_dir="/cache/hub", token=os.environ.get("HF_TOKEN"))
        print(f">>> done {repo} in {time.time()-t0:.0f}s -> {p}", flush=True)
    vol.commit()
    return "ok"


def _write_mt_bench_jsonl(path):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    with open(path, "w") as f:
        for row in ds:
            t = row.get("prompt") or row.get("turns")
            if isinstance(t, str): t = [t]
            f.write(json.dumps({"turns": list(t)}) + "\n")
    return len(ds)


def _launch_args(ngpu, dynamic, mem_fraction, max_running, block_size=16):
    # Mirrors the official z-lab/Qwen3.5-397B-A17B-DFlash launch (linear-attn draft +
    # mamba scheduler + trtllm_mha target attn + tc_piecewise prefill graph). The earlier
    # flashinfer/num-draft-tokens config crashed verify graph capture -- wrong for this draft.
    mr = max(max_running, 1)
    cmd = [
        "python", "-m", "sglang.launch_server", "--model-path", MODEL, "--trust-remote-code",
        "--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", DRAFT,
        "--speculative-dflash-block-size", str(block_size),
        "--speculative-draft-attention-backend", "fa4",
        "--attention-backend", "trtllm_mha",
        "--linear-attn-prefill-backend", "triton",
        "--linear-attn-decode-backend", "flashinfer",
        "--mamba-ssm-dtype", "bfloat16",  # required by flashinfer linear-attn decode on SM100 (B200)
        "--mamba-scheduler-strategy", "extra_buffer",
        "--tp-size", str(ngpu),
        "--max-running-requests", str(mr),
        "--cuda-graph-max-bs-decode", str(mr),
        "--cuda-graph-backend-prefill", "tc_piecewise",
        "--flashinfer-allreduce-fusion-backend", "auto",
        "--mem-fraction-static", str(mem_fraction), "--port", "30000",
    ]
    cmd.append("--speculative-dflash-dynamic-vbs" if dynamic is True
               else "--no-speculative-dflash-dynamic-vbs" if dynamic is False else "")
    return [c for c in cmd if c]


@app.function(gpu="B200:8", cloud="aws", timeout=7200, image=image,
              volumes={"/cache": vol}, secrets=[SECRET])
def qsmoke(ngpu: int = 8, mem_fraction: float = 0.8, block_size: int = 8):
    import subprocess, signal, time, requests
    _write_mt_bench_jsonl("/root/mt-bench.jsonl")
    cmd = _launch_args(ngpu, None, mem_fraction, 32, block_size)
    print(">>> SERVER:", " ".join(cmd), flush=True)
    env = dict(os.environ); env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"
    srv = subprocess.Popen(cmd, cwd="/root/sglang", env=env)
    base = "http://127.0.0.1:30000"; ready = False
    for _ in range(1400):
        if srv.poll() is not None: return {"error": f"exited rc={srv.returncode}"}
        try:
            if requests.get(base + "/health", timeout=5).status_code == 200: ready = True; break
        except Exception: pass
        time.sleep(3)
    if not ready: srv.send_signal(signal.SIGINT); return {"error": "not ready"}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Explain speculative decoding in two sentences."}],
        tokenize=False, add_generation_prompt=True)
    t0 = time.perf_counter()
    r = requests.post(base + "/generate", json={"text": prompt,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 128}}, timeout=600)
    o = r.json(); m = o.get("meta_info", {})
    res = {"ok": True, "secs": round(time.perf_counter() - t0, 1),
           "completion_tokens": m.get("completion_tokens"), "accept_len": m.get("spec_accept_length"),
           "head": (o.get("text") or "")[:160]}
    srv.send_signal(signal.SIGINT); time.sleep(5)
    print(">>> SMOKE:", json.dumps(res), flush=True)
    return res


@app.local_entrypoint()
def qsmoke_main(mem_fraction: float = 0.8):
    print(qsmoke.remote(8, mem_fraction))


@app.function(gpu="B200:8", cloud="aws", timeout=12000, image=image,
              volumes={"/cache": vol}, secrets=[SECRET])
def qfinal_bench(concurrency: int, num_prompts: int = 256, max_new_tokens: int = 1024,
                 mem_fraction: float = 0.8, vbs_margin: float = 0.0, block_size: int = 16, ngpu: int = 8):
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
        it = dataset[i % len(dataset)]
        try: p = tok.apply_chat_template([{"role": "user", "content": it["turns"][0]}], tokenize=False, add_generation_prompt=True)
        except Exception: p = it["turns"][0]
        prompts.append(p)

    def run(dynamic):
        cmd = _launch_args(ngpu, dynamic, mem_fraction, max(concurrency, 1), block_size)
        env = dict(os.environ)
        env["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"
        env["SGLANG_DFLASH_VBS_MARGIN"] = str(vbs_margin)
        print(">>> SERVER:", " ".join(cmd), f"(margin={vbs_margin} block={block_size})", flush=True)
        srv = subprocess.Popen(cmd, cwd="/root/sglang", env=env)
        base = "http://127.0.0.1:30000"; ready = False
        for _ in range(1400):
            if srv.poll() is not None: return {"error": f"exited rc={srv.returncode}"}
            try:
                if requests.get(base + "/health", timeout=5).status_code == 200: ready = True; break
            except Exception: pass
            time.sleep(3)
        if not ready: srv.send_signal(signal.SIGINT); return {"error": "not ready"}
        def send(p):
            r = requests.post(base + "/generate", json={"text": p, "sampling_params":
                {"temperature": 0.0, "max_new_tokens": max_new_tokens}}, timeout=3600)
            r.raise_for_status(); o = r.json(); return o if isinstance(o, dict) else o[0]
        try: requests.get(base + "/flush_cache", timeout=60)
        except Exception: pass
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(send, prompts[:concurrency]))
        t0 = time.perf_counter(); total = 0; accs = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for fut in as_completed([pool.submit(send, p) for p in prompts[concurrency:]]):
                m = (fut.result().get("meta_info") or {})
                total += int(m.get("completion_tokens", 0))
                if "spec_accept_length" in m:
                    try: accs.append(float(m["spec_accept_length"]))
                    except Exception: pass
        dt = time.perf_counter() - t0
        srv.send_signal(signal.SIGINT); time.sleep(10)
        return {"tok_s": round(total / max(dt, 1e-6), 2),
                "accept_len": round(statistics.mean(accs), 3) if accs else None}

    off = run(False); on = run(True)
    ratio = (on["tok_s"] / off["tok_s"]) if (off.get("tok_s") and on.get("tok_s")) else None
    res = {"concurrency": concurrency, "num_prompts": num_prompts, "off": off, "on": on,
           "ratio": round(ratio, 4) if ratio else None}
    print(f"\n>>> QWEN397 FINAL conc={concurrency} n={num_prompts}: no-dyn={off.get('tok_s')} "
          f"(acc {off.get('accept_len')}) | dyn={on.get('tok_s')} (acc {on.get('accept_len')}) "
          f"| ratio={res['ratio']}", flush=True)
    return res


@app.local_entrypoint()
def qfinal(concurrencies: str = "1,32,128", num_prompts: int = 256, low_prompts: int = 64,
           vbs_margin: float = 0.0, block_size: int = 16):
    concs = [int(c) for c in concurrencies.split(",")]
    handles = [(c, qfinal_bench.spawn(c, (num_prompts if c >= 32 else low_prompts), 1024, 0.8,
                                      vbs_margin, block_size, 8)) for c in concs]
    print(f"\n==== Qwen3.5-397B-A17B dynamic-VBS FINAL (same 8xB200, serial; margin={vbs_margin} "
          f"block={block_size}) ====")
    print(f"{'conc':>6} {'n':>6} | {'no-dyn':>9} {'acc':>6} | {'dyn':>9} {'acc':>6} | ratio")
    for c, h in handles:
        try:
            r = h.get()
            print(f"{c:>6} {r['num_prompts']:>6} | {str(r['off'].get('tok_s')):>9} "
                  f"{str(r['off'].get('accept_len')):>6} | {str(r['on'].get('tok_s')):>9} "
                  f"{str(r['on'].get('accept_len')):>6} | {r['ratio']}")
        except Exception as e:
            print(f"{c:>6}: ERROR {e}")
