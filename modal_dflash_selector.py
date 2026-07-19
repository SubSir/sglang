"""P160 DFlash candidate-selector in SGLang vs the DSpark baseline (1x B200).

The selector (direct-edge parallel-scan, r=256, K=16, block_size=7) shares the
DFlash Qwen3 backbone; the only new weights are 3 projections. We add
models/dflash_selector.py (class Qwen3DFlashSelectorModel) + a branch in the
DFLASH worker that decodes the block from the top-K lattice instead of per-position
argmax. Nothing else in sglang changes.

Comparison (same training step 2000, both propose 7 tokens, verify window 8, greedy):
  - selector: --speculative-algorithm DFLASH, class Qwen3DFlashSelectorModel,
              --speculative-dflash-block-size 8  (block-1 == 7 selector slots)
              decoder via SGLANG_DFLASH_SELECTOR_DECODER = local | normalized-map
  - dspark:   --speculative-algorithm DSPARK (native markov head, no confidence),
              --speculative-dspark-block-size 7 (gamma 7 -> verify 8)

  modal run modal_dflash_selector.py::check       # no-GPU: overlay imports + class registered
  modal run modal_dflash_selector.py::dump_ckpt   # no-GPU: config.json + weight names
  modal run modal_dflash_selector.py::smoke       # 1x B200: load + 1 generate, accept>1
  modal run modal_dflash_selector.py::bench       # 1x B200: gsm8k accept_length A/B
"""
import os

import modal

app = modal.App("dflash-selector")

# autodflash-dspark-train holds both the p160 selector and the dspark baseline.
vol_train = modal.Volume.from_name("autodflash-dspark-train")
results = modal.Volume.from_name("dflash-selector-results", create_if_missing=True)

TARGET = "Qwen/Qwen3-4B"

P160 = "p160-mass-weighted-epoch1-direct-edge-parallel-scan-r256-k16-local4-gbs64-lr6e4-3ep-seed42"
DSPARK = "dspark-qwen3-4b-no-confidence-dsblock7-bsz64-local4-3ep-scratch-seed42"
DFLASH_BASE = "dflash-qwen3-4b-baseline-dsblock7-bsz64-3ep-scratch-seed42"

# arm -> (checkpoint dir on /vol, algorithm, architectures class)
ARMS = {
    "selector": (f"/vol/{P160}/draft-2000", "DFLASH", "Qwen3DFlashSelectorModel"),
    "dspark": (f"/vol/{DSPARK}/draft-2000", "DSPARK", "Qwen3DSparkModel"),
    "dflash": (f"/vol/{DFLASH_BASE}/draft-2000", "DFLASH", "DFlashDraftModel"),
}

image = (
    # Match the fork exactly: it is a cu13 build (pyproject pins flashinfer[cu13],
    # humming-kernels[cu13], sgl-kernel 0.4.5) at commit 13b5bd96. The cu12 images
    # ship kernel 0.4.4 and cu12 kernels -> the draft path computes garbage. Use the
    # cu13 image at the same commit so the draft kernels match.
    modal.Image.from_registry("lmsysorg/sglang:nightly-dev-cu13-20260627-13b5bd96")
    .pip_install("datasets")
    .add_local_dir("python/sglang", "/root/overlay/sglang", copy=True)
    .run_commands(
        "echo dflash_selector_overlay_v1 && "
        "SITE=$(python -c 'import sglang,os;print(os.path.dirname(os.path.dirname(sglang.__file__)))') && "
        'echo "sglang site at $SITE" && cp -rf /root/overlay/sglang/. "$SITE/sglang/" && '
        "echo overlaid dflash-selector worktree on nightly sglang",
    )
)


@app.function(image=image, timeout=600)
def check():
    """No-GPU: confirm the overlay imports and the selector class is registered."""
    import subprocess as sp

    code = (
        "import sglang; print('sglang', sglang.__version__);"
        "from sglang.srt.models.dflash_selector import Qwen3DFlashSelectorModel, CandidateSelector;"
        "print('selector class', Qwen3DFlashSelectorModel.__name__);"
        "from sglang.srt.models.registry import ModelRegistry;"
        "cls = ModelRegistry.resolve_model_cls(['Qwen3DFlashSelectorModel']);"
        "print('registry resolves', cls);"
        "import inspect;"
        "src=inspect.getsource(__import__('sglang.srt.speculative.dflash_worker_v2', fromlist=['x']));"
        "print('worker wired', '_propose_selector_block' in src and 'candidate_selector' in src)"
    )
    r = sp.run(["python", "-c", code], capture_output=True, text=True)
    print("STDOUT:", r.stdout)
    print("STDERR(tail):", "\n".join(r.stderr.splitlines()[-30:]))
    return r.stdout


@app.function(image=image, timeout=600, volumes={"/vol": vol_train})
def dump_ckpt(arm: str = "selector"):
    """No-GPU: config.json + weight-name prefixes so we can confirm the load path."""
    import glob
    import json

    ckpt = ARMS[arm][0]
    print("=== ckpt dir ===", ckpt)
    print(sorted(os.listdir(ckpt)))
    with open(f"{ckpt}/config.json") as f:
        cfg = json.load(f)
    print("=== config.json ===")
    print(json.dumps(cfg, indent=2))
    names = []
    try:
        from safetensors import safe_open

        sts = glob.glob(f"{ckpt}/*.safetensors")
        if sts:
            with safe_open(sts[0], framework="pt") as g:
                names = list(g.keys())
    except Exception as e:
        print("weight enumerate failed:", e)
    print("=== selector / fc / norm weights ===")
    print([n for n in names if any(k in n for k in ("candidate_selector", "markov", "fc.", "hidden_norm", "norm."))][:40])
    print("=== embed / lm_head present? ===")
    print([n for n in names if any(k in n for k in ("lm_head", "embed_tokens"))])
    return {"n_weights": len(names)}


def _prep_draft(src: str, architectures: str) -> str:
    """Copy checkpoint to a writable dir, rewrite architectures so sglang routes to
    our class. Symlink the (multi-GB) weights, overwrite only config.json."""
    import json

    dst = f"/tmp/draft_{architectures}_{os.path.basename(os.path.dirname(src))}"
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        link = f"{dst}/{name}"
        if name != "config.json" and not os.path.exists(link):
            os.symlink(f"{src}/{name}", link)
    with open(f"{src}/config.json") as f:
        cfg = json.load(f)
    cfg["architectures"] = [architectures]
    with open(f"{dst}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return dst


def _launch(arm: str, env_extra: dict, disable_cuda_graph: bool = False,
            block_size: int = 8):
    """Start an sglang server for one arm. Returns (proc, ready)."""
    import subprocess
    import time
    import urllib.request

    src, algo, arch = ARMS[arm]
    draft = _prep_draft(src, arch)
    env = dict(os.environ)
    # cu13 image matches the fork's cuda target; only the kernel version tag lags
    # (image 0.4.4 vs fork's unreleased 0.4.5). Skip the assert and test if the cu13
    # draft kernels work.
    env["SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"] = "1"
    env.update({k: str(v) for k, v in env_extra.items()})

    # DFLASH chain uses flashinfer for the draft attention (per the fork's own
    # e2e tests: test_gemma4_dflash uses flashinfer, Kimi uses fa4). triton draft
    # attention miscomputes chain DFLASH and yields garbage draft hidden.
    if algo == "DFLASH":
        # num_draft_tokens == verify window; predictions == num_draft_tokens-1.
        spec = ["--speculative-algorithm", "DFLASH",
                "--speculative-draft-model-path", draft,
                "--speculative-num-draft-tokens", str(block_size),
                "--speculative-draft-attention-backend", "flashinfer"]
    elif algo == "DSPARK":
        # gamma 7 -> verify window gamma+1 == 8.
        spec = ["--speculative-algorithm", "DSPARK",
                "--speculative-draft-model-path", draft,
                "--speculative-dspark-block-size", "7",
                "--speculative-draft-attention-backend", "flashinfer"]
    else:
        raise ValueError(algo)

    server = [
        "python", "-m", "sglang.launch_server",
        "--model-path", TARGET, "--trust-remote-code",
        "--mem-fraction-static", "0.85", "--max-running-requests", "64",
        "--attention-backend", "triton",
        *spec, "--port", "30000",
    ]
    if disable_cuda_graph:
        server.append("--disable-cuda-graph")
    print(">>> ARM:", arm, "ENV:", env_extra, flush=True)
    print(">>> SERVER:", " ".join(server), flush=True)
    srv = subprocess.Popen(server, env=env)
    ready = False
    for _ in range(300):
        try:
            urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=2)
            ready = True
            break
        except Exception:
            time.sleep(2)
        if srv.poll() is not None:
            print("!!! server exited early, code", srv.returncode, flush=True)
            break
    return srv, ready


def _mode_env(arm: str, decoder: str):
    if arm == "selector":
        return {"SGLANG_DFLASH_SELECTOR_DECODER": decoder}
    return {}


@app.function(gpu="B200", timeout=3600, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results, "/vol": vol_train})
def smoke(arm: str = "selector", decoder: str = "local", disable_cuda_graph: bool = False,
          block_size: int = 8):
    """Load + one greedy generate; report accept_length (expect > 1)."""
    import json
    import signal
    import urllib.request

    srv, ready = _launch(arm, _mode_env(arm, decoder), disable_cuda_graph=disable_cuda_graph,
                         block_size=block_size)
    result = {"arm": arm, "decoder": decoder, "block_size": block_size, "ready": ready}
    if ready:
        prompt = "Q: If a train travels 60 miles in 1.5 hours, what is its average speed? A:"
        req = urllib.request.Request(
            "http://127.0.0.1:30000/generate",
            data=json.dumps({"text": prompt,
                             "sampling_params": {"temperature": 0.0, "max_new_tokens": 128}}).encode(),
            headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
        mi = r["meta_info"]
        result["completion_tokens"] = mi.get("completion_tokens")
        result["spec_accept_length"] = mi.get("spec_accept_length")
        result["text_head"] = r["text"][:200]
        print("=== RESULT ===", json.dumps(result, indent=2), flush=True)
    srv.send_signal(signal.SIGINT)
    with open(f"/results/smoke_{arm}_{decoder}.json", "w") as f:
        json.dump(result, f, indent=2)
    results.commit()
    return result


@app.function(gpu="B200", timeout=3600, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results, "/vol": vol_train})
def profile(arm: str = "selector", decoder: str = "local", num_steps: int = 40,
            max_new_tokens: int = 256):
    """sglang torch-profiler trace of the decode loop (cuda graph ON).

    Drives /start_profile (num_steps auto-stops) then one generate; traces land on
    /results/profiles/<arm> and are committed to the volume. Fetch with:
      modal volume get dflash-selector-results profiles/<arm> ./
    """
    import json
    import signal
    import time
    import urllib.request

    out_dir = f"/results/profiles/{arm}"
    os.makedirs(out_dir, exist_ok=True)
    env_extra = _mode_env(arm, decoder)
    # cuda graph ON (no disable) so the profile reflects the graph-replay decode.
    srv, ready = _launch(arm, env_extra)
    result = {"arm": arm, "ready": ready, "out_dir": out_dir}
    if ready:
        prompt = ("Q: If a train travels 60 miles in 1.5 hours, then speeds up and "
                  "covers 120 miles in the next 2 hours, what is its overall average "
                  "speed for the whole trip? Show your reasoning. A:")

        def _post(path, body):
            req = urllib.request.Request(
                f"http://127.0.0.1:30000/{path}",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            return urllib.request.urlopen(req, timeout=300).read()

        # Warm up (trigger cuda-graph replay path) before profiling.
        _post("generate", {"text": prompt,
                           "sampling_params": {"temperature": 0.0, "max_new_tokens": 16}})
        # Start profiler: num_steps auto-stops + writes the trace, no stop needed.
        _post("start_profile", {"output_dir": out_dir, "num_steps": num_steps,
                                "activities": ["CPU", "GPU"], "profile_prefix": arm})
        r = json.loads(_post("generate", {"text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens}}))
        result["spec_accept_length"] = r["meta_info"].get("spec_accept_length")
        result["completion_tokens"] = r["meta_info"].get("completion_tokens")
        # Give the profiler a moment to flush trace files to disk.
        time.sleep(20)
        result["trace_files"] = sorted(os.listdir(out_dir))
        print("=== PROFILE RESULT ===", json.dumps(result, indent=2), flush=True)
    srv.send_signal(signal.SIGINT)
    results.commit()
    return result


@app.function(gpu="B200", timeout=10800, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": results, "/vol": vol_train})
def bench(modes: str = "selector:local,selector:normalized-map,dspark:-",
          dataset: str = "gsm8k", n: int = 80, temperature: float = 0.0):
    """accept_length for each arm/decoder, same prompts. Sequential per mode.

    modes: comma list of 'arm:decoder' (decoder ignored for dspark).
    temperature 0.0 = greedy; >0 exercises the (T=1) sampling path.
    """
    import json
    import signal
    import time
    import urllib.request

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TARGET)
    boxed = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
        msgs = [[{"role": "user", "content": boxed.format(q=ds[i]["question"])}] for i in range(n)]
    elif dataset == "mt-bench":
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        msgs = [[{"role": "user", "content": ds[i % len(ds)]["prompt"][0]}] for i in range(n)]
    else:
        raise ValueError(f"unknown dataset {dataset}")
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for m in msgs]

    def gen(p, acc):
        req = urllib.request.Request(
            "http://127.0.0.1:30000/generate",
            data=json.dumps({"text": p,
                             "sampling_params": {"temperature": temperature,
                                                 "max_new_tokens": 512}}).encode(),
            headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        mi = r["meta_info"]
        acc.append((mi["completion_tokens"], mi.get("spec_accept_length", 0.0)))

    out = {"dataset": dataset, "n": n, "temperature": temperature, "rows": []}
    for mode in modes.split(","):
        arm, _, decoder = mode.partition(":")
        decoder = decoder or "local"
        srv, ready = _launch(arm, _mode_env(arm, decoder))
        row = {"arm": arm, "decoder": decoder if arm == "selector" else "-", "ready": ready}
        if ready:
            acc = []
            gen(prompts[0], [])  # warmup
            t0 = time.perf_counter()
            for p in prompts:
                gen(p, acc)
            dt = time.perf_counter() - t0
            row["accept_len"] = round(sum(a for _, a in acc) / len(acc), 3)
            row["tokens"] = sum(c for c, _ in acc)
            row["secs"] = round(dt, 1)
            print(f"=== {mode}: accept_len={row['accept_len']} ({dt:.0f}s) ===", flush=True)
        out["rows"].append(row)
        srv.send_signal(signal.SIGINT)
        time.sleep(8)
    with open(f"/results/bench_{dataset}_t{temperature}.json", "w") as f:
        json.dump(out, f, indent=2)
    results.commit()
    print("=== SUMMARY ===", json.dumps(out, indent=2), flush=True)
    return out
