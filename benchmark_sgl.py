"""Standalone SGLang benchmark: launches the server, runs the benchmark, then shuts down."""

from __future__ import annotations

import json
import os
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from rich import print
from tqdm import tqdm
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "z-lab/Qwen3-8B-DFlash-b16"
NUM_DRAFT_TOKENS = 16
TEMPERATURE = 0.0
TOP_P = 1.0
DATASET_PATH = "/home/zlab/workspace/jianc/gsm8k.jsonl"
# DATASET_PATH = "./mt-bench.jsonl"
TP_SIZE = 1
MEM_FRACTION_STATIC = 0.75
SERVER_TIMEOUT = 600
MAX_NEW_TOKENS = 2048
NUM_PROMPTS = 768
CONCURRENCY = 64
PORT = 30000
TIMEOUT_S = 3600


def _load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f]


def _wait_for_server(base_url: str) -> None:
    """Poll the server health endpoint until it responds or timeout."""
    start = time.time()
    while time.time() - start < SERVER_TIMEOUT:
        try:
            resp = requests.get(base_url + "/health", timeout=5)
            if resp.status_code == 200:
                print(f"Server ready after {time.time() - start:.1f}s")
                return
        except requests.ConnectionError:
            pass
        time.sleep(5)
    raise TimeoutError(f"Server did not become ready within {SERVER_TIMEOUT}s")


def _launch_server() -> subprocess.Popen:
    """Launch sglang server as a subprocess."""
    env = os.environ.copy()
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--speculative-algorithm", "DFLASH",
        "--speculative-draft-model-path", DRAFT_MODEL,
        "--speculative-num-draft-tokens", str(NUM_DRAFT_TOKENS),
        "--tp-size", str(TP_SIZE),
        "--attention-backend", "flashinfer",
        "--mem-fraction-static", str(MEM_FRACTION_STATIC),
        "--max-running-requests", "64",
        "--port", str(PORT),
        "--trust-remote-code",
    ]

    print(f"Launching server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env)
    return proc


def _send_sglang(base_url: str, text: str) -> dict:
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": text,
            "sampling_params": {
                "temperature": TEMPERATURE, "top_p": TOP_P,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
        },
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    out = resp.json()
    return out if isinstance(out, dict) else out[0]


def _run_benchmark(base_url: str) -> None:
    """Run the throughput benchmark against a running server."""
    dataset = _load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    num_prompts = NUM_PROMPTS + CONCURRENCY
    prompts: list[str] = []
    for i in range(num_prompts):
        item = dataset[i % len(dataset)]
        # user_content = item["turns"][0]
        user_content = item["prompt"]
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        ))

    def send_one(prompt: str) -> dict:
        return _send_sglang(base_url, prompt)

    try:
        requests.get(base_url + "/flush_cache", timeout=60).raise_for_status()
    except Exception:
        print("Warning: /flush_cache failed. Continuing.")

    # Warmup
    if len(prompts) > CONCURRENCY:
        print(f"[warmup] {CONCURRENCY} requests ...")
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            list(pool.map(send_one, prompts[:CONCURRENCY]))
        prompts = prompts[CONCURRENCY:]

    # Benchmark
    print(f"Running benchmark: {NUM_PROMPTS} prompts, concurrency={CONCURRENCY} ...")
    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_lengths: list[float] = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(send_one, p): i for i, p in enumerate(prompts)}
        for fut in tqdm(as_completed(futures), total=len(prompts), desc="Benchmarking"):
            out = fut.result()
            meta = out.get("meta_info", {}) or {}
            total_tokens += int(meta.get("completion_tokens", 0))
            spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
            if "spec_accept_length" in meta:
                try:
                    spec_accept_lengths.append(float(meta["spec_accept_length"]))
                except (TypeError, ValueError):
                    pass

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    print(f"\n{'='*50}")
    print(f"Backend:          sglang")
    print(f"Dataset:          mt-bench")
    print(f"Num prompts:      {NUM_PROMPTS}")
    print(f"Concurrency:      {CONCURRENCY}")
    print(f"Latency:          {latency:.1f}s")
    print(f"Output tokens:    {total_tokens}")
    print(f"Throughput:       {toks_per_s:,.2f} tok/s")
    if spec_accept_lengths:
        print(f"Accept length:    {statistics.mean(spec_accept_lengths):.3f}")
    if spec_verify_ct_sum > 0:
        print(f"Spec verify ct:   {spec_verify_ct_sum}")
    print(f"{'='*50}")


def main() -> None:
    base_url = f"http://127.0.0.1:{PORT}"
    server_proc = _launch_server()

    try:
        _wait_for_server(base_url)
        _run_benchmark(base_url)
    finally:
        print("Shutting down server ...")
        server_proc.send_signal(signal.SIGTERM)
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()
        print("Server stopped.")


if __name__ == "__main__":
    main()
