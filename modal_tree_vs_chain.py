"""Focused DFlash tree-vs-chain experiment (conc=1 regime where draft trees win).

Runs Qwen3-8B + z-lab/Qwen3-8B-DFlash-b16, block_size=16, on a single B200,
comparing chain verify (topk=1) vs tree verify (topk=4) at concurrency=1 on a
fast kernel (fa4). Reports accept length, throughput, and verify-forward count.

Usage:
  modal run modal_tree_vs_chain.py
  modal run modal_tree_vs_chain.py --target-model Qwen/Qwen3-8B \
      --draft-model z-lab/Qwen3-8B-DFlash-b16 --data-names gsm8k,humaneval
"""

import os
import modal

app = modal.App("dflash-tree-vs-chain")

base_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "libnuma-dev")
)

local_image = (
    base_image.run_commands(
        "echo 80 > /tmp/build_time",
        "git clone https://github.com/SubSir/sglang.git /root/sglang_local",
        "cd /root/sglang_local && pip install -e \"python\"",
        # sglang-kernel is NOT a pyproject dep; the Dockerfile installs it separately.
        "pip install sglang-kernel==0.4.1",
        "pip install --upgrade --force-reinstall nvidia-cudnn-cu12==9.16.0.29",
    )
    .add_local_dir("./python", remote_path="/root/sglang_local/python_local", copy=True)
    .add_local_dir("./benchmark", remote_path="/root/sglang_local/benchmark_local", copy=True)
    .run_commands(
        "rm -rf /root/sglang_local/python && cp -r /root/sglang_local/python_local /root/sglang_local/python",
        "rm -rf /root/sglang_local/benchmark && cp -r /root/sglang_local/benchmark_local /root/sglang_local/benchmark",
    )
)


@app.function(
    gpu="B200",
    timeout=5400,
    image=local_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run(
    target_model: str,
    draft_model: str,
    data_names: str,
    tree_verify: bool,
    topk: int,
    block_size: int,
    num_draft_tokens: int | None,
    samples_base: int,
    backend: str,
    mem_fraction: float = 0.85,
    disable_cuda_graph: bool = False,
):
    import sys, importlib.util

    args = [
        "bench_dflash_sweep.py",
        "--data-names", data_names,
        "--target-model", target_model,
        "--draft-model", draft_model,
        "--tp-sizes", "1",
        "--concurrencies", "1",
        "--samples-per-concurrency-base", str(samples_base),
        "--max-samples-per-config", str(samples_base),
        "--max-new-tokens", "1024",
        "--attention-backends", backend,
        "--mem-fraction-static", str(mem_fraction),
        "--speculative-eagle-topk", str(topk),
        "--speculative-dflash-block-size", str(block_size),
        "--skip-baseline",
    ]
    if disable_cuda_graph:
        args.append("--disable-cuda-graph")
    if num_draft_tokens is not None:
        args += ["--speculative-num-draft-tokens", str(num_draft_tokens)]

    env = os.environ
    env["SGLANG_DFLASH_TREE_VERIFY"] = "1" if tree_verify else "0"
    env["SGLANG_DFLASH_BLOCK_VERIFY"] = "0" if tree_verify else "1"
    # fa4's cute kernel is broken in this image (flash_attn/cutlass-dsl fmax mismatch);
    # force the draft attention backend to match the (flashinfer) main backend.
    env["DFLASH_DRAFT_ATTN_BACKEND"] = backend

    out_path = f"/root/out_{'tree' if tree_verify else 'chain'}_{topk}.md"
    args += ["--output-md", out_path]

    sys.path.insert(0, "/root/sglang_local/python")
    os.chdir("/root/sglang_local")
    sys.argv = args
    spec = importlib.util.spec_from_file_location(
        "bench", "/root/sglang_local/benchmark/dflash/bench_dflash_sweep.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    spec.loader.exec_module(mod)
    mod.main()
    with open(out_path) as f:
        return f.read()


@app.local_entrypoint()
def sweep(
    target_model: str = "Qwen/Qwen3-8B",
    draft_model: str = "z-lab/Qwen3-8B-DFlash-b16",
    data_names: str = "gsm8k",
    block_size: int = 16,
    samples_base: int = 32,
    backend: str = "flashinfer",
):
    """Accept-length vs tree budget curve (cuda graph off, low mem to avoid OOM)."""
    # (label, tree_verify, topk, num_draft_tokens)
    configs = [
        ("chain_b16", False, 1, None),
        ("tree_topk4_budget16", True, 4, 16),
        ("tree_topk4_budget32", True, 4, 32),
        ("tree_topk4_budget64", True, 4, 64),
    ]
    calls = []
    for label, tv, topk, ndt in configs:
        print(f">>> spawning {label}")
        calls.append((label, run.spawn(
            target_model, draft_model, data_names, tv, topk, block_size,
            ndt, samples_base, backend, 0.6, True,  # mem_fraction=0.6, disable_cuda_graph
        )))
    os.makedirs("tree_vs_chain_results", exist_ok=True)
    for label, c in calls:
        try:
            res = c.get()
            p = f"tree_vs_chain_results/sweep_{label}_{target_model.split('/')[-1]}_{data_names.replace(',','-')}.md"
            with open(p, "w") as f:
                f.write(res)
            # pull accept length + tok/s from the table
            al = tk = None
            lines = res.splitlines()
            for i, line in enumerate(lines):
                if "acceptance length" in line.lower():
                    al = lines[i + 3].split("|")[2].strip() if i + 3 < len(lines) else "?"
                if "DFLASH output tok/s" in line:
                    tk = lines[i + 3].split("|")[2].strip() if i + 3 < len(lines) else "?"
            print(f">>> {label}: accept_len={al} toks/s={tk}")
        except Exception as e:
            print(f">>> FAILED {label}: {e}")


@app.local_entrypoint()
def main(
    target_model: str = "Qwen/Qwen3-8B",
    draft_model: str = "z-lab/Qwen3-8B-DFlash-b16",
    data_names: str = "gsm8k",
    block_size: int = 16,
    samples_base: int = 40,
    backend: str = "flashinfer",
):
    # (label, tree_verify, topk, num_draft_tokens)
    # Matched token budget: chain=16 (block), tree=16 nodes topk=4.
    configs = [
        ("chain_b16", False, 1, None),
        ("tree_topk4_b16_budget16", True, 4, 16),
        ("tree_topk4_b16_budget32", True, 4, 32),
    ]
    calls = []
    for label, tv, topk, ndt in configs:
        print(f">>> spawning {label}")
        calls.append((label, run.spawn(
            target_model, draft_model, data_names, tv, topk, block_size,
            ndt, samples_base, backend,
        )))
    os.makedirs("tree_vs_chain_results", exist_ok=True)
    for label, c in calls:
        try:
            res = c.get()
            p = f"tree_vs_chain_results/{label}_{target_model.split('/')[-1]}_{data_names.replace(',','-')}.md"
            with open(p, "w") as f:
                f.write(res)
            print(f">>> done {label} -> {p}")
            # echo accept length + tok/s lines for quick read
            for line in res.splitlines():
                if "accept" in line.lower() or "tok/s" in line.lower() or "Speedup" in line:
                    print(f"    {label}: {line.strip()}")
        except Exception as e:
            print(f">>> FAILED {label}: {e}")
