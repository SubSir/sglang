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

WT = "/Users/subsir/Desktop/Studio/Python/sglang-v2-tree"
UPSTREAM_COMMIT = "e0c0c0a45"

base_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "curl", "libnuma-dev", "build-essential",
                 "pkg-config", "protobuf-compiler")
    .run_commands(
        "echo v2tree2 > /tmp/build_time",
        # Upstream now builds a Rust gRPC extension via setuptools-rust.
        "curl --proto '=https' --tlsv1.2 --retry 3 -sSf https://sh.rustup.rs | sh -s -- -y",
        "git clone https://github.com/sgl-project/sglang.git /root/sglang_local",
        f"cd /root/sglang_local && git checkout {UPSTREAM_COMMIT}",
        "cd /root/sglang_local && PATH=/root/.cargo/bin:$PATH pip install -e \"python\"",
        "pip install sglang-kernel==0.4.4",
        "pip install --upgrade --force-reinstall nvidia-cudnn-cu12==9.16.0.29",
    )
    # Overlay the tree-verify port (speculative dir from the worktree = upstream + port).
    .add_local_dir(
        f"{WT}/python/sglang/srt/speculative",
        remote_path="/root/sglang_local/python/sglang/srt/speculative",
        copy=True,
    )
    # Overlay the fork's patched bench harness.
    .add_local_dir(
        "./benchmark",
        remote_path="/root/sglang_local/benchmark",
        copy=True,
    )
)


@app.function(gpu="B200", timeout=10800, image=base_image,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run(target_model, draft_model, data_names, tree_verify, topk, block_size,
        num_draft_tokens, samples_base, concurrencies, backend="flashinfer",
        mem_fraction=0.8, tp=1, gpu_count=1):
    """One server launch; bench internally sweeps all concurrencies x datasets."""
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
        "--skip-baseline",
    ]
    if num_draft_tokens is not None:
        args += ["--speculative-num-draft-tokens", str(num_draft_tokens)]

    out_path = f"/root/out_{'tree' if tree_verify else 'chain'}_b{block_size}_t{topk}.md"
    args += ["--output-md", out_path]

    env = os.environ
    env["SGLANG_DFLASH_TREE_VERIFY"] = "1" if tree_verify else "0"
    env["DFLASH_DRAFT_ATTN_BACKEND"] = backend  # avoid broken fa4

    sys.path.insert(0, "/root/sglang_local/python")
    os.chdir("/root/sglang_local")
    sys.argv = [a for a in args if a]
    spec = importlib.util.spec_from_file_location(
        "bench", "/root/sglang_local/benchmark/dflash/bench_dflash_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    spec.loader.exec_module(mod)
    mod.main()
    with open(out_path) as f:
        return f.read()


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

    configs = [("chain", False, 1, block_size)]
    for b in budget_list:
        configs.append((f"tree_b{b}", True, 4, b))

    os.makedirs("v2_tree_results", exist_ok=True)
    mtag = target_model.split("/")[-1]
    # Spawn all configs in parallel (each is its own container/GPU).
    calls = []
    for label, tv, topk, ndt in configs:
        print(f">>> spawning {label} (budget={ndt}, conc={concurrencies}, data={data_names})")
        calls.append((label, ndt, run.spawn(
            target_model, draft_model, data_names, tv, topk, block_size,
            ndt, samples_base, concurrencies, "flashinfer", mem_fraction, tp, tp)))
    for label, ndt, c in calls:
        try:
            res = c.get()
            with open(f"v2_tree_results/sweep_{mtag}_{label}.md", "w") as f:
                f.write(res)
            print(f"\n========== {mtag} / {label} (budget={ndt}) ==========")
            print(_tables(res))
        except Exception as e:
            print(f">>> FAILED {label}: {e}")
