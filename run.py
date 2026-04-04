#!/usr/bin/env python3
"""
Local runner mirroring modal_dflash_sweep.py: sweeps datasets × k_online settings
by loading benchmark/dflash/bench_dflash_sweep.py and invoking main().

Run from the repo root, e.g.:

    python run.py --data-names gsm8k,mt-bench --target-model Qwen/Qwen3-8B \\
        --draft-model z-lab/Qwen3-8B-DFlash-b16 --offset 3
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def run_dataset_sweep(
    target_model: str,
    data_name: str,
    draft_model: str | None = None,
    *,
    skip_baseline: bool = True,
    k_online: bool = False,
    k_online_offset: int = 2,
    block_verify: bool = False,
    disable_cuda_graph: bool = False,
    staging_dir: Path,
) -> str:
    """Run bench_dflash_sweep for one dataset / env combo; return markdown body or error text."""
    data_tag = data_name.replace(",", "-")
    run_tag = "k_online" if k_online else "no_k_online"
    block_verify_tag = "block_verify" if block_verify else "no_block_verify"
    output_file = f"results_{data_tag}_{target_model.split('/')[-1]}"
    if draft_model:
        output_file += f"_draft_{draft_model.split('/')[-1]}"
    if k_online:
        output_file += f"_k_online_off{k_online_offset}"
    output_file += f"_{run_tag}_{block_verify_tag}.md"

    staging_dir.mkdir(parents=True, exist_ok=True)
    output_path = staging_dir / output_file

    max_concurrency = 32
    args = [
        "bench_dflash_sweep.py",
        "--data-names",
        data_name,
        "--target-model",
        target_model,
        "--tp-sizes",
        "1",
        "--concurrencies",
        "32",
        "--output-md",
        str(output_path),
        "--max-running-requests",
        str(max_concurrency),
        "--attention-backends",
        "fa3",
        "--mem-fraction-static",
        "0.7",
    ]

    if skip_baseline:
        args.append("--skip-baseline")

    if draft_model:
        args.extend(["--draft-model", draft_model])

    if disable_cuda_graph:
        args.append("--disable-cuda-graph")

    print(
        f"Executing with args: {args}, k_online={k_online}, "
        f"disable_cuda_graph={disable_cuda_graph}"
    )

    env_updates = {
        "SGLANG_DFLASH_K_ONLINE": "1" if k_online else "0",
        "SGLANG_DFLASH_K_ONLINE_OFFSET": str(k_online_offset),
        "SGLANG_DFLASH_BLOCK_VERIFY": "1" if block_verify else "0",
    }

    sglang_path = str(REPO_ROOT / "python")
    if sglang_path not in sys.path:
        sys.path.insert(0, sglang_path)

    script_path = REPO_ROOT / "benchmark" / "dflash" / "bench_dflash_sweep.py"
    spec = importlib.util.spec_from_file_location("bench_sweep", script_path)
    if spec is None or spec.loader is None:
        return f"Error: could not load {script_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_sweep"] = module

    old_argv = sys.argv
    sys.argv = args
    old_cwd = os.getcwd()
    old_env = os.environ.copy()
    merged_env = os.environ.copy()
    merged_env.update(env_updates)

    try:
        os.chdir(REPO_ROOT)
        os.environ.clear()
        os.environ.update(merged_env)
        spec.loader.exec_module(module)
        if hasattr(module, "main"):
            module.main()
        else:
            return "Error: main() function not found in bench_dflash_sweep.py."
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        os.environ.clear()
        os.environ.update(old_env)

    if output_path.is_file():
        return output_path.read_text()

    return f"Error: Output file {output_path} not found."


def main() -> None:
    parser = argparse.ArgumentParser(description="Local DFlash dataset sweep (see modal_dflash_sweep.py).")
    parser.add_argument(
        "--data-names",
        type=str,
        default="gsm8k",
        help="Comma-separated dataset names passed to bench_dflash_sweep.",
    )
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", type=str, default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument(
        "--offset",
        type=int,
        default=3,
        help="SGLANG_DFLASH_K_ONLINE_OFFSET when k_online is enabled.",
    )
    args = parser.parse_args()

    combinations = [(args.target_model, args.draft_model)]

    dataset_list = [d.strip() for d in args.data_names.split(",") if d.strip()]
    base_results_dir = f"tmp_{args.data_names.replace(',', '_')}"

    staging_dir = REPO_ROOT / "benchmark" / "dflash" / ".local_sweep_staging"

    os.makedirs(REPO_ROOT / f"no_cuda_graph_{base_results_dir}", exist_ok=True)
    os.makedirs(REPO_ROOT / f"cuda_graph_{base_results_dir}", exist_ok=True)

    for disable_cuda_graph in [True]:
        if disable_cuda_graph:
            results_dir = REPO_ROOT / f"no_cuda_graph_{base_results_dir}"
        else:
            results_dir = REPO_ROOT / f"cuda_graph_{base_results_dir}"

        for target, draft in combinations:
            skip_baseline = True

            for dataset in dataset_list:
                for k_online in (False, True):
                    block_verify = not k_online

                    print(
                        f"\n>>> Running benchmark [{dataset}] "
                        f"k_online={k_online}, block_verify={block_verify}, "
                        f"disable_cuda_graph={disable_cuda_graph}: "
                        f"Target={target}, Draft={draft}..."
                    )

                    try:
                        res_content = run_dataset_sweep(
                            target,
                            dataset,
                            draft,
                            skip_baseline=skip_baseline,
                            k_online=k_online,
                            k_online_offset=args.offset,
                            block_verify=block_verify,
                            disable_cuda_graph=disable_cuda_graph,
                            staging_dir=staging_dir,
                        )

                        run_tag = "k_online" if k_online else "no_k_online"
                        block_verify_tag = "block_verify" if not k_online else "no_block_verify"

                        filename = f"res_{dataset.replace(',', '-')}_{target.split('/')[-1]}"
                        if draft:
                            filename += f"_vs_{draft.split('/')[-1]}"
                        else:
                            filename += "_baseline"
                        filename += f"_{run_tag}_{block_verify_tag}.md"

                        local_path = os.path.join(results_dir, filename)
                        with open(local_path, "w") as f:
                            f.write(res_content)

                        print(f">>> Completed: {filename}")
                    except Exception as e:
                        print(
                            f">>> Failed: dataset={dataset}, "
                            f"Target={target}, Draft={draft}, k_online={k_online}. Error: {e}"
                        )

    print(f"\nAll benchmarks for '{args.data_names}' finished.")


if __name__ == "__main__":
    main()
