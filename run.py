import argparse
import os
import sys
from types import SimpleNamespace


def run_bench(
    target_model: str,
    data_names: str,
    draft_model: str = None,
    skip_baseline: bool = True,
    k_online: bool = False,
    k_online_offset: int = 2,
    k_online_warmup: int = 100,
    block_verify: bool = False,
    tree_verify: bool = False,
    tree_verify_topk: int = 1,
    disable_cuda_graph: bool = False,
    speculative_num_draft_tokens: int | None = None,
    results_dir: str = "test_results",
):
    """本地直接调用 bench_dflash_sweep.py"""

    # 构建输出文件名
    data_tag = data_names.replace(",", "-")
    run_tag = "k_online" if k_online else "no_k_online"
    block_verify_tag = "block_verify" if block_verify else "no_block_verify"
    tree_verify_tag = "tree_verify" if tree_verify else "no_tree_verify"
    output_file = f"results_{data_tag}_{target_model.split('/')[-1]}"
    if draft_model:
        output_file += f"_draft_{draft_model.split('/')[-1]}"
    if k_online:
        output_file += f"_k_online_off{k_online_offset}_w{k_online_warmup}"
    output_file += f"_{run_tag}_{block_verify_tag}_{tree_verify_tag}.md"

    output_path = os.path.abspath(os.path.join(results_dir, output_file))
    os.makedirs(results_dir, exist_ok=True)

    # 准备环境变量
    env = os.environ.copy()

    # Debug toggles (optional; configure in shell if needed)
    # e.g. SGLANG_DFLASH_DEBUG=1 CUDA_LAUNCH_BLOCKING=1 python local_finetune_test.py

    env["SGLANG_DFLASH_K_ONLINE"] = "1" if k_online else "0"
    env["SGLANG_DFLASH_K_ONLINE_OFFSET"] = str(k_online_offset)
    env["SGLANG_DFLASH_K_ONLINE_WARMUP"] = str(k_online_warmup)
    env["SGLANG_DFLASH_BLOCK_VERIFY"] = "1" if block_verify else "0"
    env["SGLANG_DFLASH_TREE_VERIFY"] = "1" if tree_verify else "0"

    # 确保 PYTHONPATH 包含当前目录下的 python 文件夹
    project_root = os.getcwd()
    sglang_python_path = os.path.join(project_root, "python")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{sglang_python_path}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = sglang_python_path

    # 构建 bench 参数
    max_concurrency = 1

    bench_args = SimpleNamespace(
        data_name=None,
        data_names=data_names,
        output_md=output_path,
        target_model=target_model,
        draft_model=draft_model,
        skip_baseline=skip_baseline,
        batch_requests=False,
        prompt_style="chat",
        max_new_tokens=2048,
        timeout_s=3600,
        mem_fraction_static=0.7,
        disable_radix_cache=False,
        dtype="bfloat16",
        max_running_requests=max_concurrency,
        tp_sizes="1",
        concurrencies="1",
        samples_per_concurrency_base=128,
        max_samples_per_config=1,
        attention_backends="flashinfer",
        disable_cuda_graph=disable_cuda_graph,
        enable_piecewise_cuda_graph=False,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        speculative_eagle_topk=tree_verify_topk,
    )

    print(
        "\n[Local Run] "
        f"k_online={k_online}, tree_verify={tree_verify}, disable_cuda_graph={disable_cuda_graph}"
    )

    # 直接调用 benchmark 脚本 main
    bench_cwd = os.getcwd()
    try:
        os.environ.update(env)
        from benchmark.dflash import bench_dflash_sweep

        old_argv = sys.argv[:]
        sys.argv = [
            "bench_dflash_sweep.py",
            "--data-names",
            str(bench_args.data_names),
            "--output-md",
            str(bench_args.output_md),
            "--target-model",
            str(bench_args.target_model),
            "--draft-model",
            str(bench_args.draft_model),
            "--tp-sizes",
            str(bench_args.tp_sizes),
            "--concurrencies",
            str(bench_args.concurrencies),
            "--samples-per-concurrency-base",
            str(bench_args.samples_per_concurrency_base),
            "--max-samples-per-config",
            str(bench_args.max_samples_per_config),
            "--attention-backends",
            str(bench_args.attention_backends),
            "--max-new-tokens",
            str(bench_args.max_new_tokens),
            "--timeout-s",
            str(bench_args.timeout_s),
            "--mem-fraction-static",
            str(bench_args.mem_fraction_static),
            "--dtype",
            str(bench_args.dtype),
            "--max-running-requests",
            str(bench_args.max_running_requests),
        ]
        if bench_args.speculative_num_draft_tokens is not None:
            sys.argv.extend(
                [
                    "--speculative-num-draft-tokens",
                    str(bench_args.speculative_num_draft_tokens),
                ]
            )
        if bench_args.speculative_eagle_topk is not None:
            sys.argv.extend(
                [
                    "--speculative-eagle-topk",
                    str(bench_args.speculative_eagle_topk),
                ]
            )
        if bench_args.skip_baseline:
            sys.argv.append("--skip-baseline")
        if bench_args.batch_requests:
            sys.argv.append("--batch-requests")
        if bench_args.disable_radix_cache:
            sys.argv.append("--disable-radix-cache")
        if bench_args.disable_cuda_graph:
            sys.argv.append("--disable-cuda-graph")
        if bench_args.enable_piecewise_cuda_graph:
            sys.argv.append("--enable-piecewise-cuda-graph")

        bench_dflash_sweep.main()
        sys.argv = old_argv
        print(f">>> Finished. Output saved to: {output_path}")
    except Exception as e:
        print(f">>> Failed with error: {e}")
    finally:
        os.chdir(bench_cwd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-names", type=str, default="math500"
    )
    # ,math500,humaneval,mt-bench
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", type=str, default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument("--offset", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--speculative-num-draft-tokens",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    results_dir = f"{args.data_names.replace(',', '_')}"

    # 跑多组对比：枚举 cuda graph 开关 + tree_verify 开关（k_online 一直关，block verify 一直开）
    for disable_cuda_graph in [False]:
        run_dir = (
            "no_cuda_graph_" + results_dir
            if disable_cuda_graph
            else "cuda_graph_" + results_dir
        )
        for tree_verify in [True]:
            print(f"\n" + "=" * 60)
            print(
                "Starting benchmark: "
                f"tree_verify={tree_verify}, disable_cuda_graph={disable_cuda_graph}"
            )
            print("=" * 60)

            run_bench(
                target_model=args.target_model,
                data_names=args.data_names,
                draft_model=args.draft_model,
                k_online=False,
                k_online_offset=args.offset,
                k_online_warmup=args.warmup,
                block_verify=True,
                tree_verify=tree_verify,
                tree_verify_topk=2,
                disable_cuda_graph=disable_cuda_graph,
                speculative_num_draft_tokens=args.speculative_num_draft_tokens,
                results_dir=run_dir,
            )


if __name__ == "__main__":
    main()