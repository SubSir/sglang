import os
import modal

app = modal.App("sglang-dflash-dataset-sweep")

# Reuse base image configuration
base_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "libnuma-dev")
)

local_image = (
    base_image
    .run_commands(
        "echo 60 > /tmp/build_time",
        "git clone https://github.com/SubSir/sglang.git /root/sglang_local",
        "cd /root/sglang_local && pip install -e \"python\"",
        "pip install --upgrade --force-reinstall nvidia-cudnn-cu12==9.16.0.29",
    )
)

# 将本地代码打包进镜像后，再在运行时复制到目标目录
local_image = (
    local_image
    .add_local_dir(
        "./python",
        remote_path="/root/sglang_local/python_local",
        copy=True,
    )
    .add_local_dir(
        "./benchmark",
        remote_path="/root/sglang_local/benchmark_local",
        copy=True,
    )
    .run_commands(
        "rm -rf /root/sglang_local/python && cp -r /root/sglang_local/python_local /root/sglang_local/python",
        "rm -rf /root/sglang_local/benchmark && cp -r /root/sglang_local/benchmark_local /root/sglang_local/benchmark"
    )
)

@app.function(
    gpu="H200",
    timeout=7200,
    image=local_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    cloud="aws"
)
def run_dataset_sweep(
    target_model: str,
    data_name: str,
    draft_model: str = None,
    skip_baseline: bool = True,
    k_online: bool = False,
    k_online_offset: int = 2,
    k_online_warmup: int = 100,
    block_verify: bool = False,
    tree_verify: bool = False,
    tree_verify_topk: int = 1,
    tree_verify_num_draft_tokens: int = None,
    disable_cuda_graph: bool = False,
    speculative_dflash_block_size: int = 10,
):
    """
    Run benchmark/dflash/bench_dflash_sweep.py for a specific dataset.
    """
    import sys
    import os
    import importlib.util

    # Generate output filename based on dataset and models
    data_tag = data_name.replace(",", "-")
    run_tag = "k_online" if k_online else "no_k_online"
    block_verify_tag = "block_verify" if block_verify else "no_block_verify"
    tree_verify_tag = "tree_verify" if tree_verify else "no_tree_verify"
    output_file = f"results_{data_tag}_{target_model.split('/')[-1]}"
    if draft_model:
        output_file += f"_draft_{draft_model.split('/')[-1]}"
    if k_online:
        output_file += f"_k_online_off{k_online_offset}_w{k_online_warmup}"
    output_file += f"_{run_tag}_{block_verify_tag}_{tree_verify_tag}_topk{tree_verify_topk}.md"

    output_path = f"/root/sglang_local/{output_file}"

    # Construct arguments for the generic sweep script

    max_concurrency = 32
    # DFLASH b16 * max_concurrency(5) => 80
    # piecewise_cuda_graph_max_tokens = 10 * max_concurrency

    args = [
        "bench_dflash_sweep.py",
        "--data-names", data_name,
        "--target-model", target_model,
        "--tp-sizes", "1",
        "--concurrencies", "32",
        "--output-md", output_path,
        "--max-running-requests", str(max_concurrency),
        "--attention-backends", "fa3",
        "--mem-fraction-static", "0.9",
        # "--enable-piecewise-cuda-graph",
        # "--piecewise-cuda-graph-max-tokens",
        # str(piecewise_cuda_graph_max_tokens)
        # "--samples-per-concurrency-base", "256",
        "--disable-radix-cache",
        "--speculative-eagle-topk", str(tree_verify_topk),
        "--speculative-dflash-block-size", str(speculative_dflash_block_size),
    ]
    if tree_verify_num_draft_tokens is not None:
        args.append("--speculative-num-draft-tokens")
        args.append(str(tree_verify_num_draft_tokens))

    if skip_baseline:
        args.append("--skip-baseline")

    if draft_model:
        args.extend(["--draft-model", draft_model])

    if disable_cuda_graph:
        args.append("--disable-cuda-graph")

    print(
        f"Executing with args: {args}, k_online={k_online}, "
        f"tree_verify={tree_verify}, disable_cuda_graph={disable_cuda_graph}"
    )

    # Setup environment
    env = os.environ.copy()
    env["SGLANG_DFLASH_K_ONLINE"] = "1" if k_online else "0"
    env["SGLANG_DFLASH_K_ONLINE_OFFSET"] = str(k_online_offset)
    env["SGLANG_DFLASH_K_ONLINE_WARMUP"] = str(k_online_warmup)
    env["SGLANG_DFLASH_BLOCK_VERIFY"] = "1" if block_verify else "0"
    env["SGLANG_DFLASH_TREE_VERIFY"] = "1" if tree_verify else "0"
    # env["SGLANG_DFLASH_TIMING"] = "1"
    # env["SGLANG_DFLASH_TIMING_LOG_INTERVAL"] = "100"
    # env["CUDA_LAUNCH_BLOCKING"] = "1"
    # env["SGLANG_FA_SPEC_DEBUG"] = "1"
    # env["SGLANG_ATTN_BACKEND_DEBUG"] = "1"
    
    sglang_path = "/root/sglang_local/python"
    if sglang_path not in sys.path:
        sys.path.insert(0, sglang_path)
    
    # Dynamically load the generic script and execute main()
    script_path = "/root/sglang_local/benchmark/dflash/bench_dflash_sweep.py"
    spec = importlib.util.spec_from_file_location("bench_sweep", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_sweep"] = module
    
    # Mock sys.argv
    old_argv = sys.argv
    sys.argv = args
    
    # Change CWD to project root
    old_cwd = os.getcwd()
    os.chdir("/root/sglang_local")
    
    # Update os.environ for the duration of the execution
    old_env = os.environ.copy()
    os.environ.update(env)
    
    try:
        spec.loader.exec_module(module)
        if hasattr(module, "main"):
            module.main()
        else:
            print("Error: main() function not found in script.")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        os.environ.clear()
        os.environ.update(old_env)

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            return f.read()
    
    return f"Error: Output file {output_file} not found."


@app.local_entrypoint()
def main(
    data_names: str = "gsm8k,mt-bench",
    target_model: str = "Qwen/Qwen3-8B",
    draft_model: str = "z-lab/Qwen3-8B-DFlash-b16",
    offset: int = 2,
    warmup: int = 0,
    speculative_dflash_block_size: int = 16,
):
    """\
    Local entrypoint for Modal.

    Runs the same dataset sweep for different tree_verify settings.
    Results are saved with clear tags including tree_verify status.
    """

    combinations = [
        (target_model, draft_model),
    ]

    dataset_list = [d.strip() for d in data_names.split(",") if d.strip()]
    base_results_dir = f"tmp_{data_names.replace(',', '_')}"

    # Ensure output directories exist before writing markdown files.
    os.makedirs("no_cuda_graph_" + base_results_dir, exist_ok=True)
    os.makedirs("cuda_graph_" + base_results_dir, exist_ok=True)

    for disable_cuda_graph in [False]:
        if disable_cuda_graph:
            results_dir = "no_cuda_graph_" + base_results_dir
        else:
            results_dir = "cuda_graph_" + base_results_dir

        for target, draft in combinations:
            skip_baseline = True
            k_online = False
            block_verify = True

            tasks: list[tuple[str, bool, int, str, str]] = []
            calls = []

            for dataset in dataset_list:
                for tree_verify, tree_verify_num_draft_tokens in [(True, None), (False, None)]:
                    print(
                        f"\n>>> Spawning benchmark [{dataset}] "
                        f"tree_verify={tree_verify}, disable_cuda_graph={disable_cuda_graph}: "
                        f"Target={target}, Draft={draft}..."
                    )

                    topks = [4] if tree_verify else [1]
                    for topk in topks:
                        call = run_dataset_sweep.spawn(
                            target,
                            dataset,
                            draft,
                            skip_baseline=skip_baseline,
                            k_online=k_online,
                            k_online_offset=offset,
                            k_online_warmup=warmup,
                            block_verify=block_verify,
                            tree_verify=tree_verify,
                            tree_verify_topk=topk,
                            tree_verify_num_draft_tokens=tree_verify_num_draft_tokens,
                            disable_cuda_graph=disable_cuda_graph,
                            speculative_dflash_block_size=speculative_dflash_block_size,
                        )

                        tasks.append((dataset, tree_verify, topk, target, draft))
                        calls.append(call)

            for (dataset, tree_verify, topk, target, draft), call in zip(tasks, calls):
                try:
                    res_content = call.get()

                    run_tag = "no_k_online"
                    block_verify_tag = "block_verify"
                    tree_verify_tag = "tree_verify" if tree_verify else "no_tree_verify"

                    filename = f"res_{dataset.replace(',', '-')}_{target.split('/')[-1]}"
                    if draft:
                        filename += f"_vs_{draft.split('/')[-1]}"
                    else:
                        filename += "_baseline"

                    filename += f"_{run_tag}_{block_verify_tag}_{tree_verify_tag}_topk{topk}.md"

                    local_path = os.path.join(results_dir, filename)

                    with open(local_path, "w") as f:
                        f.write(res_content)

                    print(f">>> Completed: {filename}")
                except Exception as e:
                    print(
                        f">>> Failed: dataset={dataset}, "
                        f"Target={target}, Draft={draft}, tree_verify={tree_verify}. Error: {e}"
                    )

    print(f"\nAll benchmarks for '{data_names}' finished.")