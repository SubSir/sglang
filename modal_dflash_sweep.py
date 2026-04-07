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
    gpu="B200",
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
    block_verify: bool = False,
    disable_cuda_graph: bool = False,
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
    output_file = f"results_{data_tag}_{target_model.split('/')[-1]}"
    if draft_model:
        output_file += f"_draft_{draft_model.split('/')[-1]}"
    if k_online:
        output_file += f"_k_online_off{k_online_offset}"
    output_file += f"_{run_tag}_{block_verify_tag}.md"
    
    output_path = f"/root/sglang_local/{output_file}"

    # Construct arguments for the generic sweep script

    max_concurrency = 32
    # DFLASH b16 * max_concurrency(5) => 80

    args = [
        "bench_dflash_sweep.py",
        "--data-names", data_name,
        "--target-model", target_model,
        "--tp-sizes", "1",
        "--concurrencies", "32",
        "--output-md", output_path,
        "--max-running-requests", str(max_concurrency),
        # "--samples-per-concurrency-base", "8",
        "--attention-backends", "fa3",
        "--mem-fraction-static", "0.7",
    ]

    if skip_baseline:
        args.append("--skip-baseline")

    if draft_model:
        args.extend(["--draft-model", draft_model])

    if disable_cuda_graph:
        args.append("--disable-cuda-graph")

    print(f"Executing with args: {args}, k_online={k_online}, disable_cuda_graph={disable_cuda_graph}")
    
    # Setup environment
    env = os.environ.copy()
    env["SGLANG_DFLASH_K_ONLINE"] = "1" if k_online else "0"
    env["SGLANG_DFLASH_K_ONLINE_OFFSET"] = str(k_online_offset)
    env["SGLANG_DFLASH_BLOCK_VERIFY"] = "1" if block_verify else "0"
    
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
    offset: int = 3,
):
    """\
    Local entrypoint for Modal.

    Runs the dataset sweep for k_online off vs on. When k_online is off,
    block_verify is enabled; when k_online is on, block_verify is disabled.
    """

    combinations = [
        (target_model, draft_model),
    ]

    dataset_list = [d.strip() for d in data_names.split(",") if d.strip()]
    base_results_dir = f"tmp_{data_names.replace(',', '_')}"

    # Ensure output directories exist before writing markdown files.
    os.makedirs("no_cuda_graph_" + base_results_dir, exist_ok=True)
    os.makedirs("cuda_graph_" + base_results_dir, exist_ok=True)

    for disable_cuda_graph in [True,]:
        if disable_cuda_graph:
            results_dir = "no_cuda_graph_" + base_results_dir
        else:
            results_dir = "cuda_graph_" + base_results_dir

        for target, draft in combinations:
            skip_baseline = True

            tasks: list[tuple[str, bool, str, str]] = []
            calls = []

            for dataset in dataset_list:
                for k_online in (False, True):
                    # No k_online: block_verify on. With k_online: block_verify off.
                    block_verify = not k_online

                    print(
                        f"\n>>> Spawning benchmark [{dataset}] "
                        f"k_online={k_online}, block_verify={block_verify}, "
                        f"disable_cuda_graph={disable_cuda_graph}: "
                        f"Target={target}, Draft={draft}..."
                    )

                    call = run_dataset_sweep.spawn(
                        target,
                        dataset,
                        draft,
                        skip_baseline=skip_baseline,
                        k_online=k_online,
                        k_online_offset=offset,
                        block_verify=block_verify,
                        disable_cuda_graph=disable_cuda_graph,
                    )

                    tasks.append((dataset, k_online, target, draft))
                    calls.append(call)

            for (dataset, k_online, target, draft), call in zip(tasks, calls):
                try:
                    res_content = call.get()

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

    print(f"\nAll benchmarks for '{data_names}' finished.")