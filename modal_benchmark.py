import os
import subprocess
import time
import modal

app = modal.App("sglang-dflash-benchmark")

# 基础镜像
base_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "libnuma-dev")
)

# 1. 官方 DFlash 分支镜像
official_image = (
    base_image
    .run_commands(
        "git clone -b dflash https://github.com/modal-labs/sglang.git /root/sglang_official",
        "cd /root/sglang_official && pip install -e \"python\"",
    )
    # 把本地的 test.py 拷到容器里
    .add_local_file(
        local_path="./test.py",
        remote_path="/root/test.py",
        copy=True,
    )
)

# 2. 本地修改版镜像
local_image = (
    base_image
    .add_local_dir(
        local_path=".",                     # 当前 repo
        remote_path="/root/sglang_local",   # 容器中的路径
        copy=True,                          # 直接复制进镜像，允许后续 run_commands 使用
    )
    .run_commands(
        "cd /root/sglang_local && pip install -e \"python\"",
    )
    .add_local_file(
        local_path="./test.py",
        remote_path="/root/test.py",
        copy=True,
    )
)


def run_test_script(model_path: str, draft_model_path: str, output_file: str) -> str:
    """直接调用修改后的 test.py，由其内部处理 launch/warmup/bench/kill"""
    import subprocess
    import os

    test_cmd = [
        "python3", "/root/test.py",
        "--model-path", model_path,
        "--speculative-draft-model-path", draft_model_path,
        "--speculative-algorithm", "DFLASH",
        "--tp-size", "4",
        "--trust-remote-code",
        "--output", output_file
    ]

    print(f"Running test script: {' '.join(test_cmd)}")
    test_result = subprocess.run(
        test_cmd,
        capture_output=True,
        text=True,
    )

    print("STDOUT:", test_result.stdout)
    print("STDERR:", test_result.stderr)

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            return f.read()
    
    return f"Error: Output file {output_file} not found. \nSTDOUT: {test_result.stdout}\nSTDERR: {test_result.stderr}"


@app.function(
    gpu="A100-40GB:4",
    timeout=3600,
    image=official_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_official():
    print("=== Running Official DFlash Branch ===")
    return run_test_script(
        "Qwen/Qwen3-8B",
        "z-lab/Qwen3-8B-DFlash-b16",
        "official_results.jsonl"
    )


@app.function(
    gpu="A100-40GB:4",
    timeout=3600,
    image=local_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_local():
    print("=== Running Local Modified SGLang ===")
    return run_test_script(
        "Qwen/Qwen3-8B",
        "z-lab/Qwen3-8B-DFlash-b16",
        "local_results.jsonl"
    )


@app.local_entrypoint()
def main():
    print(">>> Running OFFICIAL benchmark...")
    official_res = test_official.remote()

    print(">>> OFFICIAL benchmark done. Running LOCAL benchmark...")
    local_res = test_local.remote()

    # 把结果保存到本地文件
    with open("official_results.jsonl", "w") as f:
        f.write(official_res)

    with open("local_results.jsonl", "w") as f:
        f.write(local_res)

    print("\n" + "=" * 20 + " FINAL COMPARISON " + "=" * 20)
    print("OFFICIAL RESULTS (JSONL):\n", official_res)
    print("\nLOCAL RESULTS (JSONL):\n", local_res)