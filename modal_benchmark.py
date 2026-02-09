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
    # 把本地的 test.py 拷到容器里，放在最后，并加 copy=True 以便可以安全在 image 中使用
    .add_local_file(
        local_path="./test.py",
        remote_path="/root/test.py",
        copy=True,
    )
)

# 2. 本地修改版镜像
# 因为你要在镜像 build 过程中用这些文件（pip install -e），必须 copy=True
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


def run_sglang_and_test(model_path: str, draft_model_path: str) -> str:
    """在容器内启动 server 并运行测试脚本"""
    import subprocess
    import time

    launch_cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--speculative-algorithm", "DFLASH",
        "--speculative-draft-model-path", draft_model_path,
        "--tp-size", "4",
        "--dtype", "bfloat16",
        "--mem-fraction-static", "0.6",
        "--trust-remote-code",
        "--port", "30000",
    ]

    print(f"Starting SGLang server: {' '.join(launch_cmd)}")
    server_process = subprocess.Popen(launch_cmd)

    # 等待服务器就绪
    max_retries = 60
    for i in range(max_retries):
        try:
            import requests
            resp = requests.get("http://127.0.0.1:30000/v1/models")
            if resp.status_code == 200:
                print("Server is ready!")
                break
        except Exception:
            pass
        time.sleep(5)
        if i % 10 == 0:
            print(f"Waiting for server... ({i * 5}s)")
    else:
        server_process.kill()
        raise RuntimeError("Server failed to start in time")

    # 运行测试脚本
    print("Running test.py...")
    test_result = subprocess.run(
        ["python3", "/root/test.py"],
        capture_output=True,
        text=True,
    )

    print("STDOUT:", test_result.stdout)
    print("STDERR:", test_result.stderr)

    # 停止服务
    server_process.terminate()
    return test_result.stdout

@app.function(
    gpu="A100-40GB:4",  # 使用 4 块 H200
    timeout=3600,
    image=official_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_official():
    print("=== Running Official DFlash Branch ===")
    return run_sglang_and_test(
        "Qwen/Qwen3-8B",
        "z-lab/Qwen3-8B-DFlash-b16",
    )


@app.function(
    gpu="A100-40GB:4",
    timeout=3600,
    image=local_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_local():
    print("=== Running Local Modified SGLang ===")
    return run_sglang_and_test(
        "Qwen/Qwen3-8B",
        "z-lab/Qwen3-8B-DFlash-b16",
    )


@app.local_entrypoint()
def main():
    # 串行运行：先官方，再本地
    print(">>> Running OFFICIAL benchmark...")
    official_res = test_official.remote()

    print(">>> OFFICIAL benchmark done. Running LOCAL benchmark...")
    local_res = test_local.remote()

    print("\n" + "=" * 20 + " FINAL COMPARISON " + "=" * 20)
    print("OFFICIAL RESULTS:\n", official_res)
    print("\nLOCAL RESULTS:\n", local_res)