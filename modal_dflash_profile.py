import os
import tarfile
import tempfile
import subprocess
import time
import threading
from collections import deque
from pathlib import Path

import modal

app = modal.App("sglang-dflash-profile")

# Keep the same environment style as modal_dflash_sweep.py
base_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "libnuma-dev")
)

local_image = (
    base_image.run_commands(
        "echo 43 > /tmp/build_time",
        "git clone https://github.com/SubSir/sglang.git /root/sglang_local",
        "cd /root/sglang_local && pip install -e \"python\"",
    )
)


@app.function(
    gpu="H200",
    timeout=7200,
    image=local_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    cloud="aws",
)
def run_profile_once(
    target_model: str = "openai/gpt-oss-20b",
    draft_model: str = "z-lab/gpt-oss-20b-DFlash",
    host: str = "127.0.0.1",
    port: int = 30000,
    num_prompts: int = 64,
    random_input_len: int = 64,
    random_output_len: int = 256,
    profile_dir: str = "/root/sglang/profile_log",
    tree_verify: bool = False,
    disable_cuda_graph: bool = True,
) -> dict:
    """
    Start sglang server with DFLASH config aligned with modal_dflash_sweep.py,
    send one profiled bench_serving request, and return profile artifacts as a tar.gz blob.
    """
    os.makedirs(profile_dir, exist_ok=True)

    env = os.environ.copy()
    env["SGLANG_TORCH_PROFILER_DIR"] = profile_dir
    env["SGLANG_DFLASH_BLOCK_VERIFY"] = "1"
    env["SGLANG_DFLASH_TREE_VERIFY"] = "1" if tree_verify else "0"

    server_cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        target_model,
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        draft_model,
        "--host",
        host,
        "--port",
        str(port),
        "--tp-size",
        "1",
        "--attention-backend",
        "fa3",
        "--max-running-requests",
        "32",
        "--mem-fraction-static", "0.7",
    ]
    if disable_cuda_graph:
        server_cmd.append("--disable-cuda-graph")
    else:
        server_cmd.extend([
            "--enable-piecewise-cuda-graph",
            "--piecewise-cuda-graph-max-tokens",
            str(16 * 32),
        ])

    bench_cmd = [
        "python",
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--model",
        target_model,
        "--host",
        host,
        "--port",
        str(port),
        "--dataset-name",
        "random",
        "--num-prompts",
        str(num_prompts),
        "--random-input-len",
        str(random_input_len),
        "--random-output-len",
        str(random_output_len),
        "--profile",
    ]

    server_proc = None
    server_log_tail = deque(maxlen=400)

    def _stream_server_logs():
        if server_proc is None or server_proc.stdout is None:
            return
        for line in iter(server_proc.stdout.readline, ""):
            print(f"[server] {line}", end="")
            server_log_tail.append(line.rstrip("\n"))

    try:
        server_proc = subprocess.Popen(
            server_cmd,
            cwd="/root/sglang_local",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        log_thread = threading.Thread(target=_stream_server_logs, daemon=True)
        log_thread.start()

        # Wait for server readiness by polling /get_model_info
        import requests

        base_url = f"http://{host}:{port}"
        ready = False
        last_err = None
        for _ in range(180):
            if server_proc.poll() is not None:
                break
            try:
                r = requests.get(base_url + "/get_model_info", timeout=2)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception as e:  # noqa: BLE001
                last_err = e
            time.sleep(2)

        if not ready:
            server_logs = "\n".join(server_log_tail)
            raise RuntimeError(
                f"Server not ready. last_err={last_err}. logs:\n{server_logs[-8000:]}"
            )

        print("[bench] Starting bench_serving...")
        bench_proc = subprocess.Popen(
            bench_cmd,
            cwd="/root/sglang_local",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        bench_log_tail = deque(maxlen=400)
        if bench_proc.stdout is not None:
            for line in iter(bench_proc.stdout.readline, ""):
                print(f"[bench] {line}", end="")
                bench_log_tail.append(line.rstrip("\n"))
        bench_returncode = bench_proc.wait()

        # Stop server first, so torch profiler files are fully flushed to disk.
        server_logs_tail = ""
        if server_proc is not None:
            try:
                server_proc.terminate()
                server_proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                try:
                    server_proc.kill()
                    server_proc.wait(timeout=10)
                except Exception:  # noqa: BLE001
                    pass

            if server_proc.stdout is not None:
                try:
                    server_logs_tail = server_proc.stdout.read()[-8000:]
                except Exception:  # noqa: BLE001
                    pass

        # Give profiler writers a short grace period for file finalization.
        time.sleep(3)

        profile_path = Path(profile_dir)
        prof_files = sorted(profile_path.rglob("*"))
        existing_files = [str(p) for p in prof_files if p.is_file()]

        bench_logs = "\n".join(bench_log_tail)
        if bench_returncode != 0:
            raise RuntimeError(
                "bench_serving failed:\n"
                f"combined_logs:\n{bench_logs}\n\n"
                f"server_logs_tail:\n{server_logs_tail}\n\n"
                f"profile_files={existing_files}"
            )

        # Package profiler files for download after server has exited.
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmpf:
            tar_path = tmpf.name

        with tarfile.open(tar_path, "w:gz") as tar:
            if profile_path.exists():
                tar.add(profile_path, arcname="profile_log")

        with open(tar_path, "rb") as f:
            tar_bytes = f.read()

        return {
            "profile_tar_name": "profile_log.tar.gz",
            "profile_file_count": len(existing_files),
            "profile_files": existing_files,
            "bench_logs": bench_logs,
            "server_logs_tail": server_logs_tail,
            "profile_tar_bytes": tar_bytes,
        }

    finally:
        if server_proc is not None and server_proc.poll() is None:
            try:
                server_proc.terminate()
                server_proc.wait(timeout=20)
            except Exception:  # noqa: BLE001
                try:
                    server_proc.kill()
                except Exception:  # noqa: BLE001
                    pass


@app.local_entrypoint()
def main(
    target_model: str = "openai/gpt-oss-120b",
    draft_model: str = "z-lab/gpt-oss-120b-DFlash",
    output_dir: str = "profile_logs",
    disable_cuda_graph: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    combinations = [True, False]

    for tree_verify in combinations:
        tree_verify_tag = "tree_verify" if tree_verify else "no_tree_verify"
        output_tar = os.path.join(
            output_dir,
            f"profile_{target_model.split('/')[-1]}_vs_{draft_model.split('/')[-1]}"
            f"_{tree_verify_tag}.tar.gz",
        )

        print(
            f"\n>>> Starting profile: tree_verify={tree_verify}, "
            f"target={target_model}, draft={draft_model}"
        )

        ret = run_profile_once.remote(
            target_model=target_model,
            draft_model=draft_model,
            tree_verify=tree_verify,
            disable_cuda_graph=disable_cuda_graph,
        )

        with open(output_tar, "wb") as f:
            f.write(ret["profile_tar_bytes"])

        print(f"Saved profiler tarball to: {output_tar}")
        print(f"Profiler file count: {ret['profile_file_count']}")
        for p in ret["profile_files"]:
            print(" -", p)

    print("\nAll profile runs finished.")
