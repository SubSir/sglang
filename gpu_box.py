"""Persistent GPU box for same-card fair benchmarking.

Idea: start a long-lived B200 container (CPU keepalive loop), attach an interactive
shell, and run ALL methods on the SAME physical GPU -> perfect fairness (no host /
card variance between methods). One box per dataset = different cards per dataset,
but within a box every method shares one card.

Usage:
  modal run --detach gpu_box.py::box --label gsm8k        # start a box (repeat per dataset)
  modal container list                                     # find the container id
  modal container exec <id> --pty /bin/bash               # "ssh" in
  # inside: read /root/ONBOX.md, then `claude` to drive the runs (or run by hand)

The box has the sglang env (chain / ours-fused / ddtree) ready. JetSpec uses a
separate venv that the on-box agent sets up (instructions in ONBOX.md).
Results -> /results (modal volume `dflash-v2-results`, `modal volume get` to retrieve).
"""
import modal

app = modal.App("gpu-box")
vol = modal.Volume.from_name("dflash-v2-results", create_if_missing=True)
WT = "/Users/subsir/Desktop/Studio/Python/sglang-v2-tree"

# sglang cu12 nightly + our v2 tree-verify overlay + dev tools + node/claude-code.
box_image = (
    modal.Image.from_registry("lmsysorg/sglang:nightly-dev-cu12-20260627-13b5bd96")
    .apt_install("git", "curl", "build-essential", "tmux", "vim", "wget", "libnuma-dev")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs",
        "npm install -g @anthropic-ai/claude-code || true",
    )
    .pip_install("datasets", "accelerate")
    # overlay our v2 tree-verify port + the bench harness
    .add_local_dir(f"{WT}/python/sglang/srt/speculative", "/tmp/port_speculative", copy=True)
    .add_local_dir(f"{WT}/python/sglang/srt/arg_groups", "/tmp/port_arg_groups", copy=True)
    .add_local_dir("./benchmark", "/root/benchmark", copy=True)
    .add_local_file("./patch_cudagraph.py", "/tmp/patch_cudagraph.py", copy=True)
    .add_local_file("./ONBOX.md", "/root/ONBOX.md", copy=True)
    # reference launchers (the on-box agent reads these for exact recipes/flags)
    .add_local_file("./modal_jetspec.py", "/root/ref_modal_jetspec.py", copy=True)
    .add_local_file("./modal_v2_tree.py", "/root/ref_modal_v2_tree.py", copy=True)
    .run_commands(
        "SGLANG_DIR=$(python3 -c 'import sglang,os;print(os.path.dirname(sglang.__file__))') && "
        "cp -rf /tmp/port_speculative/. \"$SGLANG_DIR/srt/speculative/\" && "
        "cp -rf /tmp/port_arg_groups/. \"$SGLANG_DIR/srt/arg_groups/\" && echo overlaid v2 port",
        "python3 /tmp/patch_cudagraph.py || true",
    )
)


@app.function(gpu="B200", timeout=12 * 3600, image=box_image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def box(label: str = "box", hours: float = 8.0):
    """Keepalive loop. Attach with `modal container exec <id> --pty /bin/bash`."""
    import subprocess, os
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    print(f"=== GPU BOX '{label}' UP ===", flush=True)
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout, flush=True)
    print("ATTACH:  modal container list   then   modal container exec <id> --pty /bin/bash", flush=True)
    print("PLAYBOOK: /root/ONBOX.md   |   RESULTS: /results (commit with the volume)", flush=True)
    # busy-but-cheap keepalive so the container is never reclaimed as idle.
    import time
    n = int(hours * 3600 / 30)
    for i in range(n):
        time.sleep(30)
    print("box timed out; exiting", flush=True)


@app.local_entrypoint()
def up(datasets: str = "gsm8k,math500,mt-bench", hours: float = 8.0):
    """Start one box PER dataset (different cards). Each box = one dataset, all methods
    on its single GPU. Spawns detached; prints attach instructions."""
    hs = []
    for ds in [d.strip() for d in datasets.split(",")]:
        hs.append((ds, box.spawn(ds, hours)))
    print(f"started {len(hs)} boxes (one per dataset). Now run:")
    print("  modal container list")
    print("  modal container exec <id> --pty /bin/bash   # one per box")
    print("  inside each: cat /root/ONBOX.md ; claude")
    for ds, h in hs:
        h.get()  # block so the app + boxes stay alive
