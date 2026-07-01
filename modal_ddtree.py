"""Benchmark DDTree (liranringel/ddtree, adaptive best-first draft tree) on B200.

DDTree is standalone PyTorch+HF and uses the SAME z-lab DFlash draft heads as our SGLang
tree, so its `dflash` (chain) + `ddtree_tb{budget}` (adaptive tree) numbers are directly
comparable to our EAGLE-topk tree. benchmark.py reports per-method time_per_output_token +
acceptance_lengths; speedup = AR-baseline_tpt / method_tpt. Single GPU (no RANK => no torchrun).

  modal run modal_ddtree.py::bench       # gsm8k + mt-bench, budget sweep
  modal run modal_ddtree.py::pull
"""

import os
import modal

app = modal.App("ddtree-bench")
vol = modal.Volume.from_name("ddtree-results", create_if_missing=True)

image = (
    modal.Image.from_registry("lmsysorg/sglang:nightly-dev-cu12-20260627-13b5bd96")
    .run_commands(
        "echo dd2 > /tmp/bt",
        "git clone --depth 1 https://github.com/liranringel/ddtree /root/ddtree",
        # DATA CONTROL: drop shuffle(seed=0) -> first-N (version-robust, matches sglang first-N).
        "sed -i 's/\\.shuffle(seed=0)\\.select/.select/' /root/ddtree/benchmark.py",
        "pip install loguru",  # rest (torch/transformers/datasets/flash_attn/ninja) in base
    )
)

# Extract per-method accept length + speedup from the saved .pt, run-side (avoids pulling .pt).
EXTRACT = r"""
import torch, numpy as np, sys, json
run = torch.load(sys.argv[1], weights_only=False)
resps = run["responses"]
keys = [k for k in resps[0].keys()]
def tpt(k): return float(np.mean([r[k].time_per_output_token for r in resps]))
def al(k):  return float(np.mean([np.mean(r[k].acceptance_lengths) for r in resps]))
base = tpt("baseline")
rows = []
for k in keys:
    rows.append((k, al(k), tpt(k), base/tpt(k)))
print(json.dumps({"block_size": run.get("block_size"), "n": len(resps), "rows": rows}))
"""


@app.function(gpu="B200", timeout=10800, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def run(dataset, tree_budgets, max_samples, max_new, tag,
        target="Qwen/Qwen3-8B", draft="z-lab/Qwen3-8B-DFlash-b16"):
    import subprocess
    pt = f"/results/{tag}.pt"
    env = dict(os.environ)  # NO RANK => single-GPU path (distributed.init() skipped)
    cmd = ["python", "benchmark.py",
           "--model-name-or-path", target, "--draft-name-or-path", draft,
           "--dataset", dataset, "--tree-budget", tree_budgets,
           "--max-samples", str(max_samples), "--max-new-tokens", str(max_new),
           "--temperature", "0.0", "--save-path", pt]
    print(">>>", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd="/root/ddtree", env=env, capture_output=True, text=True)
    if not os.path.exists(pt):
        err = "\n".join(p.stderr.splitlines()[-50:])
        with open(f"/results/{tag}.txt", "w") as f:
            f.write("FAILED\n" + err)
        vol.commit()
        print("FAILED\n" + err, flush=True)
        return "FAILED"
    # extract metrics
    with open("/tmp/extract.py", "w") as f:
        f.write(EXTRACT)
    e = subprocess.run(["python", "/tmp/extract.py", pt], cwd="/root/ddtree",
                       env=env, capture_output=True, text=True)
    summary = e.stdout.strip() or e.stderr[-2000:]
    with open(f"/results/{tag}.txt", "w") as f:
        f.write(summary)
    vol.commit()
    print(summary, flush=True)
    return summary


@app.local_entrypoint()
def bench(datasets: str = "gsm8k,mt-bench", tree_budgets: str = "16,32,64,128,256",
          max_samples: int = 32, max_new: int = 1024):
    handles = [run.spawn(ds, tree_budgets, max_samples, max_new, f"qwen3-8b_{ds}")
               for ds in datasets.split(",")]
    print("launched ddtree:", datasets, "— waiting for completion")
    for h in handles:
        print(h.get())


@app.function(image=image, volumes={"/results": vol})
def _list():
    import os as o
    return {fn: open(f"/results/{fn}").read()
            for fn in sorted(o.listdir("/results")) if fn.endswith(".txt")}


@app.local_entrypoint()
def pull():
    os.makedirs("ddtree_results", exist_ok=True)
    for fn, c in _list.remote().items():
        with open(f"ddtree_results/{fn}", "w") as f:
            f.write(c)
        print(f"\n===== {fn} =====\n{c[:3000]}")
