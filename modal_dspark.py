"""DFlash vs DSpark on vLLM (B200), SAME-CARD A/B, concurrency sweep.

DSpark = vLLM PR #46995 (benchislett/vllm:dspark): DFlash backbone + a sequential
Markov draft head. Both draft heads share arch Qwen3DSparkModel; the dflash ckpt
has markov_rank=0 (pure block-parallel = vLLM method "dflash"), the dspark ckpt
markov_rank=256 (method "dspark", sequential head). Target = Qwen/Qwen3-8B.

Overlay install (same as modal_vllm_jetspec.py): stock vLLM nightly for the
compiled _C.so, then overlay the PR's pure-python vllm/ on top (PR adds NO .cu).

  modal run modal_dspark.py::smoke        # quick: both methods load + 1 gen, conc=1
  modal run modal_dspark.py::ab           # full A/B: dflash+dspark, conc sweep, per dataset
  modal run modal_dspark.py::pull
"""
import os, modal

app = modal.App("dspark-bench")
vol = modal.Volume.from_name("dspark-results", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "curl")
    .run_commands(
        "echo dspark4 > /tmp/bt",
        # 1) install vllm nightly -> torch + all runtime deps + working compiled _C.
        # 2) editable-install the PR's FULL python tree with --no-deps (keep nightly's
        #    torch/deps, no cuda13/torch2.11 reinstall) + --no-build-isolation +
        #    VLLM_USE_PRECOMPILED (don't compile). This replaces the package's python
        #    with the PR's CONSISTENT tree (speculator<->scheduler<->sampler in
        #    lockstep), unlike a bare *.py overlay on latest nightly which produced
        #    -1 draft tokens + CUDA illegal memory at batch>1.
        "pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly || pip install -U vllm",
        "pip install datasets transformers",
        # vLLM editable build deps (needed because we use --no-build-isolation to
        # keep the nightly torch/_C and avoid a cuda13/torch2.11 reinstall).
        "pip install setuptools_scm setuptools-rust 'setuptools>=77' wheel ninja cmake packaging",
        "git clone -b dspark https://github.com/benchislett/vllm /root/vllm-dspark",
        "cd /root/vllm-dspark && git remote add upstream https://github.com/vllm-project/vllm && "
        "git fetch --depth 300 upstream main || true",
        "cd /root/vllm-dspark && VLLM_USE_PRECOMPILED=1 "
        "pip install -e . --no-build-isolation --no-deps",
    )
)

DFLASH = "deepseek-ai/dflash_qwen3_8b_block7"
DSPARK = "deepseek-ai/dspark_qwen3_8b_block7"


def _bench_src():
    # read locally (in the local_entrypoint) so edits are ALWAYS fresh; passed to
    # the remote fn as an arg and written in-container (avoids image/mount caching).
    return open(__file__.replace("modal_dspark.py", "dspark_bench.py")).read()


def _run(method, draft, dataset, concs, n, max_tokens, tag, bench_src):
    import subprocess
    with open("/root/dspark_bench.py", "w") as f:
        f.write(bench_src)
    env = dict(os.environ)
    env["VLLM_USE_V1"] = "1"
    out_json = f"/results/{tag}.json"
    args = ["python", "/root/dspark_bench.py", "--method", method, "--draft", draft,
            "--target", "Qwen/Qwen3-8B", "--dataset", dataset, "--num-spec", "7",
            "--concs", concs, "--n", str(n), "--max-tokens", str(max_tokens),
            "--out", out_json]
    print(">>>", " ".join(args), flush=True)
    # stream stdout live (tee to log) so per-conc ">>> ... C=N:" lines show in app
    # logs as they happen (capture-only hid all progress).
    logf = open(f"/results/{tag}.log", "w")
    p = subprocess.Popen(args, cwd="/root", env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
        logf.write(line)
    p.wait()
    logf.close()
    vol.commit()
    return p.returncode


@app.function(gpu="B200", timeout=14400, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def ab(dataset, concs, n, max_tokens, bench_src):
    """SAME-CARD A/B for ONE dataset: dflash-baseline then dspark, both conc-swept,
    one container. BOTH use vLLM method "dspark" (the only one that loads the
    Qwen3DSparkModel-arch deepseek checkpoints); the DFlash baseline is the
    dflash ckpt with markov_rank=0 (Markov head is a no-op = block-parallel)."""
    rc_d = _run("dspark", DFLASH, dataset, concs, n, max_tokens, f"dflash_{dataset}", bench_src)
    rc_s = _run("dspark", DSPARK, dataset, concs, n, max_tokens, f"dspark_{dataset}", bench_src)
    return {"dataset": dataset, "dflash_rc": rc_d, "dspark_rc": rc_s}


@app.function(gpu="B200", timeout=14400, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def profile(prof_src):
    """vLLM-native profiler traces, CUDA GRAPH ON: dflash+dspark at conc 1 and 32
    (4 traces). Buckets GPU time into draft-propose vs target-verify via a
    record_function span around propose(). Both ckpts load via method=dspark."""
    import subprocess
    with open("/root/dspark_profile.py", "w") as f:
        f.write(prof_src)
    env = dict(os.environ); env["VLLM_USE_V1"] = "1"
    out = {}
    for label, draft in [("dflash", DFLASH), ("dspark", DSPARK)]:
        for conc in (1, 32):
            tag = f"{label}_c{conc}"
            tdir = f"/results/trace_{tag}"
            a = ["python", "/root/dspark_profile.py", "--method", "dspark",
                 "--draft", draft, "--num-spec", "7", "--conc", str(conc),
                 "--steps", "8", "--trace-dir", tdir,
                 "--out", f"/results/prof_{tag}.json"]
            print(">>>", " ".join(a), flush=True)
            # stream so progress is visible
            logf = open(f"/results/prof_{tag}.log", "w")
            p = subprocess.Popen(a, cwd="/root", env=env, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
            for line in p.stdout:
                print(line, end="", flush=True); logf.write(line)
            p.wait(); logf.close()
            out[tag] = p.returncode
            vol.commit()
    return out


@app.function(gpu="B200", timeout=21600, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def sweep(dataset, concs, max_tokens, bench_src):
    """SAME-CARD A/B with PROPER sampling: n = max(1024, conc*32) PER conc. One
    container per dataset; for each conc runs dflash then dspark back-to-back with
    its own large n, writing per-conc jsons that the table merges."""
    for c in [int(x) for x in concs.split(",")]:
        n = max(1024, c * 32)  # FULL sampling at every conc; cudagraph ON (bench enforce_eager=False)
        _run("dspark", DFLASH, dataset, str(c), n, max_tokens, f"sw_dflash_{dataset}_c{c}", bench_src)
        _run("dspark", DSPARK, dataset, str(c), n, max_tokens, f"sw_dspark_{dataset}_c{c}", bench_src)
    return {"dataset": dataset, "done": True}


@app.local_entrypoint()
def run_profile():
    src = open(__file__.replace("modal_dspark.py", "dspark_profile.py")).read()
    print(profile.remote(src))


@app.local_entrypoint()
def run_sweep(datasets="math500,mt-bench,gsm8k", concs="1,8,32,64", max_tokens: int = 512):
    src = _bench_src()
    hs = [(ds, sweep.spawn(ds, concs, max_tokens, src)) for ds in datasets.split(",")]
    for ds, h in hs:
        print(f"=== {ds} ==="); print(h.get())


_PARALLEL_PATCH = r'''
import re, pathlib
f = pathlib.Path("/root/vllm-dspark/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py")
s = f.read_text()
# 1) read markov_rank + env flag in __init__ (anchor: end of _anchor_idx assignment)
a1 = """        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )"""
ins1 = a1 + """

        import os as _os
        self._force_parallel_markov0 = (
            _os.environ.get("DSPARK_FORCE_PARALLEL_MARKOV0") == "1"
            and int(getattr(self.draft_model_config.hf_config, "markov_rank", 0) or 0) == 0
        )"""
assert a1 in s, "anchor1 (_anchor_idx) not found"
s = s.replace(a1, ins1, 1)
# 2) fast parallel path at top of _sample_sequential (skip the loop entirely)
a2 = """        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_logits(sample_hidden)"""
ins2 = """        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        if getattr(self, "_force_parallel_markov0", False):
            draft_tokens = self.sample_draft(
                sample_hidden, self.sample_pos[:num_sample],
                self.sample_idx_mapping[:num_sample], self.temperature, self.seeds,
                self.sample_col[:num_sample], self.draft_logits)
            self.draft_tokens[:num_reqs] = draft_tokens.view(num_reqs, n_spec)
            return
        base_logits = self.model.compute_logits(sample_hidden)"""
assert a2 in s, "anchor2 (_sample_sequential head) not found"
s = s.replace(a2, ins2, 1)
f.write_text(s)
print("PATCHED dspark speculator: parallel markov0 fast path installed", flush=True)
'''


@app.function(gpu="B200", timeout=3600, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def dflash_parallel_smoke(bench_src):
    """Patch dspark speculator -> parallel block-parallel path when markov_rank=0,
    then smoke-test the dflash(markov0) ckpt with the fast path ON. accept must stay
    ~the sequential markov0 value (5-6) -> fast path is correct (just no loop)."""
    import subprocess, os as o
    r = subprocess.run(["python", "-c", _PARALLEL_PATCH], capture_output=True, text=True)
    print("PATCH:", r.stdout, r.stderr[-1500:], flush=True)
    if "PATCHED" not in r.stdout:
        return {"error": "patch failed", "stderr": r.stderr[-1500:]}
    o.environ["DSPARK_FORCE_PARALLEL_MARKOV0"] = "1"
    rc = _run("dspark", DFLASH, "math500", "1", 16, 256, "smoke_dflash_parallel", bench_src)
    return {"rc": rc}


@app.local_entrypoint()
def run_dflash_parallel_smoke():
    print(dflash_parallel_smoke.remote(_bench_src()))


@app.function(gpu="B200", timeout=21600, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def sweep_parallel(dataset, concs, max_tokens, bench_src):
    """Same-card: TRUE block-parallel DFlash (markov0 + parallel patch) vs DSpark,
    per conc, n=max(1024,conc*32). Patch makes markov0 skip the sequential loop;
    dspark(markov256) ignores the env flag, stays sequential."""
    import subprocess, os as o
    r = subprocess.run(["python", "-c", _PARALLEL_PATCH], capture_output=True, text=True)
    print("PATCH:", r.stdout[-200:], r.stderr[-800:], flush=True)
    assert "PATCHED" in r.stdout, "patch failed"
    o.environ["DSPARK_FORCE_PARALLEL_MARKOV0"] = "1"  # only activates for markov0=dflash
    for c in [int(x) for x in concs.split(",")]:
        n = max(1024, c * 32)
        _run("dspark", DFLASH, dataset, str(c), n, max_tokens, f"swp_dflashpar_{dataset}_c{c}", bench_src)
        _run("dspark", DSPARK, dataset, str(c), n, max_tokens, f"swp_dspark_{dataset}_c{c}", bench_src)
    return {"dataset": dataset, "done": True}


@app.local_entrypoint()
def run_sweep_parallel(datasets="math500,mt-bench,gsm8k", concs="1,8,32,64", max_tokens: int = 512):
    src = _bench_src()
    hs = [(ds, sweep_parallel.spawn(ds, concs, max_tokens, src)) for ds in datasets.split(",")]
    for ds, h in hs:
        print(f"=== {ds} ==="); print(h.get())


@app.function(gpu="B200", timeout=3600, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def profile_parallel(prof_src):
    """vLLM-native profiler, CUDA GRAPH ON, with the parallel-markov0 patch:
    TRUE block-parallel DFlash vs DSpark at conc 1 and 32. Draft-propose vs verify
    GPU us + total step, so we see DSpark's per-step draft delta over a REAL parallel
    DFlash (not the sequential-loop markov0)."""
    import subprocess, os as o
    r = subprocess.run(["python", "-c", _PARALLEL_PATCH], capture_output=True, text=True)
    print("PATCH:", r.stdout[-200:], r.stderr[-600:], flush=True)
    assert "PATCHED" in r.stdout, "patch failed"
    with open("/root/dspark_profile.py", "w") as f:
        f.write(prof_src)
    env = dict(o.environ); env["VLLM_USE_V1"] = "1"
    env["DSPARK_FORCE_PARALLEL_MARKOV0"] = "1"  # only activates for markov0=dflash
    out = {}
    for label, draft in [("dflashpar", DFLASH), ("dspark", DSPARK)]:
        for conc in (1, 32):
            tag = f"{label}_c{conc}"
            a = ["python", "/root/dspark_profile.py", "--method", "dspark",
                 "--draft", draft, "--num-spec", "7", "--conc", str(conc),
                 "--steps", "8", "--trace-dir", f"/results/tracep_{tag}",
                 "--out", f"/results/profp_{tag}.json"]
            print(">>>", " ".join(a), flush=True)
            p = subprocess.Popen(a, cwd="/root", env=env, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
            for line in p.stdout:
                print(line, end="", flush=True)
            p.wait(); out[tag] = p.returncode; vol.commit()
    return out


@app.local_entrypoint()
def run_profile_parallel():
    src = open(__file__.replace("modal_dspark.py", "dspark_profile.py")).read()
    print(profile_parallel.remote(src))


@app.function(gpu="B200", timeout=3600, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def dflash_official():
    """Probe: can the deepseek dflash (markov0) weights run under the OFFICIAL vLLM
    DFlash head (method=dflash, real block-parallel, no sequential loop)? The ckpt
    arch is Qwen3DSparkModel + name has 'dflash' -> auto-detect mangles to
    DFlashQwen3DSparkModel (unregistered). Force it: download to a name WITHOUT
    'dflash'/'dspark', patch config.architectures -> the registry key the dflash
    head wants, pass method=dflash explicitly. Report load + accept."""
    import json, traceback
    from huggingface_hub import snapshot_download
    local = snapshot_download(DFLASH, local_dir="/root/dfl_ckpt")
    cfg_path = "/root/dfl_ckpt/config.json"
    orig = json.load(open(cfg_path))
    print("orig arch:", orig.get("architectures"), "markov_rank:", orig.get("markov_rank"), flush=True)

    # registry maps key "DFlashDraftModel" -> qwen3_dflash.DFlashQwen3ForCausalLM
    attempts = ["DFlashDraftModel", "Qwen3ForCausalLM"]
    out = {}
    for arch in attempts:
        cfg = dict(orig); cfg["architectures"] = [arch]
        json.dump(cfg, open(cfg_path, "w"))
        print(f"\n===== TRY arch={arch}, method=dflash =====", flush=True)
        try:
            from vllm import LLM, SamplingParams
            llm = LLM(model="Qwen/Qwen3-8B",
                      speculative_config={"method": "dflash", "model": "/root/dfl_ckpt",
                                          "num_speculative_tokens": 7,
                                          "attention_backend": "FLASH_ATTN"},
                      gpu_memory_utilization=0.85, max_model_len=4096,
                      enforce_eager=False, disable_log_stats=False,
                      max_num_seqs=8, trust_remote_code=True)
            r = llm.generate(["What is 17*23? Answer:"],
                             SamplingParams(temperature=0, max_tokens=64), use_tqdm=False)
            txt = r[0].outputs[0].text
            mets = {x.name: getattr(x, "value", getattr(x, "sum", None))
                    for x in llm.llm_engine.get_metrics()}
            acc = mets.get("vllm:spec_decode_num_accepted_tokens", 0)
            nd = mets.get("vllm:spec_decode_num_drafts", 0)
            al = (acc / nd + 1) if nd else None
            out[arch] = {"ok": True, "accept_len": al, "text": txt[:120]}
            print(f"LOADED. accept_len={al}  out={txt[:120]!r}", flush=True)
            del llm
            break  # first that works wins
        except Exception as e:
            out[arch] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    json.dump(out, open("/results/dflash_official_probe.json", "w")); vol.commit()
    return out


@app.local_entrypoint()
def run_dflash_official():
    print(dflash_official.remote())


@app.function(gpu="B200", timeout=3600, image=image,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={"/results": vol})
def smoke(bench_src, concs="1,8", n=8, max_tokens=256):
    """Quick: confirm both methods load + run on gsm8k (fast fail)."""
    import subprocess
    # diagnose _C location + whether the C ext actually loads on GPU
    diag = r"""
import vllm, os, glob
site = os.path.dirname(vllm.__file__)
print('vllm', vllm.__version__, 'at', site)
print('_C.so files:', glob.glob(os.path.join(site, '_C*')) + glob.glob(os.path.join(site, '**/_C*'), recursive=True)[:5])
try:
    import vllm._C; print('vllm._C OK')
except Exception as e:
    print('vllm._C FAIL:', type(e).__name__, e)
try:
    from vllm import _custom_ops; print('vllm._custom_ops OK')
except Exception as e:
    print('vllm._custom_ops FAIL:', type(e).__name__, e)
"""
    r = subprocess.run(["python", "-c", diag], capture_output=True, text=True)
    print("DIAG:", r.stdout, r.stderr[-800:], flush=True)
    # Both deepseek checkpoints declare arch Qwen3DSparkModel -> both load via
    # method=dspark. dflash ckpt has markov_rank=0 (Markov head is a no-op =
    # block-parallel baseline); dspark ckpt has markov_rank=256 (real seq head).
    rc_d = _run("dspark", DFLASH, "gsm8k", concs, n, max_tokens, "smoke_dflash", bench_src)
    rc_s = _run("dspark", DSPARK, "gsm8k", concs, n, max_tokens, "smoke_dspark", bench_src)
    return {"dflash_rc": rc_d, "dspark_rc": rc_s}


@app.local_entrypoint()
def run_smoke(concs="1,8", n: int = 8, max_tokens: int = 256):
    print(smoke.remote(_bench_src(), concs, n, max_tokens))


@app.local_entrypoint()
def run_ab(datasets="gsm8k,math500,mt-bench", concs="1,8,32,64", n: int = 80,
           max_tokens: int = 512):
    """Launch one detached container per dataset (each = its own card; within a
    dataset dflash & dspark share the card)."""
    src = _bench_src()
    handles = [(ds, ab.spawn(ds, concs, n, max_tokens, src)) for ds in datasets.split(",")]
    for ds, h in handles:
        print(f"=== {ds} ==="); print(h.get())


@app.function(image=image, volumes={"/results": vol})
def _list():
    import os as o
    return {fn: open(f"/results/{fn}").read()
            for fn in sorted(o.listdir("/results")) if o.path.isfile(f"/results/{fn}")}


@app.function(image=image, volumes={"/results": vol}, timeout=1800)
def _traces():
    """Return {tag: (filename, bytes)} for the newest trace file in each trace_*/ dir."""
    import os as o, glob
    out = {}
    for tag in ("dflash_c1", "dflash_c32", "dspark_c1", "dspark_c32"):
        d = f"/results/trace_{tag}"
        if not o.path.isdir(d):
            continue
        fs = sorted(glob.glob(o.path.join(d, "*.json*")), key=o.path.getmtime)
        if fs:
            out[tag] = (o.path.basename(fs[-1]), open(fs[-1], "rb").read())
    return out


@app.local_entrypoint()
def pull_traces():
    import os
    os.makedirs("dspark_results", exist_ok=True)
    for tag, (fn, b) in _traces.remote().items():
        # keep .gz if gzipped; coordinator asked for trace_{...}.json paths
        ext = ".json.gz" if fn.endswith(".gz") else ".json"
        path = f"dspark_results/trace_{tag}{ext}"
        with open(path, "wb") as f:
            f.write(b)
        print(f"{os.path.abspath(path)}  ({len(b)/1e6:.1f} MB, src {fn})")


@app.local_entrypoint()
def pull():
    os.makedirs("dspark_results", exist_ok=True)
    for fn, c in _list.remote().items():
        with open(f"dspark_results/{fn}", "w") as f:
            f.write(c)
        print(f"\n===== {fn} =====\n" + "\n".join(c.splitlines()[-25:]))
