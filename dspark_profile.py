"""vLLM-native profiler trace, CUDA GRAPH ON, of DFlash/DSpark spec decode.

Uses vLLM's own profiler (VLLM_TORCH_PROFILER_DIR + LLM.start_profile/stop_profile).
To bucket draft-propose vs target-verify GPU time we wrap the speculator's propose()
in a torch record_function("DRAFT_PROPOSE") span (cudagraph stays ON — we only add a
profiler annotation around the call; we don't break the graph or use eager).

Writes the chrome/perfetto trace to VLLM_TORCH_PROFILER_DIR and a json breakdown:
GPU us under DRAFT_PROPOSE (draft) vs the rest of the decode step (verify+overhead),
split by kernel->launcher correlation id.
"""
import argparse, glob, gzip, json, os, time


def patch_propose():
    from torch.profiler import record_function
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    for cls in (DFlashSpeculator, DSparkSpeculator):
        if "propose" in cls.__dict__:  # patch only the class that defines propose
            orig = cls.propose

            def make(orig):
                def wrapped(self, *a, **k):
                    with record_function("DRAFT_PROPOSE"):
                        return orig(self, *a, **k)
                return wrapped
            cls.propose = make(orig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--num-spec", type=int, default=7)
    ap.add_argument("--conc", type=int, required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.trace_dir, exist_ok=True)
    patch_propose()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.target)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    base = [tok.apply_chat_template([{"role": "user", "content": ds[i]["problem"]}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False) for i in range(min(64, len(ds)))]
    prompts = [base[i % len(base)] for i in range(args.conc)]

    spec = {"method": args.method, "num_speculative_tokens": args.num_spec,
            "model": args.draft, "attention_backend": "FLASH_ATTN"}
    # CUDA GRAPH ON: enforce_eager NOT passed. Enable vLLM's native torch profiler
    # via profiler_config (this vLLM deprecated VLLM_TORCH_PROFILER_DIR).
    llm = LLM(model=args.target, speculative_config=spec, gpu_memory_utilization=0.85,
              max_model_len=4096, enable_prefix_caching=False,
              max_num_seqs=args.conc, trust_remote_code=True,
              profiler_config={"profiler": "torch", "torch_profiler_dir": args.trace_dir})

    # warmup to steady decode (fills batch, warms cudagraph) - NOT profiled
    llm.generate(prompts, SamplingParams(temperature=0, max_tokens=64), use_tqdm=False)

    # short profiling window: `steps` decode steps over the saturated batch
    llm.start_profile()
    llm.generate(prompts, SamplingParams(temperature=0, max_tokens=args.steps),
                 use_tqdm=False)
    llm.stop_profile()
    time.sleep(8)  # profiler flushes async

    traces = sorted(glob.glob(os.path.join(args.trace_dir, "*.json*")), key=os.path.getmtime)
    tf = traces[-1] if traces else None
    breakdown = parse_trace(tf) if tf else {"err": "no trace"}
    out = {"method": args.method, "conc": args.conc, "trace_file": tf, **breakdown}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("PROFILE", args.method, "conc", args.conc, breakdown, flush=True)
    print("TRACE", tf, flush=True)
    print("WROTE", args.out, flush=True)


def parse_trace(tf):
    """Split GPU-kernel time by whether the launching CPU op is inside a DRAFT_PROPOSE
    span. Correlate kernel<->launcher via 'External id'/'correlation'. Chrome format."""
    op = gzip.open if tf.endswith(".gz") else open
    with op(tf, "rt") as f:
        data = json.load(f)
    ev = data.get("traceEvents", data) if isinstance(data, dict) else data

    windows = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in ev
                     if e.get("name") == "DRAFT_PROPOSE" and "dur" in e)

    def in_propose(ts):
        for s, e in windows:
            if s <= ts <= e:
                return True
        return False

    def corr(e):
        a = e.get("args", {}) or {}
        return a.get("External id", a.get("correlation"))

    # which correlation ids were launched inside a propose span (CPU/runtime ops)
    propose_corr = set()
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") in (
                "cpu_op", "user_annotation", "cuda_runtime", "runtime", "Runtime"):
            cid = corr(e)
            if cid is not None and in_propose(e["ts"]):
                propose_corr.add(cid)

    draft_us = verify_us = total = 0.0
    nk = 0
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel", "gpu_memcpy"):
            d = e.get("dur", 0); total += d; nk += 1
            if corr(e) in propose_corr:
                draft_us += d
            else:
                verify_us += d
    n_spans = len(windows)
    per = (lambda x: round(x / n_spans, 1)) if n_spans else (lambda x: None)
    return {"n_propose_spans": n_spans, "n_kernels": nk,
            "draft_gpu_us_total": round(draft_us, 1), "verify_gpu_us_total": round(verify_us, 1),
            "total_gpu_us": round(total, 1),
            "draft_gpu_us_per_step": per(draft_us), "verify_gpu_us_per_step": per(verify_us)}


if __name__ == "__main__":
    main()
