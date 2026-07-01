"""In-container vLLM bench: DFlash vs DSpark, same card, concurrency sweep.

Driven by modal_dspark.py. For ONE (method, dataset): load the engine once, then
for each concurrency C submit C prompts at once. Metrics are vLLM's cumulative
SpecDecodingStats counters -> we DELTA them per run.

Reported per (method,dataset,C):
  tok_s (batch decode throughput), per_seq_tok_s, accept_len, step_ms (per decode
  step, = num_decode_steps_delta timed by wall/steps), and raw counters.
step_ms: each spec step the engine runs ONE target verify + ONE draft propose for
the whole batch; num_drafts(delta) counts (req x steps), so steps_per_seq =
num_drafts_delta / C and step_ms = wall_s*1000 / steps_per_seq.
"""

import argparse, json, time


def load_prompts(dataset, n, tok):
    from datasets import load_dataset
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        base = [ds[i]["question"] for i in range(len(ds))]
    elif dataset == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        base = [ds[i]["problem"] for i in range(len(ds))]
    elif dataset == "mt-bench":
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        base = [ds[i]["prompt"][0] for i in range(len(ds))]
    else:
        raise ValueError(dataset)
    # CYCLE base prompts to reach n (keeps the batch saturated when n > dataset size,
    # e.g. conc=64 -> n=2048 on math500's 500 rows / mt-bench's ~80).
    qs = [base[i % len(base)] for i in range(n)]
    return [tok.apply_chat_template([{"role": "user", "content": q}],
                                    tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False) for q in qs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)          # dflash | dspark
    ap.add_argument("--draft", required=True)
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-spec", type=int, default=7)
    ap.add_argument("--concs", default="1,8,32")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.target)
    concs = [int(c) for c in args.concs.split(",")]
    # n prompts submitted per conc, with in-flight concurrency capped at C via
    # max_num_seqs (saturated batch for the whole run). Need single conc per process
    # for that (max_num_seqs is engine-init). Multi-conc calls (smoke) keep the old
    # quick path (submit C prompts).
    single = len(concs) == 1
    prompts = load_prompts(args.dataset, args.n if single else max(args.n, max(concs)), tok)

    # Both deepseek checkpoints declare arch Qwen3DSparkModel -> the right vLLM
    # method is "dspark" for BOTH. The dflash ckpt has markov_rank=0 (a no-op
    # zero-width Markov head = block-parallel DFlash-style baseline); the dspark
    # ckpt has markov_rank=256 (real sequential Markov head). Passing method
    # explicitly is honored (verified: dflash ckpt loads under method=dspark).
    # The PR's own launch command for Qwen3-8B requires attention_backend FLASH_ATTN
    # (DSpark uses non-causal attention; default backend mis-verifies -> ~0 accept).
    spec = {"method": args.method, "num_speculative_tokens": args.num_spec,
            "model": args.draft, "attention_backend": "FLASH_ATTN"}
    print("SPEC_DICT:", spec, flush=True)
    max_num_seqs = concs[0] if single else max(concs)
    llm = LLM(model=args.target, speculative_config=spec, gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=False, disable_log_stats=False,
              enable_prefix_caching=False,  # independent reqs -> clean per-step metrics
              max_num_seqs=max_num_seqs,    # cap in-flight concurrency at C
              trust_remote_code=True)

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    def snap():
        d = {}
        try:
            for x in llm.llm_engine.get_metrics():
                v = getattr(x, "value", None)
                if v is None:
                    v = getattr(x, "sum", None)
                if v is not None:
                    d[x.name] = v
        except Exception as e:
            d["__err__"] = str(e)
        return d

    results = {}
    # warmup with a held-out prompt (not measured; warms cudagraph + autotune)
    llm.generate(["Hello, how are you?"],
                 SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)

    for C in concs:
        # single-conc sweep: submit ALL n prompts, in-flight capped at C (max_num_seqs)
        # -> saturated batch for the whole run. multi-conc: just submit C (quick path).
        batch = prompts if single else prompts[:C]
        m0 = snap()
        t0 = time.perf_counter()
        outs = llm.generate(batch, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        m1 = snap()
        gen = sum(len(o.outputs[0].token_ids) for o in outs)

        def dl(k):
            return (m1.get(k, 0) or 0) - (m0.get(k, 0) or 0)
        acc = dl("vllm:spec_decode_num_accepted_tokens")
        ndraft_tok = dl("vllm:spec_decode_num_draft_tokens")
        ndrafts = dl("vllm:spec_decode_num_drafts")   # = sum over reqs of #spec steps
        accept_len = (acc / ndrafts + 1) if ndrafts else None
        steps_per_seq = (ndrafts / C) if C else None
        step_ms = (dt * 1000.0 / steps_per_seq) if steps_per_seq else None
        tps = gen / dt
        results[C] = {"conc": C, "n_prompts": len(batch), "gen": gen, "wall_s": round(dt, 3),
                      "tok_s": round(tps, 1), "per_seq_tok_s": round(tps / C, 1),
                      "accept_len": round(accept_len, 3) if accept_len else None,
                      "step_ms": round(step_ms, 3) if step_ms else None,
                      "steps_per_seq": round(steps_per_seq, 1) if steps_per_seq else None,
                      "raw_acc": acc, "raw_draft_tok": ndraft_tok, "raw_drafts": ndrafts,
                      "metric_keys": list(m1.keys()) if C == concs[0] else None}
        print(f">>> {args.method} {args.dataset} C={C}: "
              f"tok/s={results[C]['tok_s']} accept={results[C]['accept_len']} "
              f"step_ms={results[C]['step_ms']}", flush=True)

    out = {"method": args.method, "dataset": args.dataset, "draft": args.draft,
           "num_spec": args.num_spec, "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
