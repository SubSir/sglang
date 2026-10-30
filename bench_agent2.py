"""(bench_agent2: bench_agent + --prompt-cache.) Splash #329's agent-session benchmark on an SGLang server, scaled.

S sessions share one system prompt. Round 1: each session sends system + its
task. Rounds 2..R: each session, in a fixed shuffled order, sends its whole
history plus a tool output. T=0, no thinking, a fixed number of output tokens
(ignore_eos). A follow-up resumed when the server reused the prefix up to the
end of the session's previous prompt; otherwise it fell back to an older point
(usually the shared system prompt).

Prompts are token ids built by concatenating separately encoded segments and
the previous turn's exact output ids, so the cache sees an exact prefix.
--record writes every turn's output ids; --replay reuses them for the history,
so every arm is sent byte-identical prompts.

    python bench_agent.py --port P --model-path M --sessions 192 --rounds 5 \
        --system-tokens 3800 --task-tokens 256 --tool-tokens 1000 \
        --output-tokens 48 --max-inflight 32 --seed 1 --tag T --out run.jsonl
"""
import argparse
import asyncio
import json
import random
import time

import aiohttp
from transformers import AutoTokenizer

from sglang.benchmark.datasets.common import SHAREGPT_FILENAME, SHAREGPT_REPO_ID
from sglang.benchmark.utils import download_and_cache_hf_file

PAGE = 64  # resume slack: the cached point is page-aligned on the target


def real_text(tok, seed, n, length):
    """n deterministic chunks of real ShareGPT text, length tokens each."""
    path = download_and_cache_hf_file(repo_id=SHAREGPT_REPO_ID, filename=SHAREGPT_FILENAME)
    convs = json.load(open(path))
    random.Random(seed).shuffle(convs)
    out, buf = [], []
    for c in convs:
        for t in c.get("conversations", c.get("conversation", [])):
            buf.extend(tok.encode(t["value"] + "\n", add_special_tokens=False))
            while len(buf) >= length:
                out.append(buf[:length])
                buf = buf[length:]
                if len(out) == n:
                    return out
    raise ValueError(f"ShareGPT ran out at {len(out)} of {n} chunks")


def build(args, tok):
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    n_tools = args.sessions * (args.rounds - 1)
    turn_open = enc("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    system = enc("<|im_start|>system\n") + real_text(tok, args.seed, 1, args.system_tokens)[0] + enc("<|im_end|>\n")
    tasks = [enc("<|im_start|>user\n") + c + turn_open
             for c in real_text(tok, args.seed + 2, args.sessions, args.task_tokens)]
    tools = [enc("<|im_start|>user\n<tool_response>\n") + c + enc("\n</tool_response>") + turn_open
             for c in real_text(tok, args.seed + 3, n_tools, args.tool_tokens)] if n_tools else []
    order_rng = random.Random(args.seed + 1)
    order = []
    for _ in range(args.rounds):
        o = list(range(args.sessions))
        order_rng.shuffle(o)
        order.append(o)
    return system, tasks, tools, enc("<|im_end|>\n"), order


def cached_build(args, tok):
    """build() with an on-disk cache: the prompts are deterministic in these arguments, so a lane
    reuses the token ids instead of re-tokenizing ShareGPT (~25 s per lane)."""
    if not args.prompt_cache:
        return build(args, tok)
    import hashlib
    import os

    key = json.dumps([os.path.basename(args.model_path.rstrip("/")), args.seed, args.sessions, args.rounds,
                      args.system_tokens, args.task_tokens, args.tool_tokens])
    path = f"{args.prompt_cache}/{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
    if os.path.exists(path):
        return tuple(json.load(open(path)))
    out = build(args, tok)
    os.makedirs(args.prompt_cache, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, path)  # atomic: concurrent lanes on other GPUs read whole files only
    return out


async def one(session, url, ids, args):
    payload = {"input_ids": ids, "stream": True,
               "sampling_params": {"temperature": 0, "max_new_tokens": args.output_tokens, "ignore_eos": True}}
    for attempt in range(3):
        t0 = time.perf_counter()
        ttft, last = None, None
        try:
            async with session.post(url, json=payload) as r:
                async for line in r.content:
                    line = line.strip()
                    if not line.startswith(b"data:") or line == b"data: [DONE]":
                        continue
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    last = json.loads(line[5:])
            return ttft, time.perf_counter() - t0, last
        except aiohttp.ServerDisconnectedError:
            # A pooled keep-alive connection the server already closed (long outputs leave a
            # session's connection idle past the keep-alive timeout): nothing was served, so
            # resend on a fresh connection and time it from scratch. A mid-stream drop is real.
            if ttft is not None or attempt == 2:
                raise


async def main(args):
    tok = AutoTokenizer.from_pretrained(args.model_path)
    system, tasks, tools, close, order = cached_build(args, tok)
    replay = {}
    if args.replay:
        for line in open(args.replay):
            r = json.loads(line)
            if "output_ids" in r:
                replay[(r["session"], r["round"])] = r["output_ids"]
    url = f"http://{args.host}:{args.port}/generate"
    history = {s: system + tasks[s] for s in range(args.sessions)}
    prev_prompt = {}
    rows, sem = [], asyncio.Semaphore(args.max_inflight)
    out = open(args.out, "a")
    t_start = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=None, sock_read=3600)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        for rnd in range(args.rounds):
            async def turn(s):
                ids = history[s] if rnd == 0 else history[s] + tools[(rnd - 1) * args.sessions + s]
                async with sem:
                    ttft, e2e, last = await one(http, url, ids, args)
                m = last["meta_info"]
                own = last["output_ids"]
                gen = replay.get((s, rnd), own)
                cached = int(m.get("cached_tokens", 0))
                row = {"tag": args.tag, "session": s, "round": rnd, "prompt": len(ids), "cached": cached,
                       "prev_prompt": prev_prompt.get(s), "ttft": ttft, "e2e": e2e,
                       "out": len(own), "accept": m.get("spec_accept_length"),
                       "tpot": (e2e - ttft) / max(len(own) - 1, 1),
                       # server-side breakdown: queue wait, prefill (forward entry -> prefill done), first token
                       "tm": {k: m[k] for k in ("request_received_ts", "queue_time", "forward_entry_time",
                                                "prefill_finished_time", "first_token_latency") if k in m},
                       "out_hash": hash(tuple(own)), "same_as_replay": None if (s, rnd) not in replay else own == replay[(s, rnd)],
                       # joins the server's DBGMATCH lines (hash of the match key's last 32 tokens; the key may drop the last token)
                       "tail": [hash(tuple(ids[-32:])), hash(tuple(ids[-33:-1]))],
                       "resumed": None if rnd == 0 else cached >= prev_prompt[s] - 2 * PAGE,
                       "system_only": cached <= len(system) + PAGE}
                if args.record:
                    row["output_ids"] = gen
                prev_prompt[s] = len(ids)
                history[s] = ids + gen + close
                rows.append(row)
                out.write(json.dumps(row) + "\n")
                out.flush()
            # One round at a time, sessions dispatched in the round's fixed order.
            await asyncio.gather(*(turn(s) for s in order[rnd]))
    wall = time.perf_counter() - t_start
    fu = [r for r in rows if r["round"] > 0]
    t = sorted(r["ttft"] for r in fu)
    summary = {"tag": args.tag, "summary": True, "sessions": args.sessions, "rounds": args.rounds,
               "follow_ups": len(fu), "resumed": sum(r["resumed"] for r in fu),
               "system_only": sum(r["system_only"] for r in fu),
               "ttft_mean": sum(t) / len(t), "ttft_p50": t[len(t) // 2], "ttft_p90": t[int(0.9 * len(t))],
               "ttft_mean_all": sum(r["ttft"] for r in rows) / len(rows),
               "tpot_mean": sum(r["tpot"] for r in rows) / len(rows),
               "prompt_tokens": sum(r["prompt"] for r in fu), "cached_tokens": sum(r["cached"] for r in fu),
               "recomputed_tokens": sum(r["prompt"] - r["cached"] for r in fu),
               "accept_mean": sum(r["accept"] or 0 for r in rows) / len(rows), "wall": wall,
               "diverged_from_replay": sum(r["same_as_replay"] is False for r in rows),
               "args": {k: v for k, v in vars(args).items() if k not in ("record", "replay")}}
    out.write(json.dumps(summary) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--sessions", type=int, default=192)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--system-tokens", type=int, default=3800)
    p.add_argument("--task-tokens", type=int, default=256)
    p.add_argument("--tool-tokens", type=int, default=1000)
    p.add_argument("--output-tokens", type=int, default=48)
    p.add_argument("--max-inflight", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--record", action="store_true", help="store each turn's output ids")
    p.add_argument("--replay", help="a --record file: reuse its output ids for the history")
    p.add_argument("--prompt-cache", default="", help="dir caching the tokenized prompts per workload")
    asyncio.run(main(p.parse_args()))
