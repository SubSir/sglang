"""
DFLASH vs baseline dataset sweep (generic).

This is a *benchmark script* (not a CI test): it can take a long time because it
launches servers for multiple (attention_backend, tp_size) configs and runs a
workload for each (concurrency, num_samples) setting.

Example usage:
  ./venv/bin/python benchmark/dflash/bench_dflash_dataset_sweep.py \
      --data-name gsm8k \
      --output-md dflash_gsm8k_sweep.md

  ./venv/bin/python benchmark/dflash/bench_dflash_dataset_sweep.py \
      --data-name swe-bench --skip-baseline --concurrencies 32 --tp-sizes 8
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List

import requests
import torch
from datasets import load_dataset, Features, Sequence, Value
from transformers import AutoTokenizer

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)

# -------------------------
# Dataset loading & prompt
# -------------------------


def load_and_process_dataset(data_name: str):
    """
    The returned dataset must have a "turns" column, where each row is:
      {"turns": [str, str, ...]}
    We only use turns[0] as the prompt here.
    """
    # Math datasets
    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    # Chat datasets
    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(
            lambda x: {
                "formatted_input": (
                    f"{x['instruction']}\n\nInput:\n{x['input']}"
                    if x["input"]
                    else x["instruction"]
                )
            }
        )
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})

    elif data_name == "mt-bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        # prompt 字段就是一个多轮对话 turn 列表
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})

    # Coding datasets
    elif data_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = (
            "Write a solution to the following problem and make sure that it passes the tests:\n"
            "```python\n{prompt}\n```"
        )
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "mbpp":
        dataset = load_dataset(
            "google-research-datasets/mbpp", "sanitized", split="test"
        )
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})

    elif data_name == "lbpp":
        LBPP_PY_TEST_URL = "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/test.parquet"
        dataset = load_dataset("parquet", data_files={"test": LBPP_PY_TEST_URL})["test"]
        dataset = dataset.map(lambda x: {"turns": [x["instruction"]]})

    elif data_name == "swe-bench":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        prompt_fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "livecodebench":
        base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
        allowed_files = [
            "test.jsonl",
            "test2.jsonl",
            "test3.jsonl",
            "test4.jsonl",
            "test5.jsonl",
            "test6.jsonl",
        ]
        urls = [base + fn for fn in allowed_files]
        dataset = load_dataset("json", data_files={"test": urls})["test"]

        def format_lcb(doc):
            system_prompt = (
                "You are an expert Python programmer. You will be given a question (problem specification) "
                "and will generate a correct Python program that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            question_block = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                format_message = "### Format: Use the following code structure:"
                code_block = f"```python\n{doc['starter_code']}\n```"
            else:
                format_message = "### Format: Write your code in the following format:"
                code_block = "```python\n# YOUR CODE HERE\n```"
            answer_footer = "### Answer: (use the provided format with backticks)"
            return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"

        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]},
            remove_columns=dataset.column_names,
            features=target_features,
        )

    else:
        raise ValueError(f"Unsupported data_name: {data_name}")

    return dataset


def build_prompts_from_turns(
    dataset,
    tokenizer,
    max_samples: int,
    prompt_style: str,
) -> List[str]:
    """Convert dataset["turns"] into the final string prompt sent to /generate.

    - For prompt_style == 'chat': treats turns as a multi-turn conversation and calls tokenizer.apply_chat_template
    - For prompt_style == 'plain': uses turns[0] directly
    """
    prompts: List[str] = []

    n_total = len(dataset)
    if n_total == 0:
        raise RuntimeError("Dataset is empty (no samples).")

    for i in range(max_samples):
        ex = dataset[i % n_total]
        turns = ex["turns"]
        if not isinstance(turns, list) or len(turns) == 0:
            raise RuntimeError(f"Invalid turns field at index {i}: {turns}")

        if prompt_style == "plain":
            # 只用第一轮作为纯文本 prompt
            prompts.append(str(turns[0]))
        else:
            # chat 风格：把每一轮当成 user 的一条 message
            messages = [{"role": "user", "content": t} for t in turns]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(prompt)

    return prompts


# -------------------------
# Benchmark core
# -------------------------


def _is_blackwell() -> bool:
    # Prefer explicit env var, but also infer from compute capability (SM100+).
    if envs.IS_BLACKWELL.get():
        return True
    return get_device_sm() >= 100


def _flush_cache(base_url: str) -> None:
    resp = requests.get(base_url + "/flush_cache", timeout=60)
    resp.raise_for_status()


def _send_generate(
    base_url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    stop: list[str],
    timeout_s: int,
) -> dict:
    sampling_params: dict = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_new_tokens": int(max_new_tokens),
    }
    if stop:
        sampling_params["stop"] = stop
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": sampling_params,
        },
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    return resp.json()


def _send_generate_batch(
    base_url: str,
    prompts: list[str],
    *,
    max_new_tokens: int,
    stop: list[str],
    timeout_s: int,
) -> list[dict]:
    if not prompts:
        return []
    sampling_params: dict = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_new_tokens": int(max_new_tokens),
    }
    if stop:
        sampling_params["stop"] = stop
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompts,
            "sampling_params": sampling_params,
        },
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    out = resp.json()
    if not isinstance(out, list):
        raise RuntimeError(
            "Expected a list response for batched /generate, but got "
            f"type={type(out).__name__}."
        )
    return out


@dataclass(frozen=True)
class BenchMetrics:
    latency_s: float
    output_tokens: int
    output_toks_per_s: float
    accuracy: Optional[float]
    invalid_rate: Optional[float]
    spec_accept_length: Optional[float]
    spec_verify_ct_sum: int
    spec_verify_tokens_sum: int


def _run_requests(
    base_url: str,
    *,
    prompts: list[str],
    max_new_tokens: int,
    concurrency: int,
    batch_requests: bool,
    stop: list[str],
    timeout_s: int,
    expect_dflash: bool,
) -> BenchMetrics:
    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_verify_tokens_sum = 0
    spec_accept_lengths: list[float] = []

    if batch_requests:
        bs = max(int(concurrency), 1)
        for start_idx in range(0, len(prompts), bs):
            chunk_prompts = prompts[start_idx : start_idx + bs]
            outs = _send_generate_batch(
                base_url,
                chunk_prompts,
                max_new_tokens=max_new_tokens,
                stop=stop,
                timeout_s=timeout_s,
            )
            if len(outs) != len(chunk_prompts):
                raise RuntimeError(
                    "Batched /generate output length mismatch: "
                    f"got {len(outs)} outputs for {len(chunk_prompts)} prompts."
                )

            for out in outs:
                meta = out.get("meta_info", {}) or {}
                total_tokens += int(meta.get("completion_tokens", 0))
                spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
                spec_verify_tokens_sum += int(meta.get("spec_verify_tokens", 0))
                if "spec_accept_length" in meta:
                    try:
                        spec_accept_lengths.append(float(meta["spec_accept_length"]))
                    except (TypeError, ValueError):
                        pass
    else:
        with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
            futures = {
                pool.submit(
                    _send_generate,
                    base_url,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    stop=stop,
                    timeout_s=timeout_s,
                ): i
                for i, prompt in enumerate(prompts)
            }
            for fut in as_completed(futures):
                out = fut.result()
                meta = out.get("meta_info", {}) or {}
                total_tokens += int(meta.get("completion_tokens", 0))
                spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
                spec_verify_tokens_sum += int(meta.get("spec_verify_tokens", 0))
                if "spec_accept_length" in meta:
                    try:
                        spec_accept_lengths.append(float(meta["spec_accept_length"]))
                    except (TypeError, ValueError):
                        pass

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    if expect_dflash and spec_verify_ct_sum <= 0:
        raise RuntimeError(
            "DFLASH sanity check failed: did not observe any `spec_verify_ct` in responses "
            "(DFLASH may not have been enabled)."
        )

    spec_accept_length = (
        float(statistics.mean(spec_accept_lengths)) if spec_accept_lengths else None
    )

    # 通用脚本这里不做正确性评估
    acc = None
    invalid_rate = None

    return BenchMetrics(
        latency_s=float(latency),
        output_tokens=int(total_tokens),
        output_toks_per_s=float(toks_per_s),
        accuracy=acc,
        invalid_rate=invalid_rate,
        spec_accept_length=spec_accept_length,
        spec_verify_ct_sum=int(spec_verify_ct_sum),
        spec_verify_tokens_sum=int(spec_verify_tokens_sum),
    )


def _format_table(
    *,
    tp_sizes: list[int],
    concurrencies: list[int],
    values: dict[tuple[int, int], Optional[float]],
    float_fmt: str,
) -> str:
    header = ["tp\\conc"] + [str(c) for c in concurrencies]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for tp in tp_sizes:
        row = [str(tp)]
        for c in concurrencies:
            v = values.get((tp, c), None)
            row.append("N/A" if v is None else format(v, float_fmt))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# -------------------------
# main
# -------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-name",
        type=str,
        default=None,
        help=(
            "Dataset name, one of: "
            "gsm8k, math500, aime24, aime25, alpaca, mt-bench, "
            "humaneval, mbpp, lbpp, swe-bench, livecodebench"
        ),
    )
    parser.add_argument(
        "--data-names",
        type=str,
        default=None,
        help="Comma-separated list of dataset names.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Write a markdown report to this file (disabled by default).",
    )
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", type=str, default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help=(
            "Skip running the baseline (target-only) sweep; "
            "only run DFLASH and report N/A for baseline/speedup."
        ),
    )
    parser.add_argument(
        "--batch-requests",
        action="store_true",
        help=(
            "Send prompts as server-side batched /generate requests "
            "(batch size = concurrency) instead of client-side concurrent requests."
        ),
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=["chat", "plain"],
        default="chat",
        help=(
            "How to wrap dataset turns into the final prompt: "
            "'chat' uses tokenizer.apply_chat_template on all turns; "
            "'plain' uses turns[0] as raw text."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument(
        "--speculative-eagle-topk",
        type=int,
        default=None,
        help="Override speculative_eagle_topk for DFLASH server.",
    )
    parser.add_argument(
        "--speculative-num-draft-tokens",
        type=int,
        default=None,
        help="Override speculative_num_draft_tokens for DFLASH server.",
    )
    parser.add_argument(
        "--tp-sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated list, filtered by visible CUDA devices.",
    )
    parser.add_argument(
        "--concurrencies",
        type=str,
        default="1,2,4,8,16,32,64,128",
        help="Comma-separated list of client concurrency levels.",
    )
    parser.add_argument(
        "--samples-per-concurrency-base",
        type=int,
        default=128,
        help="num_samples = base * concurrency.",
    )
    parser.add_argument(
        "--max-samples-per-config",
        type=int,
        default=2048,
        help="Cap num_samples per (tp, concurrency) run.",
    )
    parser.add_argument(
        "--attention-backends",
        type=str,
        default="flashinfer,fa3",
        help="Comma-separated list. Will auto-skip fa3 on Blackwell/SM<90.",
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Disable CUDA graph optimization.",
    )
    parser.add_argument(
        "--enable-piecewise-cuda-graph",
        action="store_true",
        help="Enable piecewise CUDA graph for target prefill/extend path.",
    )
    parser.add_argument(
        "--piecewise-cuda-graph-max-tokens",
        type=int,
        default=None,
        help="Maximum token bucket for piecewise CUDA graph capture.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep.")

    visible_gpus = int(torch.cuda.device_count())
    tp_sizes = [int(x) for x in args.tp_sizes.split(",") if x.strip()]
    tp_sizes = [tp for tp in tp_sizes if 1 <= tp <= visible_gpus]
    if not tp_sizes:
        raise RuntimeError(
            f"No tp sizes are runnable with visible_gpus={visible_gpus}. "
            "Set CUDA_VISIBLE_DEVICES accordingly."
        )

    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    concurrencies = [c for c in concurrencies if c >= 1]
    if not concurrencies:
        raise RuntimeError("No concurrencies specified.")

    num_samples_by_conc = {
        c: min(
            int(args.samples_per_concurrency_base) * int(c),
            int(args.max_samples_per_config),
        )
        for c in concurrencies
    }
    max_samples = max(num_samples_by_conc.values())

    attention_backends = [
        s.strip() for s in args.attention_backends.split(",") if s.strip()
    ]
    is_blackwell = _is_blackwell()
    device_sm = get_device_sm()
    if is_blackwell:
        attention_backends = [b for b in attention_backends if b == "flashinfer"]
    if device_sm < 90:
        attention_backends = [b for b in attention_backends if b != "fa3"]
    attention_backends = attention_backends or ["flashinfer"]

    # Determine data names to sweep
    if args.data_names:
        data_names = [s.strip() for s in args.data_names.split(",") if s.strip()]
    elif args.data_name:
        data_names = [args.data_name]
    else:
        raise ValueError("Either --data-name or --data-names must be provided.")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    # Pre-load all datasets and prompts
    dataset_prompts = {}
    for dname in data_names:
        print(f"Loading dataset: {dname}")
        ds = load_and_process_dataset(dname)
        dataset_prompts[dname] = build_prompts_from_turns(
            ds,
            tokenizer=tokenizer,
            max_samples=max_samples,
            prompt_style=args.prompt_style,
        )

    # 通用脚本默认没有特殊 stop 标记；如果你有可以按数据集加逻辑
    default_stop: list[str] = []

    # Results indexed by (backend, tp, data_name, concurrency)
    baseline_toks: dict[tuple[str, int, str, int], Optional[float]] = {}
    dflash_toks: dict[tuple[str, int, str, int], Optional[float]] = {}
    dflash_accept_len: dict[tuple[str, int, str, int], Optional[float]] = {}
    dflash_verify_tokens: dict[tuple[str, int, str, int], Optional[int]] = {}
    dflash_forward_ct: dict[tuple[str, int, str, int], Optional[int]] = {}
    baseline_acc: dict[tuple[str, int, str, int], Optional[float]] = {}
    dflash_acc: dict[tuple[str, int, str, int], Optional[float]] = {}

    for backend in attention_backends:
        for tp in tp_sizes:
            port_base = find_available_port(20000)

            common_server_args: list[str] = [
                "--trust-remote-code",
                "--attention-backend",
                backend,
                "--tp-size",
                str(tp),
                "--dtype",
                str(args.dtype),
                "--mem-fraction-static",
                str(args.mem_fraction_static),
                "--max-running-requests",
                str(args.max_running_requests),
                "--page-size",
                "1"
            ]
            if args.disable_cuda_graph:
                common_server_args.append("--disable-cuda-graph")
            else:
                common_server_args.extend(
                    [
                        "--cuda-graph-bs",
                        *[str(i) for i in range(1, 129)],
                        "--cuda-graph-max-bs",
                        "128",
                    ]
                )
                if args.enable_piecewise_cuda_graph:
                    common_server_args.append("--enable-piecewise-cuda-graph")
                    if args.piecewise_cuda_graph_max_tokens is not None:
                        common_server_args.extend(
                            [
                                "--piecewise-cuda-graph-max-tokens",
                                str(args.piecewise_cuda_graph_max_tokens),
                            ]
                        )
            if args.disable_radix_cache:
                common_server_args.append("--disable-radix-cache")

            # baseline
            if not args.skip_baseline:
                print(f"\n=== backend={backend} tp={tp} (baseline) ===")
                baseline_port = port_base
                baseline_url = f"http://127.0.0.1:{baseline_port}"
                baseline_proc = popen_launch_server(
                    args.target_model,
                    baseline_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=common_server_args,
                )
                try:
                    # warm up
                    _send_generate(
                        baseline_url,
                        "Hello",
                        max_new_tokens=8,
                        stop=[],
                        timeout_s=min(int(args.timeout_s), 300),
                    )

                    for dname in data_names:
                        print(f"--- Dataset: {dname} (baseline) ---")
                        prompts = dataset_prompts[dname]
                        for conc in concurrencies:
                            n = num_samples_by_conc[conc]
                            _flush_cache(baseline_url)
                            metrics = _run_requests(
                                baseline_url,
                                prompts=prompts[:n],
                                max_new_tokens=int(args.max_new_tokens),
                                concurrency=int(conc),
                                batch_requests=bool(args.batch_requests),
                                stop=default_stop,
                                timeout_s=int(args.timeout_s),
                                expect_dflash=False,
                            )
                            baseline_toks[(backend, tp, dname, conc)] = (
                                metrics.output_toks_per_s
                            )
                            baseline_acc[(backend, tp, dname, conc)] = metrics.accuracy
                            token_info = (
                                f" output_tokens={metrics.output_tokens}"
                            )
                            print(
                                f"[{dname} baseline] conc={conc:>2} n={n:<4} "
                                f"toks/s={metrics.output_toks_per_s:,.2f} "
                                f"latency={metrics.latency_s:.1f}s"
                                f"{token_info}"
                            )
                finally:
                    kill_process_tree(baseline_proc.pid)
                    try:
                        baseline_proc.wait(timeout=30)
                    except Exception:
                        pass

            # DFLASH
            print(f"\n=== backend={backend} tp={tp} (DFLASH) ===")
            dflash_port = find_available_port(port_base + 1)
            dflash_url = f"http://127.0.0.1:{dflash_port}"
            dflash_other_args = [
                *common_server_args,
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                args.draft_model,
            ]
            if args.speculative_num_draft_tokens is not None:
                dflash_other_args.extend(
                    [
                        "--speculative-num-draft-tokens",
                        str(args.speculative_num_draft_tokens),
                    ]
                )
            if args.speculative_eagle_topk is not None:
                dflash_other_args.extend(
                    [
                        "--speculative-eagle-topk",
                        str(args.speculative_eagle_topk),
                    ]
                )
            dflash_proc = popen_launch_server(
                args.target_model,
                dflash_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=dflash_other_args,
            )
            try:
                _send_generate(
                    dflash_url,
                    "Hello",
                    max_new_tokens=8,
                    stop=[],
                    timeout_s=min(int(args.timeout_s), 300),
                )
                for dname in data_names:
                    print(f"--- Dataset: {dname} (DFLASH) ---")
                    prompts = dataset_prompts[dname]
                    for conc in concurrencies:
                        n = num_samples_by_conc[conc]
                        _flush_cache(dflash_url)
                        metrics = _run_requests(
                            dflash_url,
                            prompts=prompts[:n],
                            max_new_tokens=int(args.max_new_tokens),
                            concurrency=int(conc),
                            batch_requests=bool(args.batch_requests),
                            stop=default_stop,
                            timeout_s=int(args.timeout_s),
                            expect_dflash=True,
                        )
                        dflash_toks[(backend, tp, dname, conc)] = (
                            metrics.output_toks_per_s
                        )
                        dflash_accept_len[(backend, tp, dname, conc)] = (
                            metrics.spec_accept_length
                        )
                        dflash_verify_tokens[(backend, tp, dname, conc)] = (
                            metrics.spec_verify_tokens_sum
                        )
                        dflash_forward_ct[(backend, tp, dname, conc)] = (
                            metrics.spec_verify_ct_sum
                        )
                        dflash_acc[(backend, tp, dname, conc)] = metrics.accuracy
                        token_info = (
                            f" output_tokens={metrics.output_tokens}"
                        )
                        print(
                            f"[{dname} DFLASH]   conc={conc:>2} n={n:<4} "
                            f"toks/s={metrics.output_toks_per_s:,.2f} "
                            f"latency={metrics.latency_s:.1f}s "
                            f"accept_len={metrics.spec_accept_length if metrics.spec_accept_length is not None else float('nan'):.3f} "
                            f"forward_ct={metrics.spec_verify_ct_sum} "
                            f"spec_verify_tokens_sum={metrics.spec_verify_tokens_sum}"
                            f"{token_info}"
                        )
            finally:
                kill_process_tree(dflash_proc.pid)
                try:
                    dflash_proc.wait(timeout=30)
                except Exception:
                    pass

    # Render markdown.
    md_lines: list[str] = []
    data_names_str = ", ".join(data_names)
    md_lines.append(f"# DFLASH Sweep: {data_names_str}")
    md_lines.append("")
    md_lines.append("## Settings")
    md_lines.append(f"- data_names: `{data_names_str}`")
    md_lines.append(f"- target_model: `{args.target_model}`")
    md_lines.append(f"- draft_model: `{args.draft_model}`")
    md_lines.append(f"- prompt_style: `{args.prompt_style}`")
    md_lines.append(f"- max_new_tokens: `{args.max_new_tokens}`")
    md_lines.append(f"- attention_backends: `{', '.join(attention_backends)}`")
    md_lines.append(f"- tp_sizes: `{', '.join(str(x) for x in tp_sizes)}`")
    md_lines.append(f"- concurrencies: `{', '.join(str(x) for x in concurrencies)}`")
    md_lines.append(
        f"- samples_per_concurrency: `base={args.samples_per_concurrency_base}`"
    )
    md_lines.append(f"- device_sm: `{device_sm}`")
    md_lines.append(f"- is_blackwell: `{is_blackwell}`")
    md_lines.append(f"- skip_baseline: `{bool(args.skip_baseline)}`")
    md_lines.append("")
    md_lines.append(
        "Note: This sweep focuses on throughput. Correctness is not evaluated "
        "for this generic dataset script."
    )
    md_lines.append("")

    for dname in data_names:
        md_lines.append(f"# Results for Dataset: `{dname}`")
        for backend in attention_backends:
            md_lines.append(f"## Backend: `{backend}`")
            md_lines.append("")

            baseline_values = {
                (tp, conc): baseline_toks.get((backend, tp, dname, conc), None)
                for tp in tp_sizes
                for conc in concurrencies
            }
            dflash_values = {
                (tp, conc): dflash_toks.get((backend, tp, dname, conc), None)
                for tp in tp_sizes
                for conc in concurrencies
            }
            speedup_values: dict[tuple[int, int], Optional[float]] = {}
            for tp in tp_sizes:
                for conc in concurrencies:
                    b = baseline_values.get((tp, conc), None)
                    d = dflash_values.get((tp, conc), None)
                    speedup_values[(tp, conc)] = (
                        None if (b is None or d is None or b <= 0) else (d / b)
                    )

            md_lines.append("### Baseline output tok/s")
            md_lines.append(
                _format_table(
                    tp_sizes=tp_sizes,
                    concurrencies=concurrencies,
                    values=baseline_values,
                    float_fmt=",.2f",
                )
            )
            md_lines.append("")
            md_lines.append("### DFLASH output tok/s")
            md_lines.append(
                _format_table(
                    tp_sizes=tp_sizes,
                    concurrencies=concurrencies,
                    values=dflash_values,
                    float_fmt=",.2f",
                )
            )
            md_lines.append("")
            md_lines.append("### Speedup (DFLASH / baseline)")
            md_lines.append(
                _format_table(
                    tp_sizes=tp_sizes,
                    concurrencies=concurrencies,
                    values=speedup_values,
                    float_fmt=".3f",
                )
            )
            md_lines.append("")
            md_lines.append(
                "### DFLASH acceptance length (mean per-request spec_accept_length)"
            )
            md_lines.append(
                _format_table(
                    tp_sizes=tp_sizes,
                    concurrencies=concurrencies,
                    values={
                        (tp, conc): dflash_accept_len.get(
                            (backend, tp, dname, conc), None
                        )
                        for tp in tp_sizes
                        for conc in concurrencies
                    },
                    float_fmt=".3f",
                )
            )
            md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        md_lines.append("### Speedup (DFLASH / baseline)")
        md_lines.append(
            _format_table(
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                values=speedup_values,
                float_fmt=".3f",
            )
        )
        md_lines.append("")
        md_lines.append(
            "### DFLASH acceptance length (mean per-request spec_accept_length)"
        )
        md_lines.append(
            _format_table(
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                values={
                    (tp, conc): dflash_accept_len.get((backend, tp, dname, conc), None)
                    for tp in tp_sizes
                    for conc in concurrencies
                },
                float_fmt=".3f",
            )
        )
        md_lines.append("")

        md_lines.append("### DFLASH total forward count")
        md_lines.append(
            _format_table(
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                values={
                    (tp, conc): dflash_forward_ct.get(
                        (backend, tp, dname, conc), None
                    )
                    for tp in tp_sizes
                    for conc in concurrencies
                },
                float_fmt="d",
            )
        )
        md_lines.append("")

        md_lines.append("### DFLASH total verified tokens")
        md_lines.append(
            _format_table(
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                values={
                    (tp, conc): dflash_verify_tokens.get(
                        (backend, tp, dname, conc), None
                    )
                    for tp in tp_sizes
                    for conc in concurrencies
                },
                float_fmt="d",
            )
        )
        md_lines.append("")

    if args.output_md:
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
            f.write("\n")
        print(f"\nWrote markdown report to: {args.output_md}")
    else:
        print("\nMarkdown report disabled (pass --output-md to write one).")


if __name__ == "__main__":
    main()