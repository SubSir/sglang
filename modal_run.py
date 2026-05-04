from __future__ import annotations

import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import modal
import requests

APP_NAME = "sglang-dflash-qwen3-benchmark"
HUGGINGFACE_SECRET_NAME = "huggingface-secret"
SOURCE_VOLUME_NAME = "draft-dynamic-verify"
WORKSPACE_VOLUME_NAME = "dflash_workspace"
SOURCE_VOLUME_MOUNT = "/root/draft-dynamic-verify"
WORKSPACE_VOLUME_MOUNT = "/root/dflash_workspace"

DEFAULT_TARGET_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_DATASETS = ("gsm8k",)
DEFAULT_CONCURRENCIES = (64)
DEFAULT_NUM_PROMPTS = 1024
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_PORT = 30000
DEFAULT_TIMEOUT_S = 3600
DEFAULT_SERVER_TIMEOUT_S = 1800
DEFAULT_MEM_FRACTION_STATIC = 0.90
DEFAULT_MAX_RUNNING_REQUESTS = 128
DEFAULT_CUDA_GRAPH_MAX_BS = 64
DEFAULT_ATTENTION_BACKEND = "flashinfer"
DEFAULT_TP_SIZE = 1
DEFAULT_GPU = "B200"
DEFAULT_FLASHINFER_WORKSPACE_SIZE_MB = 1024

DROP_WEIGHT_KEYS = frozenset({"embed_tokens.weight", "lm_head.weight"})
EXTRA_EXCLUDED_FILES = frozenset(
    {"optimizer.pt", "scheduler.pt", "training_state.pt"}
)

MODE_CONFIGS = (
    {
        "name": "no_dynamic",
        "dynamic_vbs": False,
        "predictor": None,
    },
    {
        "name": "confidence",
        "dynamic_vbs": True,
        "predictor": "confidence",
    },
    {
        "name": "mlp_head",
        "dynamic_vbs": True,
        "predictor": "mlp_head",
    },
)
MODE_CONFIG_BY_NAME = {mode["name"]: mode for mode in MODE_CONFIGS}
DEFAULT_MODE_NAMES = tuple(mode["name"] for mode in MODE_CONFIGS)
DEFAULT_EXECUTION_MODE = "spawn"

app = modal.App(APP_NAME)

source_volume = modal.Volume.from_name(SOURCE_VOLUME_NAME)
workspace_volume = modal.Volume.from_name(WORKSPACE_VOLUME_NAME, create_if_missing=True)

base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "wget", "libnuma-dev")
)

image = (
    base_image.run_commands(
        "git clone https://github.com/SubSir/sglang.git /root/sglang_local",
        "cd /root/sglang_local && pip install -e \"python\"",
        "pip install --upgrade --force-reinstall nvidia-cudnn-cu12==9.16.0.29",
        "pip install datasets loguru numpy requests rich safetensors tqdm transformers",
    )
    .add_local_dir(
        "./python",
        remote_path="/root/sglang_local/python_local",
        copy=True,
    )
    .add_local_file(
        "./benchmark.py",
        remote_path="/root/sglang_local/benchmark.py",
        copy=True,
    )
    .run_commands(
        "rm -rf /root/sglang_local/python && cp -r /root/sglang_local/python_local /root/sglang_local/python",
    )
)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _parse_csv_str(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_csv_int(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _iter_checkpoint_candidate_dirs(source_root: Path) -> list[Path]:
    candidates: set[Path] = set()
    if (source_root / "model.safetensors.index.json").exists() or (
        source_root / "model.safetensors"
    ).exists():
        candidates.add(source_root)

    for pattern in ("model.safetensors.index.json", "model.safetensors"):
        for path in source_root.rglob(pattern):
            if path.is_file():
                candidates.add(path.parent)

    return sorted(candidates)


def _resolve_source_checkpoint_dir(source_root: Path, model_name: str) -> Path:
    if not source_root.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_root}")

    if source_root.is_file():
        raise ValueError(f"Expected a directory for source_root, got file: {source_root}")

    model_tail = model_name.split("/")[-1].lower()
    candidates = _iter_checkpoint_candidate_dirs(source_root)
    if not candidates:
        raise FileNotFoundError(
            "No model.safetensors.index.json or model.safetensors found under "
            f"source root: {source_root}"
        )

    exact_name_matches = [p for p in candidates if p.name.lower() == model_tail]
    if len(exact_name_matches) == 1:
        return exact_name_matches[0]

    contains_matches = [p for p in candidates if model_tail in str(p).lower()]
    if len(contains_matches) == 1:
        return contains_matches[0]

    if len(candidates) == 1:
        return candidates[0]

    rendered = "\n".join(f"- {p}" for p in candidates[:20])
    raise ValueError(
        "Could not uniquely resolve source checkpoint directory. "
        f"Found multiple candidates under {source_root}:\n{rendered}"
    )


def _prepare_export_dir(export_dir: Path, *, allow_overwrite: bool) -> None:
    if export_dir.exists():
        if not allow_overwrite and any(export_dir.iterdir()):
            raise ValueError(
                f"Export dir {export_dir} already exists and is non-empty. "
                "Pass allow_overwrite=True to replace it."
            )
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)


def _copy_passthrough_files(source_dir: Path, export_dir: Path) -> list[str]:
    copied: list[str] = []
    passthrough_files = [
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
    ]
    for name in passthrough_files:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, export_dir / name)
            copied.append(name)
    return copied


def _rewrite_sharded_safetensors(source_dir: Path, export_dir: Path) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    index_payload = _load_json(source_dir / "model.safetensors.index.json")
    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("source model.safetensors.index.json is missing weight_map")

    shard_to_keys: OrderedDict[str, list[str]] = OrderedDict()
    removed_keys: list[str] = []
    kept_keys = 0
    total_parameters = 0
    total_size_bytes = 0
    new_weight_map: dict[str, str] = {}

    for key, shard_name in weight_map.items():
        if key in DROP_WEIGHT_KEYS:
            removed_keys.append(str(key))
            continue
        shard_to_keys.setdefault(str(shard_name), []).append(str(key))

    shard_names = list(shard_to_keys)
    total_shards = len(shard_names)
    if total_shards == 0:
        raise ValueError("All weights were filtered out; nothing left to export.")

    for shard_index, original_shard_name in enumerate(shard_names, start=1):
        source_shard_path = source_dir / original_shard_name
        if not source_shard_path.exists():
            raise FileNotFoundError(f"Missing source shard: {source_shard_path}")

        new_shard_name = f"model-{shard_index:05d}-of-{total_shards:05d}.safetensors"
        export_tensors: dict[str, Any] = {}
        with safe_open(str(source_shard_path), framework="pt") as handle:
            for key in shard_to_keys[original_shard_name]:
                tensor = handle.get_tensor(key)
                export_tensors[key] = tensor
                new_weight_map[key] = new_shard_name
                kept_keys += 1
                total_parameters += int(tensor.numel())
                total_size_bytes += int(tensor.numel() * tensor.element_size())

        save_file(
            export_tensors,
            str(export_dir / new_shard_name),
            metadata={"format": "pt"},
        )

    new_index_payload = {
        "metadata": {
            "total_parameters": int(total_parameters),
            "total_size": int(total_size_bytes),
        },
        "weight_map": dict(sorted(new_weight_map.items())),
    }
    (export_dir / "model.safetensors.index.json").write_text(
        json.dumps(new_index_payload, indent=2, sort_keys=True) + "\n"
    )

    return {
        "removed_keys": sorted(removed_keys),
        "kept_keys": int(kept_keys),
        "num_shards": int(total_shards),
        "total_parameters": int(total_parameters),
        "total_size_bytes": int(total_size_bytes),
    }


def _rewrite_single_safetensors(source_dir: Path, export_dir: Path) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    source_path = source_dir / "model.safetensors"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source weights: {source_path}")

    export_tensors: dict[str, Any] = {}
    removed_keys: list[str] = []
    kept_keys = 0
    total_parameters = 0
    total_size_bytes = 0

    with safe_open(str(source_path), framework="pt") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            if key in DROP_WEIGHT_KEYS:
                removed_keys.append(str(key))
                continue
            tensor = handle.get_tensor(key)
            export_tensors[str(key)] = tensor
            kept_keys += 1
            total_parameters += int(tensor.numel())
            total_size_bytes += int(tensor.numel() * tensor.element_size())

    if not export_tensors:
        raise ValueError("All weights were filtered out; nothing left to export.")

    save_file(
        export_tensors,
        str(export_dir / "model.safetensors"),
        metadata=metadata or {"format": "pt"},
    )

    return {
        "source_format": "single_file",
        "removed_keys": sorted(removed_keys),
        "kept_keys": int(kept_keys),
        "num_shards": 1,
        "total_parameters": int(total_parameters),
        "total_size_bytes": int(total_size_bytes),
    }


def _rewrite_model_weights(source_dir: Path, export_dir: Path) -> dict[str, Any]:
    index_path = source_dir / "model.safetensors.index.json"
    single_path = source_dir / "model.safetensors"
    if index_path.exists():
        stats = _rewrite_sharded_safetensors(source_dir, export_dir)
        stats["source_format"] = "sharded_index"
        return stats
    if single_path.exists():
        return _rewrite_single_safetensors(source_dir, export_dir)
    raise FileNotFoundError(
        "Expected either model.safetensors.index.json or model.safetensors under "
        f"{source_dir}"
    )


def _export_draft_checkpoint(
    *,
    source_dir: Path,
    export_dir: Path,
    allow_overwrite: bool,
) -> dict[str, Any]:
    _prepare_export_dir(export_dir, allow_overwrite=allow_overwrite)
    copied_files = _copy_passthrough_files(source_dir, export_dir)
    export_stats = _rewrite_model_weights(source_dir, export_dir)
    return {
        "source_dir": str(source_dir),
        "export_dir": str(export_dir),
        "copied_files": copied_files,
        "excluded_files": sorted(EXTRA_EXCLUDED_FILES),
        **export_stats,
    }


def _wait_for_server(
    base_url: str,
    timeout_s: int,
    *,
    server_proc: subprocess.Popen,
) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        if server_proc.poll() is not None:
            raise RuntimeError(
                "Server exited before becoming ready "
                f"(returncode={server_proc.returncode}). Check terminal output above."
            )
        try:
            resp = requests.get(base_url + "/health", timeout=5)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(
        f"Server did not become ready within {timeout_s}s. Check terminal output above."
    )


def _shutdown_server(server_proc: subprocess.Popen) -> None:
    if server_proc.poll() is not None:
        return
    server_proc.send_signal(signal.SIGTERM)
    try:
        server_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server_proc.kill()
        server_proc.wait(timeout=30)


def _send_generate_request(
    base_url: str,
    text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_s: int,
) -> dict[str, Any]:
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": text,
            "sampling_params": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    out = resp.json()
    return out if isinstance(out, dict) else out[0]


def _launch_server(
    *,
    target_model: str,
    draft_model_path: str,
    mode: dict[str, Any],
    port: int,
    tp_size: int,
    attention_backend: str,
    mem_fraction_static: float,
    max_running_requests: int,
    cuda_graph_max_bs: int | None,
    disable_cuda_graph: bool,
    flashinfer_workspace_size_mb: int | None,
) -> tuple[subprocess.Popen, list[str]]:
    env = os.environ.copy()
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    if flashinfer_workspace_size_mb is not None:
        env["SGLANG_FLASHINFER_WORKSPACE_SIZE"] = str(
            flashinfer_workspace_size_mb * 1024 * 1024
        )

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        target_model,
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        draft_model_path,
        "--tp-size",
        str(tp_size),
        "--attention-backend",
        attention_backend,
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--max-running-requests",
        str(max_running_requests),
        "--port",
        str(port),
        "--trust-remote-code",
    ]

    if cuda_graph_max_bs is not None:
        cmd.extend(["--cuda-graph-max-bs", str(cuda_graph_max_bs)])

    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")

    if not mode["dynamic_vbs"]:
        cmd.append("--no-speculative-dflash-dynamic-vbs")
    else:
        cmd.extend(
            [
                "--speculative-dflash-dynamic-vbs-predictor",
                str(mode["predictor"]),
            ]
        )

    proc = subprocess.Popen(
        cmd,
        cwd="/root/sglang_local",
        env=env,
    )
    return proc, cmd


def _make_prompts(
    *,
    benchmark_module,
    tokenizer,
    dataset_name: str,
    num_prompts: int,
    max_concurrency: int,
    enable_thinking: bool,
) -> list[str]:
    dataset = benchmark_module.load_and_process_dataset(dataset_name)
    prompts: list[str] = []
    total = num_prompts + max_concurrency
    for i in range(total):
        item = dataset[i % len(dataset)]
        user_content = item["turns"][0]
        prompts.append(
            benchmark_module._apply_chat_template(
                tokenizer,
                [{"role": "user", "content": user_content}],
                enable_thinking,
            )
        )
    return prompts


def _run_single_benchmark(
    *,
    base_url: str,
    prompts: list[str],
    num_prompts: int,
    concurrency: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_s: int,
) -> dict[str, Any]:
    try:
        requests.get(base_url + "/flush_cache", timeout=60).raise_for_status()
    except Exception:
        pass

    warmup_size = max(concurrency, 1)
    warmup_prompts = prompts[:warmup_size]
    run_prompts = prompts[warmup_size : warmup_size + num_prompts]

    if warmup_prompts:
        with ThreadPoolExecutor(max_workers=warmup_size) as pool:
            list(
                pool.map(
                    lambda prompt: _send_generate_request(
                        base_url,
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        timeout_s=timeout_s,
                    ),
                    warmup_prompts,
                )
            )

    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_lengths: list[float] = []

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _send_generate_request,
                base_url,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                timeout_s=timeout_s,
            ): idx
            for idx, prompt in enumerate(run_prompts)
        }
        for fut in as_completed(futures):
            out = fut.result()
            meta = out.get("meta_info", {}) or {}
            total_tokens += int(meta.get("completion_tokens", 0))
            spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
            if "spec_accept_length" in meta:
                try:
                    spec_accept_lengths.append(float(meta["spec_accept_length"]))
                except (TypeError, ValueError):
                    pass

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)
    return {
        "latency_s": latency,
        "output_tokens": total_tokens,
        "throughput_tok_s": toks_per_s,
        "spec_verify_ct": spec_verify_ct_sum,
        "accept_length_mean": (
            statistics.mean(spec_accept_lengths) if spec_accept_lengths else None
        ),
    }


def _render_tsv(results: list[dict[str, Any]]) -> str:
    headers = [
        "dataset",
        "mode",
        "concurrency",
        "throughput_tok_s",
        "latency_s",
        "output_tokens",
        "accept_length_mean",
        "spec_verify_ct",
    ]
    lines = ["\t".join(headers)]
    for row in results:
        lines.append(
            "\t".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("mode", "")),
                    str(row.get("concurrency", "")),
                    str(row.get("throughput_tok_s", "")),
                    str(row.get("latency_s", "")),
                    str(row.get("output_tokens", "")),
                    str(row.get("accept_length_mean", "")),
                    str(row.get("spec_verify_ct", "")),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _render_summary(
    *,
    target_model: str,
    source_dir: str,
    export_dir: str,
    export_stats: dict[str, Any],
    results: list[dict[str, Any]],
) -> str:
    lines = [
        f"Target model: {target_model}",
        f"Source draft dir: {source_dir}",
        f"Export draft dir: {export_dir}",
        f"Exported size: {_format_bytes(int(export_stats['total_size_bytes']))}",
        f"Removed keys: {', '.join(export_stats['removed_keys'])}",
        "",
        "Results:",
    ]
    for row in results:
        accept = row["accept_length_mean"]
        accept_rendered = f"{accept:.4f}" if accept is not None else "n/a"
        lines.append(
            f"- {row['dataset']} | {row['mode']} | c={row['concurrency']} | "
            f"{row['throughput_tok_s']:.2f} tok/s | latency={row['latency_s']:.2f}s | "
            f"accept_length={accept_rendered}"
        )
    return "\n".join(lines) + "\n"


def _validate_inputs(
    *,
    datasets: tuple[str, ...],
    concurrencies: tuple[int, ...],
    num_prompts: int,
    max_new_tokens: int,
) -> None:
    if not datasets:
        raise ValueError("At least one dataset must be provided.")
    if not concurrencies:
        raise ValueError("At least one concurrency value must be provided.")
    if any(concurrency <= 0 for concurrency in concurrencies):
        raise ValueError(f"Concurrency must be positive, got {concurrencies}.")
    if num_prompts <= 0:
        raise ValueError(f"num_prompts must be positive, got {num_prompts}.")
    if max_new_tokens <= 0:
        raise ValueError(
            f"max_new_tokens must be positive, got {max_new_tokens}."
        )


def _resolve_modes(mode_names: tuple[str, ...]) -> list[dict[str, Any]]:
    if not mode_names:
        raise ValueError("At least one mode must be provided.")
    unknown = [name for name in mode_names if name not in MODE_CONFIG_BY_NAME]
    if unknown:
        raise ValueError(
            f"Unknown mode(s): {unknown}. Available: {sorted(MODE_CONFIG_BY_NAME)}"
        )
    return [MODE_CONFIG_BY_NAME[name] for name in mode_names]


def _make_job_root(
    *,
    target_model: str,
    source_draft_dir: str,
    job_dir: str | None,
) -> Path:
    if job_dir is not None:
        path = Path(job_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    source_dir = _resolve_source_checkpoint_dir(Path(source_draft_dir), target_model)
    target_slug = target_model.split("/")[-1]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = Path(WORKSPACE_VOLUME_MOUNT) / "jobs" / (
        f"{_slugify(target_slug)}-modal-bench-{timestamp}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_mode_on_existing_export(
    *,
    target_model: str,
    source_dir: Path,
    export_dir: Path,
    mode: dict[str, Any],
    mode_job_dir: Path,
    datasets: tuple[str, ...],
    concurrencies: tuple[int, ...],
    num_prompts: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    enable_thinking: bool,
    tp_size: int,
    attention_backend: str,
    mem_fraction_static: float,
    max_running_requests: int,
    cuda_graph_max_bs: int | None,
    disable_cuda_graph: bool,
    flashinfer_workspace_size_mb: int | None,
    port: int,
    timeout_s: int,
    server_timeout_s: int,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    sys.path.insert(0, "/root/sglang_local")
    import benchmark as benchmark_module

    mode_job_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
    prompts_by_dataset = {
        dataset: _make_prompts(
            benchmark_module=benchmark_module,
            tokenizer=tokenizer,
            dataset_name=dataset,
            num_prompts=num_prompts,
            max_concurrency=max(concurrencies),
            enable_thinking=enable_thinking,
        )
        for dataset in datasets
    }

    base_url = f"http://127.0.0.1:{port}"
    server_proc, server_cmd = _launch_server(
        target_model=target_model,
        draft_model_path=str(export_dir),
        mode=mode,
        port=port,
        tp_size=tp_size,
        attention_backend=attention_backend,
        mem_fraction_static=mem_fraction_static,
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        disable_cuda_graph=disable_cuda_graph,
        flashinfer_workspace_size_mb=flashinfer_workspace_size_mb,
    )

    results: list[dict[str, Any]] = []
    try:
        _wait_for_server(
            base_url,
            timeout_s=server_timeout_s,
            server_proc=server_proc,
        )
        for dataset in datasets:
            prompts = prompts_by_dataset[dataset]
            for concurrency in concurrencies:
                metrics = _run_single_benchmark(
                    base_url=base_url,
                    prompts=prompts,
                    num_prompts=num_prompts,
                    concurrency=concurrency,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    timeout_s=timeout_s,
                )
                results.append(
                    {
                        "dataset": dataset,
                        "mode": mode["name"],
                        "concurrency": int(concurrency),
                        "server_cmd": " ".join(server_cmd),
                        **metrics,
                    }
                )
    finally:
        _shutdown_server(server_proc)
        workspace_volume.commit()

    tsv_text = _render_tsv(results)
    summary_text = _render_combined_summary(
        label=mode["name"],
        target_model=target_model,
        export_draft_dir=str(export_dir),
        source_draft_dir=str(source_dir),
        results=results,
    )

    results_json_path = mode_job_dir / "results.json"
    results_tsv_path = mode_job_dir / "results.tsv"
    summary_path = mode_job_dir / "summary.txt"

    results_json_path.write_text(
        json.dumps(
            {
                "target_model": target_model,
                "source_draft_dir": str(source_dir),
                "export_draft_dir": str(export_dir),
                "mode": mode["name"],
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    results_tsv_path.write_text(tsv_text)
    summary_path.write_text(summary_text)
    workspace_volume.commit()

    return {
        "job_dir": str(mode_job_dir),
        "source_draft_dir": str(source_dir),
        "export_draft_dir": str(export_dir),
        "mode": mode["name"],
        "results_json_path": str(results_json_path),
        "results_tsv_path": str(results_tsv_path),
        "summary_path": str(summary_path),
        "results": results,
        "results_tsv": tsv_text,
        "summary": summary_text,
    }


def _render_combined_summary(
    *,
    label: str,
    target_model: str,
    export_draft_dir: str,
    source_draft_dir: str,
    results: list[dict[str, Any]],
) -> str:
    lines = [
        f"Run label: {label}",
        f"Target model: {target_model}",
        f"Source draft dir: {source_draft_dir}",
        f"Export draft dir: {export_draft_dir}",
        "",
        "Results:",
    ]
    for row in results:
        accept = row["accept_length_mean"]
        accept_rendered = f"{accept:.4f}" if accept is not None else "n/a"
        lines.append(
            f"- {row['mode']} | {row['dataset']} | c={row['concurrency']} | "
            f"{row['throughput_tok_s']:.2f} tok/s | latency={row['latency_s']:.2f}s | "
            f"accept_length={accept_rendered}"
        )
    return "\n".join(lines) + "\n"


@app.function(
    gpu=None,
    timeout=6 * 60 * 60,
    image=image,
    secrets=[modal.Secret.from_name(HUGGINGFACE_SECRET_NAME)],
    volumes={
        SOURCE_VOLUME_MOUNT: source_volume,
        WORKSPACE_VOLUME_MOUNT: workspace_volume,
    },
)
def prepare_benchmark_assets(
    target_model: str = DEFAULT_TARGET_MODEL,
    source_draft_dir: str = SOURCE_VOLUME_MOUNT,
    allow_overwrite_export: bool = True,
    job_dir: str | None = None,
) -> dict[str, Any]:
    source_dir = _resolve_source_checkpoint_dir(Path(source_draft_dir), target_model)
    job_root = _make_job_root(
        target_model=target_model,
        source_draft_dir=source_draft_dir,
        job_dir=job_dir,
    )
    export_dir = job_root / "draft-export"

    export_stats = _export_draft_checkpoint(
        source_dir=source_dir,
        export_dir=export_dir,
        allow_overwrite=allow_overwrite_export,
    )
    workspace_volume.commit()

    return {
        "job_dir": str(job_root),
        "source_draft_dir": str(source_dir),
        "export_draft_dir": str(export_dir),
        "export_stats": export_stats,
    }


@app.function(
    gpu=DEFAULT_GPU,
    timeout=12 * 60 * 60,
    image=image,
    secrets=[modal.Secret.from_name(HUGGINGFACE_SECRET_NAME)],
    volumes={
        SOURCE_VOLUME_MOUNT: source_volume,
        WORKSPACE_VOLUME_MOUNT: workspace_volume,
    },
)
def run_mode_benchmark(
    target_model: str = DEFAULT_TARGET_MODEL,
    source_draft_dir: str = SOURCE_VOLUME_MOUNT,
    export_draft_dir: str | None = None,
    mode_name: str = "no_dynamic",
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    concurrencies: tuple[int, ...] = DEFAULT_CONCURRENCIES,
    num_prompts: int = DEFAULT_NUM_PROMPTS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    enable_thinking: bool = False,
    tp_size: int = DEFAULT_TP_SIZE,
    attention_backend: str = DEFAULT_ATTENTION_BACKEND,
    mem_fraction_static: float = DEFAULT_MEM_FRACTION_STATIC,
    max_running_requests: int = DEFAULT_MAX_RUNNING_REQUESTS,
    cuda_graph_max_bs: int | None = DEFAULT_CUDA_GRAPH_MAX_BS,
    disable_cuda_graph: bool = False,
    flashinfer_workspace_size_mb: int | None = DEFAULT_FLASHINFER_WORKSPACE_SIZE_MB,
    port: int = DEFAULT_PORT,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    server_timeout_s: int = DEFAULT_SERVER_TIMEOUT_S,
    job_dir: str | None = None,
) -> dict[str, Any]:
    _validate_inputs(
        datasets=datasets,
        concurrencies=concurrencies,
        num_prompts=num_prompts,
        max_new_tokens=max_new_tokens,
    )
    mode = _resolve_modes((mode_name,))[0]
    source_dir = _resolve_source_checkpoint_dir(Path(source_draft_dir), target_model)
    if export_draft_dir is None:
        raise ValueError("export_draft_dir must be provided for run_mode_benchmark.")
    export_dir = Path(export_draft_dir)
    if not export_dir.exists():
        raise FileNotFoundError(f"Export dir does not exist: {export_dir}")

    job_root = _make_job_root(
        target_model=target_model,
        source_draft_dir=source_draft_dir,
        job_dir=job_dir,
    )
    mode_job_dir = job_root / mode_name
    return _run_mode_on_existing_export(
        target_model=target_model,
        source_dir=source_dir,
        export_dir=export_dir,
        mode=mode,
        mode_job_dir=mode_job_dir,
        datasets=datasets,
        concurrencies=concurrencies,
        num_prompts=num_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        enable_thinking=enable_thinking,
        tp_size=tp_size,
        attention_backend=attention_backend,
        mem_fraction_static=mem_fraction_static,
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        disable_cuda_graph=disable_cuda_graph,
        flashinfer_workspace_size_mb=flashinfer_workspace_size_mb,
        port=port,
        timeout_s=timeout_s,
        server_timeout_s=server_timeout_s,
    )


@app.function(
    gpu=DEFAULT_GPU,
    timeout=12 * 60 * 60,
    image=image,
    secrets=[modal.Secret.from_name(HUGGINGFACE_SECRET_NAME)],
    volumes={
        SOURCE_VOLUME_MOUNT: source_volume,
        WORKSPACE_VOLUME_MOUNT: workspace_volume,
    },
)
def run_single_gpu_suite(
    target_model: str = DEFAULT_TARGET_MODEL,
    source_draft_dir: str = SOURCE_VOLUME_MOUNT,
    export_draft_dir: str | None = None,
    mode_names: tuple[str, ...] = DEFAULT_MODE_NAMES,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    concurrencies: tuple[int, ...] = DEFAULT_CONCURRENCIES,
    num_prompts: int = DEFAULT_NUM_PROMPTS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    enable_thinking: bool = False,
    tp_size: int = DEFAULT_TP_SIZE,
    attention_backend: str = DEFAULT_ATTENTION_BACKEND,
    mem_fraction_static: float = DEFAULT_MEM_FRACTION_STATIC,
    max_running_requests: int = DEFAULT_MAX_RUNNING_REQUESTS,
    cuda_graph_max_bs: int | None = DEFAULT_CUDA_GRAPH_MAX_BS,
    disable_cuda_graph: bool = False,
    flashinfer_workspace_size_mb: int | None = DEFAULT_FLASHINFER_WORKSPACE_SIZE_MB,
    port: int = DEFAULT_PORT,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    server_timeout_s: int = DEFAULT_SERVER_TIMEOUT_S,
    job_dir: str | None = None,
) -> dict[str, Any]:
    _validate_inputs(
        datasets=datasets,
        concurrencies=concurrencies,
        num_prompts=num_prompts,
        max_new_tokens=max_new_tokens,
    )
    modes = _resolve_modes(mode_names)
    source_dir = _resolve_source_checkpoint_dir(Path(source_draft_dir), target_model)
    if export_draft_dir is None:
        raise ValueError("export_draft_dir must be provided for run_single_gpu_suite.")
    export_dir = Path(export_draft_dir)
    if not export_dir.exists():
        raise FileNotFoundError(f"Export dir does not exist: {export_dir}")

    job_root = _make_job_root(
        target_model=target_model,
        source_draft_dir=source_draft_dir,
        job_dir=job_dir,
    )

    mode_results: list[dict[str, Any]] = []
    flat_results: list[dict[str, Any]] = []
    for mode in modes:
        result = _run_mode_on_existing_export(
            target_model=target_model,
            source_dir=source_dir,
            export_dir=export_dir,
            mode=mode,
            mode_job_dir=job_root / mode["name"],
            datasets=datasets,
            concurrencies=concurrencies,
            num_prompts=num_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            tp_size=tp_size,
            attention_backend=attention_backend,
            mem_fraction_static=mem_fraction_static,
            max_running_requests=max_running_requests,
            cuda_graph_max_bs=cuda_graph_max_bs,
            disable_cuda_graph=disable_cuda_graph,
            flashinfer_workspace_size_mb=flashinfer_workspace_size_mb,
            port=port,
            timeout_s=timeout_s,
            server_timeout_s=server_timeout_s,
        )
        mode_results.append(result)
        flat_results.extend(result["results"])

    summary_text = _render_combined_summary(
        label="single_gpu",
        target_model=target_model,
        export_draft_dir=str(export_dir),
        source_draft_dir=str(source_dir),
        results=flat_results,
    )
    tsv_text = _render_tsv(flat_results)

    results_json_path = job_root / "results.json"
    results_tsv_path = job_root / "results.tsv"
    summary_path = job_root / "summary.txt"
    results_json_path.write_text(
        json.dumps(
            {
                "target_model": target_model,
                "source_draft_dir": str(source_dir),
                "export_draft_dir": str(export_dir),
                "mode_results": mode_results,
                "results": flat_results,
            },
            indent=2,
        )
        + "\n"
    )
    results_tsv_path.write_text(tsv_text)
    summary_path.write_text(summary_text)
    workspace_volume.commit()

    return {
        "job_dir": str(job_root),
        "source_draft_dir": str(source_dir),
        "export_draft_dir": str(export_dir),
        "mode_results": mode_results,
        "results": flat_results,
        "results_json_path": str(results_json_path),
        "results_tsv_path": str(results_tsv_path),
        "summary_path": str(summary_path),
        "results_tsv": tsv_text,
        "summary": summary_text,
    }


@app.local_entrypoint()
def main(
    datasets: str = "gsm8k",
    concurrencies: str = "1,8,32,64",
    target_model: str = DEFAULT_TARGET_MODEL,
    source_draft_dir: str = SOURCE_VOLUME_MOUNT,
    num_prompts: int = DEFAULT_NUM_PROMPTS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    enable_thinking: bool = False,
    tp_size: int = DEFAULT_TP_SIZE,
    attention_backend: str = DEFAULT_ATTENTION_BACKEND,
    mem_fraction_static: float = DEFAULT_MEM_FRACTION_STATIC,
    max_running_requests: int = DEFAULT_MAX_RUNNING_REQUESTS,
    cuda_graph_max_bs: int | None = DEFAULT_CUDA_GRAPH_MAX_BS,
    disable_cuda_graph: bool = False,
    flashinfer_workspace_size_mb: int | None = DEFAULT_FLASHINFER_WORKSPACE_SIZE_MB,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    server_timeout_s: int = DEFAULT_SERVER_TIMEOUT_S,
    allow_overwrite_export: bool = True,
    mode_names: str = ",".join(DEFAULT_MODE_NAMES),
    execution_mode: str = DEFAULT_EXECUTION_MODE,
) -> None:
    dataset_list = tuple(_parse_csv_str(datasets))
    concurrency_list = tuple(_parse_csv_int(concurrencies))
    mode_name_list = tuple(_parse_csv_str(mode_names))
    _validate_inputs(
        datasets=dataset_list,
        concurrencies=concurrency_list,
        num_prompts=num_prompts,
        max_new_tokens=max_new_tokens,
    )
    modes = _resolve_modes(mode_name_list)

    if execution_mode not in {"spawn", "single_gpu", "both"}:
        raise ValueError(
            "execution_mode must be one of {'spawn', 'single_gpu', 'both'}, "
            f"got {execution_mode!r}."
        )

    assets = prepare_benchmark_assets.remote(
        target_model=target_model,
        source_draft_dir=source_draft_dir,
        allow_overwrite_export=allow_overwrite_export,
    )
    job_root = assets["job_dir"]
    export_draft_dir = assets["export_draft_dir"]
    resolved_source_draft_dir = assets["source_draft_dir"]

    all_outputs: dict[str, Any] = {
        "assets": assets,
        "execution_mode": execution_mode,
        "parallel_spawn": None,
        "single_gpu": None,
    }

    if execution_mode in {"spawn", "both"}:
        spawn_calls = [
            run_mode_benchmark.spawn(
                target_model=target_model,
                source_draft_dir=resolved_source_draft_dir,
                export_draft_dir=export_draft_dir,
                mode_name=mode["name"],
                datasets=dataset_list,
                concurrencies=concurrency_list,
                num_prompts=num_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
                tp_size=tp_size,
                attention_backend=attention_backend,
                mem_fraction_static=mem_fraction_static,
                max_running_requests=max_running_requests,
                cuda_graph_max_bs=cuda_graph_max_bs,
                disable_cuda_graph=disable_cuda_graph,
                flashinfer_workspace_size_mb=flashinfer_workspace_size_mb,
                timeout_s=timeout_s,
                server_timeout_s=server_timeout_s,
                job_dir=str(Path(job_root) / "parallel_spawn"),
            )
            for mode in modes
        ]
        spawn_results = list(modal.FunctionCall.gather(*spawn_calls))
        parallel_results = [
            row for item in spawn_results for row in item["results"]
        ]
        parallel_summary = _render_combined_summary(
            label="parallel_spawn",
            target_model=target_model,
            export_draft_dir=export_draft_dir,
            source_draft_dir=resolved_source_draft_dir,
            results=parallel_results,
        )
        all_outputs["parallel_spawn"] = {
            "mode_results": spawn_results,
            "results": parallel_results,
            "summary": parallel_summary,
            "results_tsv": _render_tsv(parallel_results),
        }

    if execution_mode in {"single_gpu", "both"}:
        all_outputs["single_gpu"] = run_single_gpu_suite.remote(
            target_model=target_model,
            source_draft_dir=resolved_source_draft_dir,
            export_draft_dir=export_draft_dir,
            mode_names=mode_name_list,
            datasets=dataset_list,
            concurrencies=concurrency_list,
            num_prompts=num_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            tp_size=tp_size,
            attention_backend=attention_backend,
            mem_fraction_static=mem_fraction_static,
            max_running_requests=max_running_requests,
            cuda_graph_max_bs=cuda_graph_max_bs,
            disable_cuda_graph=disable_cuda_graph,
            flashinfer_workspace_size_mb=flashinfer_workspace_size_mb,
            timeout_s=timeout_s,
            server_timeout_s=server_timeout_s,
            job_dir=str(Path(job_root) / "single_gpu"),
        )

    target_slug = target_model.split("/")[-1]
    local_prefix = f"modal_benchmark_{_slugify(target_slug)}"
    Path(f"{local_prefix}.results.json").write_text(
        json.dumps(all_outputs, indent=2) + "\n"
    )

    summary_parts: list[str] = []
    if all_outputs["parallel_spawn"] is not None:
        Path(f"{local_prefix}.parallel_spawn.summary.txt").write_text(
            all_outputs["parallel_spawn"]["summary"]
        )
        Path(f"{local_prefix}.parallel_spawn.results.tsv").write_text(
            all_outputs["parallel_spawn"]["results_tsv"]
        )
        summary_parts.append(all_outputs["parallel_spawn"]["summary"])
        print(all_outputs["parallel_spawn"]["summary"])
    if all_outputs["single_gpu"] is not None:
        Path(f"{local_prefix}.single_gpu.summary.txt").write_text(
            all_outputs["single_gpu"]["summary"]
        )
        Path(f"{local_prefix}.single_gpu.results.tsv").write_text(
            all_outputs["single_gpu"]["results_tsv"]
        )
        summary_parts.append(all_outputs["single_gpu"]["summary"])
        print(all_outputs["single_gpu"]["summary"])

    Path(f"{local_prefix}.summary.txt").write_text("\n".join(summary_parts))
    print(f"Saved local summary to {local_prefix}.summary.txt")
    print(f"Saved local JSON to {local_prefix}.results.json")
