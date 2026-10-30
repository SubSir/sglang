# Agentic workload: per-layer vs shared context K/V (Qwen3.6-35B-A3B)

The agent-session serving comparison between a 6-layer drafter with per-layer context K/V (24 KiB/token) and one with a single shared context K/V (4 KiB/token), both block 8, at an equal 27 GiB cache budget on one B300.

## Files

| file | what |
|---|---|
| `sglang_shared_kv.patch` | Changes to stock SGLang `c1c287a42928495166e14dbcef31b9e1f1dac199` (the `lmsysorg/sglang:nightly-dev-cu13-20261021-c1c287a4` image). The draft pool stores one layer when the draft config has `dflash_config.shared_context_kv: "fused"`, and every draft layer reads and writes it. Two new kernel files, `dflash_boundary*.py`, are imported by `models/dflash.py`. |
| `lane_agent_b300.sh` | One lane: starts the server in podman with the patched files, checks that both pools came out at the requested size, runs the client, tears down. |
| `bench_agent2.py` | Agent-session client. Runs 192 sessions × 5 rounds (768 follow-ups). Each request has a 3,800-token system prompt, 1,000-token tool outputs and 48 output tokens. T=0, 32 requests in flight. |
| `agent_sizing_b300.py` | Computes each arm's pool sizes for a cache budget: N (int8 mamba checkpoint slots) and P (full-attention tokens). |
| `b300_rec_base_n450_s192_o48.jsonl` | Request schedule recorded once by the base arm (`RECORD=1`). Every lane replays it, so both arms see byte-identical prompts in the same order. |

## Patching

The lane copies an overlay directory (`$OVL/sglang_pkg/`) over the image's `site-packages/sglang`. To build that directory from the patch:

```bash
# copy the four modified files from the image's site-packages/sglang into ovl/sglang/ (same relative paths):
#   srt/speculative/dflash_worker_v2.py  srt/models/dflash.py  srt/mem_cache/kv_cache_configurator.py
#   srt/model_executor/model_runner_components/spec_aux_hidden_state.py
patch -p2 -d ovl < sglang_shared_kv.patch   # patch paths are b/python/sglang/...; also creates the two new kernel files
mv ovl/sglang ovl/sglang_pkg
```

Alternatively, apply the patch with `-p1` to a checkout of SGLang at the commit above.

## Running

The lane expects `~/ev/hf` (HF cache holding `Qwen/Qwen3.6-35B-A3B`) and the drafters at `~/ev/drafts/q36-dflash2` (per-layer) and `~/ev/drafts/q36-fusekv` (shared). Copy `bench_agent2.py` into `~/ev/agent/`. Drafter checkpoints: HF `SubSir/dflash2-paper-ckpt-backup`, `runs/q36-{dflash2,fusekv}-L6-b8-kl/draft-epoch-5/serving-bf16`.

The 27 GiB sizes come from `python agent_sizing_b300.py 1300 bf16:27`: base N 380 / P 493,952, shared N 583 / P 757,888.

```bash
R=$PWD/b300_rec_base_n450_s192_o48.jsonl
REPLAY=$R OUTDIR=out/B1 TAG=r1_ bash lane_agent_b300.sh 0 base   380 493952
REPLAY=$R OUTDIR=out/B1 TAG=r1_ bash lane_agent_b300.sh 0 shared 583 757888
```

The reported runs used 8 batches. Each batch ran four lanes in the order base, shared, shared, base, giving two pairs, then one extra base lane as an A/A check. The batches were split across two cards, for 16 pairs in total. Each lane's last client line is a JSON summary: resumed follow-ups, mean and p90 TTFT, TPOT, recomputed tokens and accept length.

Result: base resumed 396 of 768 follow-ups at 212.7 ms mean TTFT. Shared resumed 711 at 169.7 ms, a TTFT change of −20.2 ± 0.5%, with the shared arm faster in 16 of 16 pairs.
