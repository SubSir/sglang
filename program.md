# autodflash

Autonomous LLM-driven implementation of dynamic block size for DFlash verification on SGLang.

## Setup

Work with the user to:

1. **Agree on a run tag** (e.g. `mar5`). Branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files** for full context:
   - `python/sglang/srt/speculative/dflash_info.py` — verify input, verification logic (`DFlashVerifyInput.verify()`, `DFlashVerifyInput.prepare_for_verify()`)
   - `python/sglang/srt/speculative/dflash_utils.py` — acceptance length computation, probability calculations
   - `python/sglang/srt/speculative/dflash_worker.py` — draft preparation, server-side orchestration
4. **Initialize results.tsv**: Create with header row only. Baseline recorded after first run.
5. **Confirm and go**.

## Rules

- **Edit only** these files:
  - `python/sglang/srt/speculative/dflash_info.py`
  - `python/sglang/srt/speculative/dflash_utils.py`
  - `python/sglang/srt/speculative/dflash_worker.py`
  And other files you think will be helpful for implementing dynamic verification length.
- **Do not** modify `benchmark_sgl.py` or install packages.
- **Simplicity criterion**: All else equal, simpler is better. A tiny gain that adds ugly complexity → not worth it. Removing code for equal results → great outcome.
- **Do not invent alternative approaches.** The design is fully specified below. Your job is to implement it correctly and then optimize the implementation. Every commit must be an incremental step toward completing or improving this single design — never a different idea.

## Goal

Maximize `Throughput`.

DFlash is a parallel draft model (one forward pass per block, not autoregressive). It uses block size 16 for drafting, producing 15 draft tokens in one pass. However, when acceptance length is only 3–4, most draft tokens get rejected during verification. At high concurrency this wastes significant compute on verifying tokens that will be rejected.

The solution is dynamic block size: keep the draft forward pass at block size 16, but truncate each request's draft tokens before verification based on estimated acceptance length. This is an engineering task with a known design — not a research problem.

## Running experiments

All commands run from the **repo root** (`/home/zlab/workspace/jianc/inference-optimization/sglang/`).

Each run uses 1 GPU:

```bash
conda activate inference
python benchmark_sgl.py > run.log 2>&1
```

Always redirect output — never use tee or let output flood your context.

Extract results:
```bash
grep "^Throughput:\|^Accept length:" run.log
```

If grep is empty → crash. Run `tail -n 150 run.log` for the traceback.

**Timeout**: If a run exceeds 3 minutes, kill it (`pkill -f benchmark_sgl`) and treat as failure.

## Logging results

Append to `results.tsv` (tab-separated, untracked by git). 5 columns:

```
commit	accept_len	throughput	status	description
```

- status: `keep`, `discard`, or `crash`
- Use `0.000000` / `0.0` / `0` for crash metrics

## Implementation Design (fixed — do not change the approach)

### Core Idea

Forward the draft model with the full block size 16 as usual, but before submitting to verification, **truncate each request's draft tokens based on estimated acceptance length**. This avoids verifying tokens that are likely to be rejected.

### How to estimate acceptance length

After the draft forward pass in `_prepare_for_speculative_decoding()` (dflash_worker.py, ~line 649–673), you have draft logits of shape `[bs, draft_token_num, vocab_size]`. Compute the top-1 softmax probability (confidence) for each drafted token:

```python
probs = F.softmax(draft_logits, dim=-1)       # [bs, draft_token_num, vocab_size]
confidence = probs.max(dim=-1).values          # [bs, draft_token_num]
```

For each request, use the confidence as the estimated acceptance rate to calculate the acceptance length: i.e, 1+0.9+0.9*0.8+0.9*0.8*0.78..., then add a margin (e.g. +2). The estimated acceptance length is that position index. The margin makes truncation conservative — you keep a few extra tokens beyond the estimated acceptance point.

### CUDA graph constraint

The draft forward always runs with block size 16 (fixed shape, cuda-graphable via `TARGET_VERIFY` mode). Only the **verification step** sees variable-length blocks.

The verify forward in `DFlashVerifyInput.prepare_for_verify()` (dflash_info.py, ~line 178–251) builds attention masks and allocates KV cache. You cannot capture a separate CUDA graph for every possible block size — too many graphs. Instead:

- **Pre-capture CUDA graphs for a fixed set of block sizes**: {4, 6, 8, 10, 12, 14, 16}.
- **Round each request's dynamic block size up** to the nearest captured size.
- Different requests in the same batch may have different block sizes — the attention mask in `prepare_for_verify()` already supports per-request custom masks (lines ~234–251). You need to handle variable-length blocks within a batch by padding shorter blocks and masking the padded positions.

### Key code locations

| What | Where | Lines |
|------|-------|-------|
| Draft forward + sampling | `dflash_worker.py` → `_prepare_for_speculative_decoding()` | ~649–673 |
| Verify batch assembly | `dflash_info.py` → `DFlashVerifyInput.prepare_for_verify()` | ~178–251 |
| Verify execution | `dflash_info.py` → `DFlashVerifyInput.verify()` | ~312–501 |
| Acceptance length calc | `dflash_utils.py` → `compute_dflash_accept_len_and_bonus()` | greedy path |
| Softmax probabilities | `dflash_utils.py` → `compute_dflash_sampling_accept_len_and_bonus()` | ~554–598 |
| CUDA graph mask padding | `dflash_info.py` → `prepare_for_verify()` | ~296–309 |

## Phase 1: Get it working

The goal of Phase 1 is a correct end-to-end implementation. Do not optimize yet. Steps:

1. **Baseline run**: Run `benchmark_sgl.py` with no code changes. Record results.
2. **Add confidence estimation**: In `_prepare_for_speculative_decoding()`, compute per-request estimated acceptance length from draft logits as described above.
3. **Add truncation**: Before handing draft tokens to verification, truncate each request's tokens to `min(estimated_len + margin, 16)`. Round up to the nearest CUDA graph bucket size.
4. **Modify `prepare_for_verify()`**: Handle variable-length blocks within a batch — pad shorter blocks and mask padded positions in the attention mask.
5. **Modify `verify()`**: Ensure acceptance logic correctly handles the truncated/padded blocks.
6. **Run and debug** until the benchmark completes without crashing and produces valid throughput/acceptance numbers.

**Phase 1 is complete when**: the benchmark runs successfully with dynamic block sizes active and results.tsv has a non-crash entry with the implementation enabled. Throughput may be worse than baseline at this point — that is fine.

## Phase 2: Optimize the implementation

Once Phase 1 is complete, optimize to maximize throughput. Each commit should change **one thing** from this list:

- **Tune the margin**: Try values like +1, +2, +3, +4. Pick the one that maximizes throughput.
- **Tune bucket sizes**: Try different sets (e.g. {4, 8, 12, 16} vs {4, 6, 8, 10, 12, 14, 16} vs {2, 4, 8, 16}).
- **Reduce overhead**: Profile where the confidence computation and truncation logic add latency. Minimize tensor operations and memory allocations.
- **Reduce padding waste**: If most requests round up significantly, the buckets may be wrong. Check the distribution of dynamic sizes and adjust.
- **Simplify code**: If you can remove complexity without hurting throughput, do it.

**Do not** try alternative estimation methods, alternative truncation strategies, or any other "new idea." The design is fixed. You are tuning and polishing one implementation.

## The implementation & optimization loop

**First run**: Establish baseline by running `benchmark_sgl.py` as-is (no code changes).

Then LOOP:

1. **Make one incremental change** — a bug fix, a parameter tweak, or a code simplification. Never a new approach.
2. **Commit**: `git commit -am "description"`.
3. **Run**: `python benchmark_sgl.py > run.log 2>&1`
4. **Read results**: grep the metrics from run.log.
5. **Log** to results.tsv.
6. **Keep or revert**:
   - If `Throughput` improved → keep the commit, advance the branch.
   - If equal or worse → `git reset --hard` to previous good commit.
7. **Crashes**: Typo/easy fix → fix and re-run. If stuck on the same crash for 3 attempts → revert to last working commit and try a different incremental change to the same design.

## NEVER STOP

Once the loop begins, do NOT pause to ask the human anything. Do NOT ask "should I keep going?" — the human may be asleep and expects you to work until manually stopped. If throughput plateaus, try adjusting the margin, changing bucket sizes, reducing padding overhead, or simplifying the code. Review `results.tsv` to see which parameter values worked best and which didn't. Stay focused on this one design.