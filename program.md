# autodflash

Autonomous LLM-driven research to implement dynamic block size for DFlash on SGLang.

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
- **Do not** modify `benchmark_sgl.py` or install packages.
- **Simplicity criterion**: All else equal, simpler is better. A tiny gain that adds ugly complexity → not worth it. Removing code for equal results → great outcome.

## Goal

Maximize `Throughput`.

DFlash is a parallel draft model (one forward pass per block, not autoregressive). It uses block size 16 for drafting, producing 15 draft tokens in one pass. However, when acceptance length is only 3–4, most draft tokens get rejected during verification. At high concurrency this wastes significant compute on verifying tokens that will be rejected. The goal is to implement dynamic block size to reduce wasted verification.

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

## The experiment loop

**First run**: Establish baseline by running `benchmark_sgl.py` as-is (no code changes).

Then LOOP FOREVER:

1. **Implement** your change in the in-scope files listed above.
2. **Commit**: `git commit -am "description"`.
3. **Run**: `python benchmark_sgl.py > run.log 2>&1`
4. **Read results**: grep the metrics from run.log.
5. **Log** to results.tsv.
6. **Keep or revert**:
   - If `Throughput` improved → keep the commit, advance the branch.
   - If equal or worse → `git reset --hard` to previous good commit.
7. **Crashes**: Typo/easy fix → fix and re-run. Fundamentally broken idea → log crash, revert, move on.

## Implementation Detail

### Core Idea

Forward the draft model with the full block size 16 as usual, but before submitting to verification, **truncate each request's draft tokens based on estimated acceptance length**. This avoids verifying tokens that are likely to be rejected.

### How to estimate acceptance length

After the draft forward pass in `_prepare_for_speculative_decoding()` (dflash_worker.py, ~line 649–673), you have draft logits of shape `[bs, draft_token_num, vocab_size]`. Compute the top-1 softmax probability (confidence) for each drafted token:

```python
probs = F.softmax(draft_logits, dim=-1)       # [bs, draft_token_num, vocab_size]
confidence = probs.max(dim=-1).values          # [bs, draft_token_num]
```

For each request, scan left-to-right and find the first position where confidence drops below a threshold (e.g. 0.5). The estimated acceptance length is that position index. Add a margin (e.g. +2) to be conservative. The result is how many draft tokens to submit for verification per request.

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

## NEVER STOP

Once the loop begins, do NOT pause to ask the human anything. Do NOT ask "should I keep going?" — the human may be asleep and expects you to work indefinitely until manually stopped. If you run out of ideas, re-read `results.tsv` for patterns, re-read the source files for new angles, try combining insights, try radical changes. The loop runs until interrupted.
