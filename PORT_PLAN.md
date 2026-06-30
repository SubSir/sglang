# Dynamic-VBS → sglang v2 (overlap) port plan

## Goal
Port DFlash "dynamic verify block size" (truncate each request's draft tokens before
verify by estimated accept length) from the v1 non-overlap worker onto upstream sglang
v2 overlap worker. Target Qwen3-8B + z-lab/Qwen3-8B-DFlash-b16, flashinfer, B200.
Success: dyn tok/s >= no-dyn at conc=1 AND > no-dyn at conc=128, accept length ~unchanged.
Optimize using **sglang profiler** (torch profiler trace), NOT cuda-event timers.

## Worktrees
- THIS = /Users/.../sglang-dynamic-verify-v2  (branch dynamic-vbs-v2, base e0c0c0a45c =
  recent upstream w/ official Spec-V2 DFlash chain, 2026-06-27). Do all work here.
- v1 reference = ../sglang-dynamic-verify (branch dynamic-verify). The feature source.
- v2-tree ref = ../sglang-v2-tree (domino/tree on v2). Reference only.
Modal harness: modal_vbs_v2.py (dynamic-mode none|on|off).

## v2 base facts
- Worker: dflash_worker_v2.py `DFlashWorkerV2(BaseSpecWorker)`. Chain draft forward at
  ~1651-1764 → draft_hidden [bs, block_size, -1]; greedy sample block_size-1 tokens via
  `_greedy_sample_from_vocab_parallel_head` (~699-969).
- Verify: dflash_info.py `DFlashVerifyInput.prepare_for_verify`; accept computed in
  worker ~1766-2087 (Triton/python). Verify is FIXED block_size every step.
- Overlap KV: dflash_info_v2.py `DFlashDraftInputV2.prepare_for_decode` (~131-281)
  FUTURE-allocates 2*block_size headroom ahead of time (planning vs reserved lens).
- Target verify reuses target worker's standard cuda graph (fixed shape). No DFlash
  per-bucket verify graphs. No dynamic truncation anywhere.
- Draft model type auto-selected; z-lab b16 = plain chain (projector None).

## v1 feature spec (what to port) — confidence→VBS→truncate
1. server arg `speculative_dflash_dynamic_vbs` (BooleanOptionalAction, default True).
2. Worker init: buckets {4,6,8,12,block_size}; lookup table tpbs→bucket; margin=2;
   confidence_scale=0.5.
3. After draft forward, get per-token confidence = sigmoid(top1_logit - top2_logit)
   (fused via topk(2) in the greedy-sample head; add return_confidence path).
   scaled = sqrt(confidence); cum = cumprod(scaled, dim=1); est_accept = cum.sum(1).
   raw_vbs = ceil(est_accept + margin).clamp(buckets[0], block_size).int()  # [bs]
4. effective_tpbs = bucket_lookup[ceil(sum(vbs)/bs)]  (batch-level bucket).
   If all requests == effective_tpbs → rectangular verify (reuse fixed path, smaller).
   Else tight-pack: flat tokens/positions buf, vbs_cumsum, gather by (row,col) idx.
5. DFlashVerifyInput extra fields: per_request_vbs[bs], per_request_vbs_cumsum[bs+1],
   total_real_tokens. prepare_for_verify builds variable q_len causal mask + per-req KV
   alloc (end_offset = seq_len + per_req_vbs). generate_attn_arg_prefill: variable
   qo_indptr from vbs cumsum.
6. verify: unpack tight logits → [bs,max_vbs], accept clamped to per_req_vbs-1, free
   padding KV slots, gather committed hidden by same mask.
7. CUDA graph runner: capture target verify graph per bucket, key (bs, tokens_per_bs);
   `_get_dflash_tokens_per_bs` rounds batch draft_token_num up to bucket; replay picks it.

## v2-specific challenges (differences from v1)
A. Overlap future-allocation: prepare_for_decode runs BEFORE draft confidence is known,
   so it can't know per-req VBS. Simplest correct: keep allocating block_size headroom
   (as today), then dynamic VBS only shrinks the VERIFY forward + frees unused at commit.
   i.e. dynamic VBS truncates verify tokens/graph, not the KV pre-alloc. Free extra after.
B. Target verify cuda graph: v2 uses target worker graph (fixed). Need per-bucket verify
   graphs OR run verify eager for non-full buckets. The verify-compute SAVING is the whole
   point — must shrink the verify forward token count. Start: rectangular bucket only
   (all reqs same effective_tpbs) so target graph shape = bs*effective_tpbs; capture the
   bucket set. Tight-pack (mixed) is an optimization for later if needed.
C. Profiling: use SGLang /start_profile /stop_profile (torch profiler) at conc=1, compare
   dyn vs no-dyn per-step CPU vs GPU. Overlap should hide the VBS bookkeeping; verify the
   trace shows verify GPU time actually dropping.

## Strategy (lazy: smallest correct path first)
Step 1 DONE-ish: clean v2 base + modal harness. Confirm baseline builds/runs.
Step 2: add server arg + worker confidence/est_accept/effective_tpbs (RECTANGULAR ONLY:
   batch-level single bucket, all reqs truncated to effective_tpbs). This is the laziest
   variant — one bucket per batch, reuses fixed verify path at a smaller block size, no
   tight-pack, no per-req mask. Verify it runs and accept-len ~unchanged.
Step 3: per-bucket target verify cuda graph capture so the smaller block actually runs on
   graph (else eager verify may erase the win).
Step 4: profile conc=1; if still losing, add per-request tight-pack only if profile says
   the batch-level rounding wastes too much at higher conc.
Step 5: tune margin/buckets; loop sweep until success.

## Progress log
- v2 base e0c0c0a45c builds + serves on Modal B200 (image: strip rust grpc ext-module,
  upgrade pip; pure-python editable install). Harness modal_vbs_v2.py readiness loop
  fixed (503-during-warmup must still sleep).
- Implemented RECTANGULAR dynamic-VBS (commit pending): server arg
  speculative_dflash_dynamic_vbs (BooleanOptionalAction, default True); worker
  confidence via greedy-head topk(2) sigmoid(top1-top2); _dynamic_verify_len = bucket(
  ceil(mean(ceil(est_accept+margin)))); buckets {4,6,8,12,16}; gate bs>=dynamic_verify_min_bs
  (=2); truncate verify forward+accept+KV-materialize to verify_len, keep out_tokens
  block_size-wide; eager verify when verify_len<block_size (can_run_graph falls back).
  do_dynamic only for greedy. dyn-off path byte-identical to upstream.
- KNOWN tradeoff to measure: rectangular-mean caps accept for upper-half requests ->
  accept length may drop. If it drops much, escalate to per-request tight-packing (v1
  spec). KNOWN overhead: _dynamic_verify_len does one .item() host sync per step (picks
  the verify shape) -> serializes draft vs verify. Profile this; it's the prime target.

## Profiling (torch profiler, NOT cuda-event timers)
SGLang: env SGLANG_TORCH_PROFILER_DIR=/root/prof; POST /start_profile with
ProfileReq(num_steps=N, activities=["CPU","GPU"]) -> auto-stops after N forward steps,
writes chrome trace json. Metric: over the profiled window, GPU-busy = sum(kernel
durations); GPU-idle% = 1 - busy/wall. Compare dyn-on vs dyn-off idle% to expose the
host-sync bubble + eager-verify metadata CPU. Add `profile` entrypoint to modal harness.
