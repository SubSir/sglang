# DFlash Draft-Tree: Status, Findings & Blog Material

_Investigation date: 2026-06-27. Scope: the four goals — (1) upstream merge, (2) throughput /
backend / linear-attention confirmation + fix, (3) new-model tree-verify support, (4) competitive
analysis (JetSpec / DDTree) and a blog-ready draft-tree results summary._

This fork (`dflash_fixed`) implements **tree verification on top of DFlash** in the spec-v1 worker
(`python/sglang/srt/speculative/dflash_worker.py`), gated by `SGLANG_DFLASH_TREE_VERIFY=1` and
`--speculative-eagle-topk`. Upstream SGLang ships DFlash too, but **chain-only** (see Goal 1).

---

## TL;DR for the team / Slack thread

- **Our tree algorithm ≈ DDTree** (arXiv 2604.12989, `github.com/liranringel/ddtree`). Same chain→tree
  gain on a block-diffusion drafter under a fixed node budget, ancestor-only verify mask. DDTree is the
  closest published prior/concurrent work — **we must cite it and benchmark head-to-head on its budget
  grid {16,32,64,128,256,512,1024}**. Combining tree + **dynamic block size** is the differentiator
  Jian suggested (DDTree is fixed-budget; we can be adaptive).
- **JetSpec** (arXiv 2606.18394, `github.com/hao-ai-lab/JetSpec`) is a *trained parallel draft head* +
  tree, not "sloppy" in methodology (all baselines retrained on the same mixture, open MIT code +
  checkpoints), but its **headline 9.6× is best-cased**: batch-1, budget-256, B200, CUDA-graphed drafter.
  Its repo admits a naive run is ~2× slower than the headline. The "blocksize=32 uses the HF checkpoint
  without retraining" claim could **not** be substantiated — JetSpec fixes block size at 16 and states
  all baselines were retrained; the "32" in that thread is a *node budget*, not a block size. So the
  premise that they "cheated" on bs=32 looks wrong; the fair criticism is *best-case-only reporting*.
- **Backend premise is outdated** (Goal 2): it is **no longer true** that "only flashinfer supports the
  custom tree mask." In the current code, **flashinfer, fa3, fa4, and triton all consume
  `spec_info.custom_mask` in TARGET_VERIFY**; only **`trtllm_mha` does not**. Since the official DFlash
  recipe uses `trtllm_mha` for the *fastest chain decode*, the real tension is "tree verify forces you
  off trtllm_mha," not "tree verify only runs on flashinfer."
- **Linear-attention slowdown is real and architectural** (Goal 2/3): Qwen3.5/3.6, MiniMax-M2, Kimi-K2,
  Qwen3-Next, GLM use gated-delta-net / Mamba layers whose verify kernel is a **sequential recurrent
  scan over the draft block** (loop over T tokens), so a budget-N verify costs ≈N sequential steps *per
  linear layer* — independent of full-attention being parallel at the same budget. Trees make it worse
  (parent-state loads) and a linear layer **cannot express an arbitrary tree mask** (only ancestor
  chains). This is why "verify is much slower even at the same token budget" on Qwen3.5.

---

## ★ Measured draft-tree results (v1, CORRECT) — Qwen3-8B, B200, cuda graph ON

v1 tree verify is validated-correct (v2 port has an accept bug — see "v2 Tree-Verify Port").
Qwen3-8B + `z-lab/Qwen3-8B-DFlash-b16`, block 16, flashinfer, gsm8k, conc {1,8,32}, baseline + chain +
tree budgets {16,32,64}:

| config | accept (c1/c8/c32) | tok/s (c1/c8/c32) | speedup vs baseline (c1/c8/c32) |
|---|---|---|---|
| baseline (no spec) | — | 227 / 1,413 / 4,901 | 1× |
| chain (block 16) | 5.87 / 6.39 / 6.41 | 319 / 3,649 / 8,935 | **1.41 / 2.58 / 1.82** |
| tree b16 | 6.65 / 7.14 / 7.20 | 318 / 2,148 / 6,382 | 1.40 / 1.52 / 1.30 |
| tree b32 | 7.30 / 7.75 / 7.84 | 275 / 2,061 / 5,481 | 1.21 / 1.46 / 1.12 |
| tree b64 | **7.74 / 8.12 / 8.22** | 344 / 1,967 / 4,281 | 1.52 / 1.39 / 0.87 |

**Headline conclusions (the blog point):**
1. Chain DFLASH alone gives 1.4–2.6× over baseline.
2. **Tree raises accept length (6.4 → 8.2, +28%) but NOT throughput — and at serving concurrency it
   HURTS throughput** (conc=32: chain 8,935 > tree-b16 6,382 > tree-b64 4,281; tree-b64 is even slower
   than baseline, 0.87×). The bigger tree's verify FLOPs saturate the GPU at high batch, and the extra
   accept length doesn't pay for it. This quantifies the team's original "accept length up, throughput
   not" observation.
3. **Tree is a low-concurrency / latency-bound win**; at high concurrency use a small budget or fall back
   to chain. → **tree + dynamic block size** (big budget at low conc, small/chain at high conc) is the
   right design, matching the Slack discussion. mt-bench accept is lower (2.7–5.8, chattier/less draftable).

## ★★ v2 (upstream `dflash_worker_v2`) tree verify — FIXED + results

The tree-verify feature is now ported AND CORRECT on the upstream spec-v2 worker (branch
`dflash-tree-v2-port`), running on the official `lmsysorg/sglang:nightly-...20260627` image. **The bug
that made it wrong for ~20 cycles:** v2's verify KV slots are a standing per-req reservation
(`committed + 2*block_size`), so adjacent verify windows OVERLAP in `req_to_token`. The "no-move
re-pointing" of scattered tree-accept slots left an accepted node aliased into the NEXT verify window →
overwritten → KV corruption → non-lossless → accept stuck ~4 (< chain). **Fix = EAGLE v2's
`move_accept_tokens_to_target_kvcache`**: physically move the accepted tree-path KV to the contiguous
FRONT of each block (chain needs no move — its accepts are already front-aligned). Plus: decouple draft
block_size (16, the trained `-b16`) from the verify budget; size verify buffers/accept/logits-adjust by
the budget; run with `--disable-overlap-schedule` (spec-v2 overlap is topk=1-only by design). Validated:
Qwen3-8B tree_b16 3.23 → **7.05**, matching v1.

**Qwen3-8B (v2, flashinfer, cuda graph ON):**

| gsm8k | accept c1/c8/c32 | tok/s c1/c8/c32 |   | mt-bench accept c1/c8/c32 |
|---|---|---|---|---|
| chain | 5.79 / 6.37 / 6.38 | 904 / 4,571 / 10,579 | | 2.14 / 4.43 / 3.97 |
| tree b16 | 6.54 / 7.21 / 7.18 | 832 / 2,005 / 6,218 | | 2.66 / 5.16 / 4.70 |
| tree b32 | 7.23 / 7.33 / 7.36 | 797 / 1,049 / 4,074 | | 2.83 / 6.03 / 5.42 |
| tree b64 | 7.51 / 8.09 / 7.66 | 806 / 1,333 / 3,338 | | 2.97 / 6.21 / 5.87 |

**gemma-4-31B-it (v2, triton, cuda graph ON):**

| gsm8k | accept c1/c8/c32 | tok/s c1/c8/c32 |   | mt-bench accept c1/c8/c32 |
|---|---|---|---|---|
| chain | 6.92 / 7.12 / 7.33 | 382 / 1,721 / 3,305 | | 2.07 / 4.45 / 4.01 |
| tree b16 | 7.34 / 7.52 / 7.75 | **418** / 1,286 / 2,673 | | 2.39 / 4.78 / 4.36 |
| tree b32 | 7.91 / 7.75 / 7.89 | 426 / 1,183 / 2,154 | | 2.49 / 4.73 / 4.37 |
| tree b64 | 8.34 / 8.66 / 8.32 | 403 / 783 / 1,430 | | 2.65 / 5.68 / 5.04 |

**Qwen3.6-27B (v2, triton + triton_attn vision, cuda graph ON) — HYBRID GDN/mamba + tree:**

| gsm8k | accept c1/c8/c32 | tok/s c1/c8/c32 |   | mt-bench accept c1/c8/c32 |
|---|---|---|---|---|
| chain | 6.98 / 7.22 / 7.30 | 324 / 1,446 / 1,619 | | 2.58 / 3.38 / 4.45 |
| tree b16 | 7.79 / 7.92 / 8.14 | 376 / 1,124 / **1,868** | | 2.94 / 5.56 / 5.07 |
| tree b32 | 8.48 / 7.76 / 7.97 | 353 / 820 / 1,098 | | 3.15 / 5.48 / 5.00 |
| tree b64 | **8.63 / 8.77 / 9.03** | 271 / 476 / 533 | | 3.19 / 6.21 / 5.69 |

**Headline (v2, the requested setup — cuda graph ON, tree-vs-chain, all 3 models):**
- Tree raises accept length on ALL three (gsm8k conc1: Qwen3-8B 5.79→7.51, gemma 6.92→8.34, Qwen3.6
  6.98→8.63). **Mamba/hybrid (Qwen3.6) tree verify works correctly** — the GDN layers handle the tree mask
  and the mamba state commits at the deepest accepted node's block position (port fix).
- **Small dense model (Qwen3-8B):** tree costs throughput at high concurrency (conc32: chain 10.6k >
  tree-b64 3.3k tok/s) — latency-bound win only.
- **Large / hybrid models (gemma-31B, Qwen3.6-27B):** tree is a NET WIN at low concurrency on BOTH accept
  and throughput (gemma tree-b16 418 > chain 382 tok/s @c1; Qwen3.6 tree-b16 1,868 > chain 1,619 tok/s even
  @conc32). The verify overhead is small relative to the big/hybrid forward, so the extra accept pays off.
  → **bigger / hybrid models favor trees; small dense models + high concurrency favor chain or small
  budgets → dynamic block size is the right knob.**
- gpt-oss-120b dropped (per request; fa4-cute broken). Qwen3.6-27B is a VL model whose vision tower needs
  `--mm-attention-backend triton_attn` (fa4/cute broken, fa3 unsupported on Blackwell) — the tree-verify
  path itself is unaffected.

**Per-model server recipe (v2, what actually works on B200):**
- Qwen3-8B: `--attention-backend flashinfer`.
- gemma-4-31B-it: `--attention-backend triton` (gemma rejects flashinfer).
- Qwen3.6-27B: `--attention-backend triton --mm-attention-backend triton_attn --mamba-scheduler-strategy extra_buffer`.
- All: `--speculative-algorithm DFLASH`, `SGLANG_DFLASH_TREE_VERIFY=1`, `--speculative-eagle-topk 4`,
  `--speculative-num-draft-tokens <budget>`, `--speculative-dflash-block-size 16`, `--disable-overlap-schedule`,
  draft attention backend flashinfer, mem-fraction ~0.6 for the 27-31B (cuda-graph + tree-mask buffer headroom).

## Attention-backend capability matrix (from `docs/advanced_features/attention_backend.md`)

Upstream documents exactly which backends support the **custom tree mask** (the "Spec topk>1" column) and
the GDN target-verify path:

**MHA — "Spec topk>1" = tree-mask support:** FlashInfer ✅, FA3 ✅, FA4 ✅, Triton ✅, **TRTLLM MHA ❌**,
Torch-native/Flex/DualChunk/Wave/Intel ❌. (MLA backends: almost all ❌ for topk>1; only FA3/Triton ⚠️ at
page_size=1.) **GDN/linear-attn target verify: only Triton ✅** (CuTe DSL ❌). On Blackwell, hybrid-GDN
full-attn layers are restricted to `triton / trtllm_mha / fa4`.

This confirms our backend choices: gemma rejects flashinfer → must use **triton** (trtllm_mha is ❌ for
tree mask); Qwen3.5/3.6 linear layers need **triton** for target verify.

**Critical:** the doc states **Spec V2 (overlap scheduling, the `dflash_worker_v2` lineage) requires
`--speculative-eagle-topk 1` and only covers EAGLE/EAGLE3.** So upstream spec-v2 is **topk=1-only by
design** — tree (topk>1) is not a supported spec-v2 path. This is the root reason the v2 tree port fights
the framework, and why **v1 (non-overlap spec-v1) is the correct home for tree/topk>1 verify.**

## Goal 2 — Throughput, backends, and the linear-attention slowdown

### 2a. Which attention backends support the tree custom mask (CONFIRMED via code)

Tree verify builds a flat boolean "allow" mask (`DFlashVerifyInput.prepare_for_verify` →
`build_tree_kernel_efficient`, `dflash_info.py`) and the target-model attention backend must apply it
during `ForwardMode.TARGET_VERIFY`. Evidence:

| Backend | Consumes `spec_info.custom_mask` in verify? | Evidence |
|---|---|---|
| **flashinfer** (`FlashInferAttnBackend`) | ✅ yes | `flashinfer_backend.py` cuda-graph custom_mask path |
| **fa3 / fa4** (`FlashAttentionBackend`, `fa_impl_ver=4`) | ✅ yes | `flashattention_backend.py:367` and `:1930` — `mask = spec_info.custom_mask[mask_extraction_indices]...` (gated on `topk>1`) |
| **triton** (`TritonAttnBackend`) | ✅ yes | `triton_backend.py` `init_forward_metadata` reads `spec_info.custom_mask` |
| **trtllm_mha** (`TRTLLMHAAttnBackend`) | ❌ NO | verify path only builds causal metadata; comment "we only support topk = 1 for now" |

`fa4` is **not** a separate backend — it is `FlashAttentionBackend(runner, fa_impl_ver=4)`
(`attention_registry.py:146-150`), identical custom-mask handling as fa3.

> **Empirical caveat discovered while benchmarking:** in this fork's pinned environment, **fa4 is
> currently broken at runtime** — its CUTLASS-DSL kernel crashes with
> `TypeError: fmax() takes 2 positional arguments but 3 given` in `flash_attn/cute/softmax.py`
> (a `flash_attn` ↔ `nvidia-cutlass-dsl` version mismatch). This affects both the main and draft
> attention paths when set to fa4, so any fa4 sweep dies on the first decode. This is independent of the
> tree-mask question and is a strong reason the old fa4 throughput numbers looked bad/unusable. It is a
> **dependency-pinning bug** that the upstream merge (Goal 1) would resolve. For now, benchmark on
> **flashinfer** (robust, supports the tree mask). The bench script's hardcoded `fa4` draft backend on
> Blackwell is now overridable via `DFLASH_DRAFT_ATTN_BACKEND`.

**Conclusion:** the old belief ("only flashinfer; fa3/fa4 can't") is no longer accurate for this
codebase. The current correct statement: **tree verify runs on flashinfer / fa3 / fa4 / triton; it does
NOT run on `trtllm_mha`.** The `_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS` set in `dflash_utils.py`
(flashinfer, fa3, trtllm_mha, MLA variants) only governs the **chain** path, where those backends do
native causal verify and skip building a mask for speed; tree verify always forces `build_custom_mask=True`.

### 2b. Why throughput didn't improve much

Three compounding reasons, all consistent with the earlier observation:

1. **Tree verify is incompatible with the fastest decode kernel.** The official DFlash recipe runs the
   target on `trtllm_mha`. Tree verify needs the custom mask → you must drop to flashinfer/fa4, giving
   back part of the base-decode speed. Net throughput ≈ (bigger accept length) × (slightly slower kernel),
   so the throughput delta is smaller than the accept-length delta.
2. **Tree gains are a batch-1 phenomenon.** DDTree/JetSpec headline numbers are all concurrency=1. At
   serving concurrency (our old sweep ran conc=32), the GPU is already saturated by the batch, so wider
   per-request trees mostly add verify FLOPs without raising throughput — they can even *lower* it. The
   right way to show our win is **conc=1 (or very low) accept-length + per-request latency**, matching how
   DDTree/JetSpec report. (Our `modal_tree_vs_chain.py` runs conc=1 for exactly this reason.)
3. **Tree-build + mask overhead per step.** `build_tree_verify_tokens` + `build_tree_kernel_efficient`
   add CPU/GPU work each decode step. Mitigated by the fused topk=4 triton op
   (`triton_ops/dflash_tree_expand_topk.py`), but still nonzero; at high accept length it's amortized,
   at low accept length it can erase the gain.

**Actionable fixes / recommendations:**
- Report **accept length + low-concurrency speedup**, not high-conc throughput, as the headline (this is
  also what reviewers will expect vs DDTree/JetSpec).
- Keep `topk=4` (fused triton path) — other topk values fall back to the slower `torch.compile` loop.
- For the **chain** path, prefer `trtllm_mha` (fastest); only switch to fa4/flashinfer when tree is on.
- Consider a **dynamic policy**: tree only when the running batch is small (low conc), chain (on
  trtllm_mha) when the batch is large. This directly implements the "tree + dynamic block size" combo and
  recovers serving throughput. (New work, not yet implemented here.)

### 2c. Linear-attention (Qwen3.5 family) verify slowdown — root cause

Files: `models/qwen3_5.py` (`Qwen3_5GatedDeltaNet`), `models/qwen3_next.py`, backend
`layers/attention/linear/gdn_backend.py` (`GDNAttnBackend` over `MambaAttnBackendBase` in
`hybrid_linear_attn_backend.py`), kernel `layers/attention/fla/fused_sigmoid_gating_recurrent.py`.

- In `GDNAttnBackend.forward_extend`, `is_target_verify` reshapes the block to
  `[bs, draft_token_num, ...]` and runs `causal_conv1d_update` + a recurrent update **per draft position**.
- The core kernel `fused_sigmoid_gating_delta_rule_update_kernel` has a **sequential `for _ in range(0, T)`
  loop** (one iteration per token in the verify block); each step depends on the previous step's recurrent
  state `b_h`. Tree mode adds a **parent-state load** (`HAS_EAGLE_TREE_CUSTOM_ATTN_MASK`) and an
  intermediate-state store every step.
- Full attention verifies all N draft tokens **in parallel**; a linear layer verifies them **serially**.
  So at the *same* token budget N, a hybrid model pays ≈N sequential recurrent steps in every linear layer
  — explaining "verify much slower even at the same token budget."
- A linear layer **cannot represent an arbitrary tree mask** — only ancestor chains via parent pointers.
  Tree verify on these models is therefore both slower *and* algorithmically constrained.

**Can it be "fixed"?** Not removed (it's inherent to SSM recurrence), but mitigable:
- Restrict tree depth/branching on hybrid models (shallow, bushy trees cost the same serial depth as a
  chain of the tree's *depth*, not its node count — so keep depth small).
- Or run **chain verify on hybrid models** and reserve tree verify for full-attention models. Given the
  constraint, this is the pragmatic recommendation: **tree verify is a full-attention feature**; on
  hybrid/linear models, ship chain DFlash (which is what upstream does).

---

## Goal 3 — New models and tree-verify support

z-lab publishes DFlash drafts for (HF `z-lab` org, June 2026):

**Full-attention / standard (tree verify works well):**
`Qwen3-4B-DFlash-b16`, `Qwen3-8B-DFlash-b16`, `Qwen3-Coder-30B-A3B-DFlash`, `gpt-oss-20b-DFlash`,
`gpt-oss-120b-DFlash`, `LLaMA3.1-8B-Instruct-DFlash-UltraChat`.
→ Tree verify supported on flashinfer/fa3/fa4/triton. Verify time grows sub-linearly with tree budget
(parallel mask). These are the right models to showcase the draft-tree win.

**Hybrid linear-attention (tree verify is slow / constrained — see 2c):**
`Qwen3.5-{4B,9B,27B,35B-A3B,122B-A10B,397B-A17B}-DFlash`, `Qwen3.6-{27B,35B-A3B}-DFlash`,
`Qwen3-Coder-Next-DFlash`, `MiniMax-M2.5-DFlash`, `MiniMax-M2.7-DFlash`, `Kimi-K2.5-DFlash`,
`Kimi-K2.6-DFlash`, `GLM-5.1-FP8-DFlash`.
→ Gated-delta-net / Mamba layers ⇒ serial recurrent verify ⇒ tree verify time **grows ~linearly with
budget** and trees can't be arbitrary. Recommend chain DFlash here, or shallow trees only.

**Sliding-window attention (works, mask must respect the window):**
`gemma-4-26B-A4B-it-DFlash`, `gemma-4-31B-it-DFlash`, `gemma4-12B-it-DFlash`. The official dflash repo
added "Draft sliding window" / "interleaved SWA draft model" — tree verify should work but the verify
mask must intersect the SWA window. Worth a correctness check before quoting numbers.

**Recommended experiment matrix for the blog (Goal 4):**
- Full-attention: Qwen3-8B, Qwen3-Coder-30B (and optionally gpt-oss-20b) — tree vs chain, conc=1, budget
  sweep {16,32,64,128,256}. Expect clear accept-length gains.
- Hybrid: Qwen3.5-27B / MiniMax-M2.x — measure verify-time blowup vs chain at matched budget; this is the
  "where tree does NOT help" writeup the advisor explicitly green-lit ("if it doesn't, we can also have a
  writeup on it").

---

## Goal 4 — Competitive landscape and draft-tree results

### 4a. DFlash / DDTree / JetSpec (all block-diffusion drafters)

- **DFlash** (arXiv 2602.06036, `github.com/z-lab/dflash`): base block-diffusion drafter; one parallel
  forward proposes a *block* of N tokens (block size 16 = 15 draft tokens). Chain verify, ~6× lossless,
  up to ~2.5× over EAGLE-3. The `-b16` suffix = block-16, not bf16.
- **DDTree** (arXiv 2604.12989, `github.com/liranringel/ddtree`, SGLang PR #27509): **tree on top of
  DFlash**. Best-first heap over per-position draft log-probs, ≤2 children per pop (next-sibling +
  first-child), O(B log B), verified in one pass with an **ancestor-only mask**. MATH-500 τ: Qwen3-4B
  7.72→**10.71**, Qwen3-8B 7.79→**10.73**, Qwen3-Coder-30B 5.58→**8.10**. Speedups up to ~7.5× (vs ~5.6×
  chain). **This is essentially our algorithm** → cite as concurrent work; differentiate via dynamic block
  size + serving-aware (low-conc-only) tree policy.
- **JetSpec** (arXiv 2606.18394, `github.com/hao-ai-lab/JetSpec`): trains a causal parallel draft head;
  frozen target verifies the tree losslessly. Qwen3-8B greedy: budget16 τ6.0/4.8×, budget32 τ6.14/4.9×,
  budget256 τ8.62/7.82× (GSM8K). Headline 9.64× (MATH-500, budget256) is **batch-1/B200/CUDA-graphed
  drafter** best case. Single model family (Qwen3), no Llama/Vicuna, arXiv-only.

**Honest assessment of "JetSpec 胡来":** methodology is *not* sloppy (retrained baselines, open code).
The fair critique is **best-case-only headline numbers** and **narrow model coverage**. The specific
"bs=32 reused HF checkpoint" accusation appears **unsupported** — recommend not making it publicly; lead
instead with "DDTree (training-free) matches JetSpec's accept length without training a head, and our
dynamic-block-size extension closes the small remaining gap at realistic serving settings."

### 4b. Our draft-tree results (this fork, Qwen3-8B, fa4, conc=1)

`modal_tree_vs_chain.py` (B200) compares chain (topk=1) vs tree (topk=4) at matched budget.
**Measured 2026-06-27**, Qwen3-8B + `z-lab/Qwen3-8B-DFlash-b16`, gsm8k, conc=1, 40 prompts,
max_new_tokens=1024, **flashinfer** backend (fa4 is broken in this image — see 2a caveat), block size 16:

| Config | Accept length | tok/s | Latency (s) | Verify forwards |
|---|---:|---:|---:|---:|
| **Chain** (block 16, topk 1) | **6.44** | 669 | 18.6 | 1967 |
| **Tree** (topk 4, budget 16) | **7.27** | 690 | 17.8 | 1710 |

**Read:** at the *same* token budget (16), tree verify raises accept length **+12.8%** (6.44→7.27) and
cuts verify-forward count **−13%** (1967→1710), but output throughput rises only **+3.1%** (669→690).
This is exactly the "accept length up, throughput barely up" observation — now quantified. The throughput
gain is small because (i) at budget 16 the tree is shallow (only 15 non-root nodes, topk 4), and (ii)
tree-build + mask overhead and flashinfer (not the fastest kernel) eat most of the per-forward saving.

**Accept-length vs tree budget** (same setup, cuda-graph **off**, `--mem-fraction-static 0.6`, 32 prompts
— the curve the advisor asked for, "does tree verification really work"):

| Config | Budget | Accept length | tok/s |
|---|---:|---:|---:|
| Chain | 16 | 6.30 | 538 |
| Tree (topk 4) | 16 | 7.18 | 578 |
| Tree (topk 4) | 32 | **7.73** | 604 |
| Tree (topk 4) | 64 | **8.15** | 600 |

Accept length rises **monotonically 6.30 → 8.15** with budget — **tree verify demonstrably works**. At
budget 32 it's **+22.7% accept length** and **+12% throughput** over chain; gains taper by budget 64
(throughput plateaus ~600 tok/s as per-step tree overhead catches up). Same shape as DDTree's curve, at
smaller gsm8k magnitudes (DDTree's 7.8→10.7 is on MATH-500, whose longer reasoning chains give more
draftable runway). Absolute tok/s here is lower than the cuda-graph-on numbers above (538 vs 669 chain) —
expected, since graphs are off; the **relative** tree-vs-chain comparison is what's valid.

**Next runs to strengthen the blog:** (1) re-enable cuda graph with a small bs range (`--cuda-graph-bs 1`)
to get production throughput at each budget; (2) extend the budget grid to {128,256} (lower mem further);
(3) add **MATH-500** (bigger accept lengths, matches DDTree's headline table) and **Qwen3-Coder-30B**
(wider chain→tree gap). Raw reports in `tree_vs_chain_results/`.

---

## v2 Tree-Verify Port (new model study) — status & precise root cause

The three requested models (gpt-oss-120b, gemma-4-31B-it, Qwen3.6-27B) only exist on **current
upstream** sglang, so the tree-verify feature was **ported onto the upstream spec-v2 DFlash worker**
(`dflash_worker_v2.py`). Branch `dflash-tree-v2-port` in the `../sglang-v2-tree` worktree.

**block_size (→ budgets):** gpt-oss-120b-DFlash = **10** (→ 10,16,32,64); gemma-4-31B-it-DFlash = **16**;
Qwen3.6-27B-DFlash = **16** (→ 16,32,64).

**What works (validated on Qwen3-8B, B200, official `lmsysorg/sglang:nightly-...20260627` image):**
- The port builds and the DFLASH **server launches with tree verify on** (`SGLANG_DFLASH_TREE_VERIFY=1`,
  `--speculative-eagle-topk 4`, `--speculative-num-draft-tokens 32 --speculative-dflash-block-size 16`).
- Chain DFLASH is **correct**: accept length **6.20**, ~986 tok/s @ conc=1 gsm8k.
- End-to-end generation runs (no crash).

**What's broken — tree accept length is too LOW (4.28 < chain 6.20; should be ~7.7):**
Instrumentation of `verify_tree_greedy` shows the raw per-step `accept_token_num` = **1, 1, 3** (expected
~7). The target predictions don't reflect the tree structure → **the tree attention mask is not being
applied during the DFLASH verify forward.** Root cause chain:
- Upstream's DFLASH verify is chain-only; its decode CUDA graph is captured **without** a custom-mask
  buffer. Replaying a tree batch (which carries a custom mask) crashes (`raw_num_token` unset), so the
  port **forces the tree-verify step eager** (`can_run_graph` patched to return False for DFLASH +
  custom_mask).
- But the **eager** DFLASH verify path does not wire the tree `custom_mask` into the flashinfer verify
  wrapper (EAGLE captures its verify graph *with* a mask buffer; DFLASH never needed one). So the verify
  runs effectively maskless → wrong target predictions → only 1–3 tokens accepted.

**REFINED DIAGNOSIS (after instrumentation):** the tree mask IS built correctly (true_frac ~0.24,
valid topk-4 positions) AND is passed to the flashinfer verify wrapper, and per-step the tree picks the
right branch (e.g. takes topk[1]=13 when the target wants 13 and topk[0]=11 is wrong). The real symptom
is the output going **degenerate / non-lossless** → the committed tokens diverge from greedy → the
sequence goes off-track → average accept falls to 4.28. So the bug is in the **v2 inline accept
extraction** (the agent reimplemented v1's `dflash_info.verify()` accept + KV-repointing inline in the
worker), NOT the mask or the tree build. **Recommended fix: port v1's `dflash_info.verify()` tree-accept
logic verbatim** into the v2 worker rather than the inline reimplementation (v1 is proven correct:
Qwen3-8B chain 6.30 → tree 8.15). Also fixed along the way: the verify token count is now consistently the
tree budget (`_verify_num_tokens`), and the verify CUDA graph is captured mask-capable for tree
(`get_spec_info` patch, mirroring v1).

**The earlier (now superseded) fix framing:** make DFLASH verify apply the tree mask like EAGLE —
either (a) capture the DFLASH verify CUDA graph **with** a custom-mask buffer (dummy tree mask at capture,
populate per-step at replay; this is how EAGLE/v1-fork do it and is also the fastest), or (b) ensure the
eager flashinfer `prefill_wrappers_verify` are `plan()`-ed with `spec_info.custom_mask` for DFLASH. (a) is
preferred and matches the v1 fork, which **does** run tree verify correctly under CUDA graph
(Qwen3-8B: chain 6.30 → tree 8.15, see §4b).

**Other fixes already in the port (committed):** topk-drafting via `return_topk`; `build_tree_verify_tokens`
+ fused topk=4 op ported; tree-greedy accept + KV re-pointing inline (v2 has no `dflash_info.verify`);
GDN/mamba support for Qwen3.6 (`retrieve_*` field naming so `HybridLinearAttnBackend` finds the tree
pointers; mamba `last_correct_step_indices` = deepest accepted node's block position); arg-hook patched to
allow `topk>1` and `num_draft_tokens>block_size` for DFLASH tree; image/bench fixes (rust+protoc for the
gRPC ext, `sglang-kernel`/`flashinfer` pins, deep_gemm/nvrtc, cuda-graph-bs bound, broken-fa4 override).

**Modal harness:** `modal_v2_tree.py` (`main` = Qwen3-8B validation; `three` = 3-model sweep, baseline +
chain + tree budgets across conc {1,8,32} × {gsm8k, mt-bench}, CUDA graph on; `dbg`/`debug` = direct
server launch with captured logs; results persist to a modal Volume for session-death resilience). Once
the mask fix above lands, `modal run modal_v2_tree.py::three` produces the requested numbers. The
**chain** half of that sweep is already correct and runnable today (`--chain-only`).

---

## Goal 1 — Merging recent upstream SGLang

**Finding (measured):** merge base `0c204fbd5`; upstream `main` is **3194 commits ahead, touching 4864
files** (fork HEAD 2026-04-09, upstream tip 2026-06-27, ~2.5 months). A non-destructive `git merge-tree`
dry-run yields **~14 conflicted files**:

```
.gitignore
python/sglang/srt/layers/attention/flashinfer_backend.py
python/sglang/srt/managers/scheduler.py
python/sglang/srt/model_executor/model_runner.py
python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py
python/sglang/srt/models/dflash.py            (add/add)
python/sglang/srt/models/qwen3.py
python/sglang/srt/models/qwen3_5.py
python/sglang/srt/server_args.py
python/sglang/srt/speculative/dflash_info.py  (add/add — fork & upstream each wrote their own)
python/sglang/srt/speculative/dflash_utils.py (add/add)
python/sglang/srt/speculative/spec_info.py
python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py
test/registered/spec/dflash/test_dflash.py
```

The **add/add** conflicts on `dflash_info.py`, `dflash_utils.py`, `models/dflash.py` are the crux: both
sides created these files independently after the merge base, so git cannot auto-merge — they are two
different DFlash implementations. Crucially, **upstream now ships DFlash natively but re-architected**:

- Upstream: `dflash_worker_v2.py` (spec-v2 / overlap scheduler), `dflash_info.py` + `dflash_info_v2.py`,
  `dflash_utils.py`, `triton_ops/dflash.py`. **Chain-only** (`custom_mask=None` in the worker).
- This fork: `dflash_worker.py` (spec-v1) + **tree verify** + `triton_ops/dflash_tree_expand_topk.py`,
  with `dflash_info.py` / `dflash_utils.py` **diverged** from upstream.

So a naive `git merge upstream/main` will conflict heavily exactly in the files we customized
(`dflash_info.py`, `dflash_utils.py`, `flashattention_backend.py`, model files), **and** upstream moved
DFlash to a v2 worker that doesn't exist here.

**Recommended strategy (do NOT brute-force one giant merge):**
1. **Re-base the feature, not the fork.** Treat upstream `main` as the new base; re-apply the tree-verify
   patch set as a focused diff on top. Our tree contribution is concentrated in: `dflash_worker.py` (tree
   branch), `dflash_info.py` (`build_tree_kernel_efficient` plumbing), `dflash_utils.py`
   (`build_tree_verify_tokens`), `triton_ops/dflash_tree_expand_topk.py`, and the `topk>1` mask paths in
   `flashattention_backend.py` (already upstream).
2. **Decide v1 vs v2.** Port tree verify onto upstream's `dflash_worker_v2` (preferred — gets the overlap
   scheduler + future maintenance) rather than keeping the v1 worker alive. This is a reimplementation of
   the tree branch against the v2 data structures, not a merge.
3. Pull in upstream model files for the new targets (Qwen3.5/3.6, MiniMax-M2.x, Kimi-K2.x, gemma-4) — these
   are needed for Goal 3 and are cleaner to take from upstream than to merge.

Only ~14 files conflict, but the add/add DFlash files require the v1-tree-vs-upstream-v2 decision above,
and the fork's v1 worker calls many internal APIs that moved across 3194 upstream commits — so a
hand-resolved merge will not build/run without iterative GPU testing. The maintainable path is re-applying
the tree-verify patch onto upstream, not resolving one giant merge in place.

**Internal bug fixed during this investigation:** `benchmark/dflash/bench_dflash_sweep.py:_is_blackwell`
referenced `envs.IS_BLACKWELL`, which this fork's `environ.py` does not define → every sweep crashed on
launch. Patched to a safe `getattr` fallback (SM≥100 detection already covers B200).
