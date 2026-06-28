# Draft-tree speculative decoding: JetSpec vs DDTree vs our SGLang/DFlash tree (B200)

Benchmark assignment (Zihan): (1) JetSpec on B200 vs DFlash, (2) DDTree + our tree vs JetSpec,
(3) SGLang vs vLLM tree implementation — speedup & overhead. All numbers below: **Qwen3-8B,
B200, batch=1 (single stream), greedy (temp 0)**, 24 samples unless noted.

Repos: JetSpec `hao-ai-lab/JetSpec` (engine) + `JetSpec-project/vllm-jetspec` (vLLM fork);
DDTree `liranringel/ddtree` (arXiv 2604.12989); our DFlash tree on SGLang spec-v2 (this fork).

---

## The methods, and what makes each tree

| method | draft head | tree construction | engine / kernels |
|---|---|---|---|
| **DFlash chain (JetSpec ref)** | `JetSpec/jetspec-qwen3-8b` (SAME head as JetSpec tree) | none (linear block) | JetSpec HF reference, sdpa |
| **DFlash chain (DDTree ref / ours)** | z-lab DFlash-b16 | none (linear block) | DDTree PyTorch ref / SGLang spec-v2 |
| **our SGLang tree** | z-lab DFlash-b16 | EAGLE top-k (fixed topk=4) | SGLang spec-v2, flashinfer/triton, cuda-graph |
| **DDTree** | z-lab DFlash-b16 (SAME as ours) | adaptive best-first heap over draft logits (budget-bounded) | PyTorch reference, torch-sdpa, no cuda-graph |
| **JetSpec** | `JetSpec/jetspec-qwen3-8b` (causal-parallel) | budget over high-prob branches | optimized engine (triton paged-tree + fused gemm + cuda-graph) OR HF reference |

**The DFlash baseline is not one thing — note WHICH head:**
- **Task 1's "DFlash"** is JetSpec's own HF reference (`bench/reference/dflash_baseline.py`) running the
  **`JetSpec/jetspec-qwen3-8b` head in linear-block mode** — the *same head* JetSpec's tree uses. So
  Task 1 (DFlash 3.11×/acc6.00 vs JetSpec tree 4.69×/acc8.35) is a clean **same-head, linear-vs-tree**
  ablation: the gain is purely the tree, not head quality.
- **DDTree's "dflash" and our SGLang chain** use the **z-lab/Qwen3-8B-DFlash-b16** head instead.
- The head architecture in all cases is the original DFlash design (`DFlashDraftModel`, vendored from
  `causal_parallel_drafting/model/dflash.py`): a Qwen3 variant sharing the target's embed/lm_head,
  conditioning on tapped target hidden states via an `fc` fusion, emitting `block_size` per-depth logits
  in one parallel pass. Algorithm (all identical): propose linear block → verify → accept longest
  greedy-prefix + 1 correction → crop KV → repeat (greedy/lossless).

The crucial axis: DDTree and our SGLang tree both reuse the z-lab DFlash head and only change the tree
shape; JetSpec uses its own causal-parallel head. DDTree's novelty is the adaptive best-first tree (vs
our fixed EAGLE top-k).

---

## Task 1 — JetSpec vs DFlash on B200 (gsm8k)

**JetSpec HF reference** (sdpa base + triton tree, budget 256 / depth 20 / width 7):

| method | accept_len | decode tok/s | speedup vs AR |
|---|---|---|---|
| DFlash chain (bs16) | 6.00 | 258 | 3.11× |
| **JetSpec tree** (accum_logp) | **8.35** | 389 | **4.69×** |

**JetSpec optimized engine** (cuda-graph drafter + fused gemm, budget 127 / depth 15):

| prompt set | accept_len | tree tok/s | AR tok/s | speedup |
|---|---|---|---|---|
| gsm8k | 7.57 | **954** | 166 | **5.73×** |
| mt-bench | 3.45 | 451 | 160 | 2.83× |

→ **JetSpec's tree beats DFlash chain on B200 on BOTH accept and speedup** (4.69× vs 3.11× in the
same reference harness; 5.73× in its optimized engine). The trained causal-parallel head + efficient
tree kernel is why.

---

## Task 2 — DDTree + our tree vs JetSpec (gsm8k)

**DDTree reference** (PyTorch, torch-sdpa, z-lab DFlash-b16 head, speedup vs AR=64.7 tok/s):

| method | accept_len | speedup vs AR | tree tok/s (≈) |
|---|---|---|---|
| dflash chain | 6.59 | 3.56× | 230 |
| ddtree_tb16 | 7.33 | 2.43× | 157 |
| ddtree_tb32 | 7.91 | 2.63× | 170 |
| ddtree_tb64 | 8.38 | 2.81× | 182 |
| ddtree_tb128 | 8.64 | 2.05× | 133 |

**DDTree reference — mt-bench** (48 samples, AR=59.3 tok/s) — same pattern as gsm8k:

| method | accept_len | speedup vs AR |
|---|---|---|
| dflash chain | 4.14 | 2.13× |
| ddtree_tb16 | 4.87 | 1.50× |
| ddtree_tb64 | 5.60 | 1.72× |
| ddtree_tb128 | 5.90 | 1.38× |

(DDTree tree again has higher accept than its chain but lower speedup — overhead-bound. cf. JetSpec
engine mt-bench: accept 3.45 but **2.83×** — the optimized engine wins on chat too.)

**Our SGLang tree** (optimized engine, cuda-graph, conc=1; no-spec AR not run):

| method | accept_len | tree tok/s |
|---|---|---|
| DFlash chain | 5.79 | 904 |
| tree b16 | 6.54 | 832 |
| tree b32 | 7.23 | 797 |
| tree b64 | 7.51 | 806 |

### Cross-method ranking (gsm8k, B200, bs=1)

**1. Accept length (algorithm quality, engine-independent), at comparable budget:**
> DDTree adaptive (7.3→8.6) ≈ JetSpec trained (7.6→8.35) > our SGLang EAGLE top-k (6.5→7.5) > DFlash chain (6.0)

DDTree's best-first search and JetSpec's trained head BOTH beat a plain EAGLE top-k tree on accept.
DDTree gets there **without training** (pure search on the existing DFlash head) — that's its appeal.

**2. Absolute B200 tok/s (the real bottom line):**
> JetSpec engine **954** ≈ our SGLang tree **832** ≫ JetSpec reference 389 ≫ DDTree reference ~180

**3. Speedup ratio within each harness:** JetSpec reference tree 4.69× > its chain 3.11× (tree WINS);
DDTree reference chain 3.56× > its tree 2.05–2.81× (tree LOSES).

**The headline:** *Implementation dominates algorithm for wall-clock.* DDTree has the best trees
(highest accept) but its unoptimized reference (sdpa + CPU heap build, no cuda-graph) is so slow the
accept gain is wasted — its tree is slower than even its own chain. JetSpec turns the same kind of
accept gain into 5.73× via an optimized engine. Our SGLang tree (832 tok/s) is in JetSpec's league on
throughput while using the untrained head + a simpler EAGLE top-k tree.

---

## Task 3 — SGLang vs vLLM tree implementation & overhead

### Implementation, axis by axis

| axis | SGLang (our DFlash tree) | vLLM-JetSpec |
|---|---|---|
| **tree mask** | dense `custom_mask` tensor → flashinfer/triton `begin_forward(custom_mask=)` (EAGLE-shared) | dense `tree_attn_bias` (triton) **OR** sparse `ancestor_masks` → **Optimus cutedsl tree kernel** (no dense bias) |
| **tree build** | `build_tree_kernel_efficient` (GPU) | CPU-resident `DraftTreeCPU` (BFS/DFS/entropy modes), budget-bounded |
| **verify/accept** | `verify_tree_greedy` GPU kernel | `gpu_tree_accept` (GPU prefix-match, single `.item()` sync) |
| **KV commit** | `move_accept_tokens_to_target_kvcache` — physical move-to-front | **physical** (move-to-front) **OR** `logical` layout = pre-reserved contiguous slots → **zero post-verify copy** |
| **cuda graph** | batch-size capture (verify graph w/ mask buffer) | **budget-aware** capture sizes (per tree-node-count) + batch |

**Same idea, two refinements vLLM adds:** (a) a native **sparse tree-attention kernel** (Optimus)
that skips O(budget²) dense-mask materialization, and (b) a **logical KV layout** that avoids the
post-verify KV move entirely by reserving contiguous logical slots up front. SGLang currently always
materializes the dense mask and always does the move-to-front (which we just had to fix — the KV-window
aliasing bug — so its correctness is load-bearing).

### Overhead, measured

A clean overhead metric is **efficiency = speedup ÷ accept_len** (fraction of the algorithmic
accept-length that survives as wall-clock speedup; 1.0 = zero overhead):

| config | accept_len | speedup | efficiency | overhead |
|---|---|---|---|---|
| JetSpec **engine** (cuda-graph) gsm8k | 7.57 | 5.73× | **0.76** | low |
| JetSpec reference (HF) gsm8k | 8.35 | 4.69× | 0.56 | medium |
| DFlash chain (JetSpec ref) | 6.00 | 3.11× | 0.52 | medium |
| DDTree reference tree tb64 | 8.38 | 2.81× | **0.34** | **high** |

→ The optimized engine (cuda-graph drafter + fused gemm + efficient tree kernel) recovers **76%** of
the accept-length as speedup; the PyTorch reference tree recovers only **34%**. The ~24–66% lost is the
tree path's per-step overhead: **draft-head forward, tree construction, tree-mask build, verify forward,
KV commit.** vLLM-JetSpec's Optimus kernel + logical KV + GPU-accept + budget-aware graphs target exactly
these; SGLang's spec-v2 cuda-graph path (mask-capable verify graph) puts us near the JetSpec-engine end
for our EAGLE top-k tree (832 vs 954 tok/s).

---

## Task 3b — vLLM-DFlash (linear) vs vLLM-JetSpec (tree), SAME vLLM engine

`vllm-jetspec`'s spec `method="dflash"` is unified: `--tree-width 1` == linear DFlash (vLLM's
DFlash), `--tree-width >1` == JetSpec parallel tree. Same engine, same head (`jetspec-qwen3-8b`),
so this isolates the tree contribution INSIDE vLLM (the in-engine analogue of Task 1's HF-reference
ablation). Built by overlaying the fork's python onto stock vLLM 0.23.0 (the fork adds no .cu, so
stock `_C` is valid). Qwen3-8B, gsm8k, B200, batch=1, 32 samples.

| config | tree kernel | KV layout | cuda-graph | accept_len | tok/s |
|---|---|---|---|---|---|
| **DFlash linear** (tw1), default mode | — | — | piecewise (default) | 5.60 | 955 |
| **DFlash linear** (tw1), `full_decode_only` | — | — | ✅ **FULL** target replay (num_tokens=16) | 5.67 | **981** |
| JetSpec tree tw7/budget128 | triton | physical | ❌ (captures=0) | 6.80 | 226 |
| JetSpec tree tw7/budget128 | triton | logical | ✅ full | 7.85 | 428 |
| JetSpec tree tw4/budget32 | triton | logical | ✅ full | 6.86 | 287 |
| JetSpec tree tw7/budget128 | **optimus** | logical | ✅ full | — | engine-init failed: `optimus_cutedsl` not installed |

**Findings (triton kernel):**
- **Inside vLLM, the linear DFlash is much faster than the JetSpec tree** (955 vs best-tree 428 tok/s),
  even though the tree's accept length is higher (7.85 vs 5.60). This is the OPPOSITE of Task 1's HF
  reference (where the tree won 4.69× vs 3.11×) and of JetSpec's own standalone engine (954 tok/s).
- The tree config matters enormously: **cuda-graph + logical zero-copy KV nearly doubled the tree**
  (226 → 428 tok/s). With captures=0 (default) the tree verify is NOT graphed — an easy way to mis-measure.
- **cuda-graph mode caveat (corrected):** DFlash linear's *target verify* CAN be a FULL cuda graph (like
  SGLang) via `--cudagraph-mode full_decode_only` → 955 → 981 tok/s (target verify shows mode=FULL,
  num_tokens=16). The default mode left it PIECEWISE; the gain from FULL is small here because the linear
  verify is only 16 tokens. Note a hard design limit in vllm-jetspec: the **draft proposer forward is
  always PIECEWISE-captured** ("The proposer itself uses PIECEWISE capture", `eagle.py:405`;
  "Only supports PIECEWISE cudagraphs", `extract_hidden_states.py:217`) — only the target verify can be
  FULL. This applies to both linear and tree, so it does not change the linear-vs-tree ranking.
- Counter-intuitively, budget 32 (287) was slower than budget 128 (428): the per-step overhead (draft
  passes `max_draft_passes=5`, CPU tree build, draft-head forward) dominates over verify-forward size on
  B200, so the smaller tree's lower accept (6.86 vs 7.85) means more steps and lower throughput.
- The acceptance RATE is very low (5–19%): a 128-node tree accepting ~8 tokens wastes most draft work.
- **Optimus kernel could not be tested:** JetSpec's headline tree throughput (954 tok/s, standalone
  engine) relies on the **Optimus sparse-tree cutedsl kernel**, not triton. It requires the
  `optimus_cutedsl` package, which is NOT on PyPI / not in stock vLLM (confirmed: `ModuleNotFoundError`),
  so the optimus path fails at engine init in this overlay. Reproducing JetSpec's tree speed in vLLM
  would require building/obtaining that proprietary kernel — i.e. the tree's wall-clock win is gated on a
  JetSpec-specific kernel, not available in the generally-installable vLLM stack.

**Takeaway:** the JetSpec *tree algorithm* (higher accept) does not automatically beat linear DFlash in
the vLLM integration with the triton kernel — the draft-pass + tree-build overhead per step outweighs the
accept gain at batch=1. The tree's win in JetSpec's own numbers depends on its specific optimized engine
(fused gemm, graphed drafter, and likely the Optimus kernel). Same lesson as Task 2/3: *the engine and
kernel decide whether a better tree turns into wall-clock speedup.*

## Bottom line

- **JetSpec leads on B200 batch=1** (954 tok/s, 5.73× on gsm8k) — trained head + optimized engine.
- **DDTree is an algorithm, not an engine**: best accept-length (adaptive trees, no training) but its
  reference impl is overhead-bound (tree slower than chain). Port its tree-build into an optimized
  engine and it should rival JetSpec on accept.
- **Our SGLang tree is competitive on throughput** (832 tok/s, in JetSpec's league) with the untrained
  head and a simpler EAGLE top-k tree; the win path is a better tree (adopt DDTree-style adaptive
  construction) + vLLM-style sparse-mask / logical-KV to cut the remaining overhead.
- **Overhead lesson**: accept-length is necessary but not sufficient — only an optimized engine
  (cuda-graph, fused gemm, sparse tree attn, zero-copy KV) converts it to speedup (0.76 vs 0.34
  efficiency).
