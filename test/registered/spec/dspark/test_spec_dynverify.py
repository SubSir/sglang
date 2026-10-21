"""Dynamic verify (entropy confidence, DFLASH ragged verify): CPU checks.

Base-parity cases load the unmodified worker/forward-batch sources from
SGLANG_BASE_SRC (the image's /sgl-workspace/sglang/python) and are skipped
without it.
"""

import importlib.util
import math
import os
import sys
import types
import unittest

import numpy as np
import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptGreedy
from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    ScheduleVerifyLensTopk,
    schedule_verify_lens_topk_from_survival,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    _spec_mrope_token_deltas,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    DSparkVerifyPlanner,
    compute_verify_token_budget,
    ratio_policy_budgets,
)
from sglang.srt.speculative.dspark_components.dspark_sps import SpsCostTable
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.spec_confidence import (
    ConfidenceSource,
    EntropyCalibrationTable,
    EntropyCalibrator,
    conditional_from_survival,
    drawn_prob_from_candidates,
    drawn_prob_from_logits,
    entropy_confidence,
    entropy_from_logits,
    entropy_from_probs,
    prompt_opens_thinking,
    raw_survival_chain,
    shifted_draft_prob,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

BINS = 20


def _offline_calibrate(train_x, train_y, test_x):
    """analyze_sink.calibrate, verbatim: the tables are fitted with it."""
    out = np.empty_like(test_x)
    for j in range(train_x.shape[1]):
        edges = np.unique(np.quantile(train_x[:, j], np.linspace(0, 1, BINS + 1)))
        if len(edges) < 2:
            out[:, j] = np.nanmean(train_y[:, j])
            continue
        tb = np.clip(
            np.searchsorted(edges, train_x[:, j], side="right") - 1, 0, len(edges) - 2
        )
        means = np.array(
            [
                (
                    np.nanmean(train_y[tb == b, j])
                    if np.isfinite(train_y[tb == b, j]).any()
                    else np.nan
                )
                for b in range(len(edges) - 1)
            ]
        )
        means = np.where(np.isnan(means), np.nanmean(train_y[:, j]), means)
        sb = np.clip(
            np.searchsorted(edges, test_x[:, j], side="right") - 1, 0, len(edges) - 2
        )
        out[:, j] = means[sb]
    return out


def _export_table(train_x, train_y) -> EntropyCalibrationTable:
    """What the offline exporter writes: per slot, unique edges and bin means."""
    edges, values = [], []
    for j in range(train_x.shape[1]):
        e = np.unique(np.quantile(train_x[:, j], np.linspace(0, 1, BINS + 1)))
        if len(e) < 2:
            edges.append(e.tolist())
            values.append([float(np.nanmean(train_y[:, j]))])
            continue
        probe = (e[:-1] + e[1:]) / 2  # one point inside each bin
        means = _offline_calibrate(train_x[:, j : j + 1], train_y[:, j : j + 1], probe[:, None])
        edges.append(e.tolist())
        values.append(means[:, 0].tolist())
    return EntropyCalibrationTable(edges=edges, values=values)


class _SyncStub:
    def sync(self, site, tensor):
        return tensor


def _load_base(module_name: str, rel_path: str):
    base = os.environ.get("SGLANG_BASE_SRC")
    if not base:
        return None
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(base, rel_path)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses resolve fields through it
    spec.loader.exec_module(module)
    return module


class TestEntropyCalibration(CustomTestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        gamma = 7
        ent = rng.gamma(2.0, 0.6, size=(4000, gamma))
        self.train_x = np.minimum.accumulate(np.cumprod(np.exp(-ent), 1), 1)
        alpha = np.clip(np.exp(-ent) + rng.normal(0, 0.1, ent.shape), 0, 1)
        self.train_y = np.cumprod(alpha, 1)
        test_ent = rng.gamma(2.0, 0.6, size=(500, gamma))
        self.test_ent = test_ent
        self.test_x = np.minimum.accumulate(np.cumprod(np.exp(-test_ent), 1), 1)
        self.gamma = gamma

    def test_runtime_lookup_matches_offline_calibrate(self):
        """The served survival must equal the offline per-slot table, then the
        monotone fix the offline E[L] applies (running min of clip(0, 1))."""
        table = _export_table(self.train_x, self.train_y)
        cal = EntropyCalibrator(table=table, gamma=self.gamma, device="cpu")
        got = cal.survival(torch.tensor(self.test_x, dtype=torch.float32)).numpy()
        ref = _offline_calibrate(self.train_x, self.train_y, self.test_x)
        ref = np.minimum.accumulate(np.clip(ref, 0, 1), 1)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)

    def test_out_of_range_and_constant_slots(self):
        """Below the first edge -> bin 0, above the last -> last bin, a constant
        training slot -> its mean (analyze_sink's three edge cases)."""
        x = self.train_x.copy()
        x[:, 3] = 0.5
        table = _export_table(x, self.train_y)
        cal = EntropyCalibrator(table=table, gamma=self.gamma, device="cpu")
        probe = np.stack([np.full(self.gamma, -1.0), np.full(self.gamma, 2.0)])
        got = cal.survival(torch.tensor(probe, dtype=torch.float32)).numpy()
        ref = _offline_calibrate(x, self.train_y, probe)
        ref = np.minimum.accumulate(np.clip(ref, 0, 1), 1)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)

    def test_raw_chain_matches_offline_chain_of(self):
        got = raw_survival_chain(torch.exp(-torch.tensor(self.test_ent))).numpy()
        np.testing.assert_allclose(got, self.test_x, rtol=1e-5)

    def test_conditional_round_trip(self):
        """The planner cumprods the per-slot confidence; it must get the
        calibrated survival back, including chains that hit zero."""
        surv = torch.tensor(
            [[0.9, 0.6, 0.6, 0.2, 0.0, 0.0, 0.0], [1.0, 1.0, 0.5, 0.25, 0.2, 0.1, 0.05]]
        )
        cond = conditional_from_survival(surv)
        self.assertTrue(bool(((cond >= 0) & (cond <= 1)).all()))
        torch.testing.assert_close(torch.cumprod(cond, 1), surv)

    def test_gamma_mismatch_is_rejected(self):
        table = _export_table(self.train_x, self.train_y)
        with self.assertRaises(ValueError):
            EntropyCalibrator(table=table, gamma=self.gamma + 1, device="cpu")

    def test_json_round_trip(self):
        table = _export_table(self.train_x, self.train_y)
        import msgspec

        again = EntropyCalibrationTable.from_json(msgspec.json.encode(table))
        self.assertEqual(again, table)


class TestBudgetFromEntropy(CustomTestCase):
    def test_topk_picks_highest_calibrated_survival(self):
        """Budget B over a batch verifies exactly the B highest survival slots
        (parent-closed by monotonicity), the property the offline replay assumed."""
        torch.manual_seed(0)
        bs, gamma = 6, 7
        logits = torch.randn(bs, gamma, 50) * torch.linspace(0.5, 4, bs).view(bs, 1, 1)
        rng = np.random.default_rng(1)
        tx = np.minimum.accumulate(np.cumprod(rng.uniform(0, 1, (2000, gamma)), 1), 1)
        table = _export_table(tx, tx * 0.9)
        cal = EntropyCalibrator(table=table, gamma=gamma, device="cpu")
        conf = entropy_confidence(
            entropy=entropy_from_logits(logits, None), calibrator=cal
        )
        surv = torch.cumprod(conf, 1)
        budget = 13
        cfg = DSparkScheduleConfig(gamma=gamma)
        lens = ScheduleVerifyLensTopk.torch(confidence=conf, budget=budget, cfg=cfg)
        self.assertEqual(int((lens - 1).sum()), budget)
        chosen = torch.zeros_like(surv, dtype=torch.bool)
        for i, n in enumerate(lens.tolist()):
            chosen[i, : n - 1] = True
        self.assertGreaterEqual(float(surv[chosen].min()), float(surv[~chosen].max()))


class _Kv:
    def __init__(self):
        self.req_pool_idx = None
        self.kv_allocated_len = 0

    @property
    def holds_kv(self):
        return self.req_pool_idx is not None


class TestPerRequestThinkingTable(CustomTestCase):
    """Thinking and non-thinking requests in one batch each use their own entropy
    table; the per-request choice is fixed by the prompt and lives on the pool row."""

    START, END = 7, 8  # stand-in <think> / </think> ids

    def _tables(self, gamma):
        rng = np.random.default_rng(3)
        tx = np.minimum.accumulate(np.cumprod(rng.uniform(0, 1, (2000, gamma)), 1), 1)
        on = EntropyCalibrator(table=_export_table(tx, tx * 0.95), gamma=gamma, device="cpu")
        off = EntropyCalibrator(table=_export_table(tx, np.sqrt(tx)), gamma=gamma, device="cpu")
        return on, off

    def test_prompt_rule(self):
        S, E = self.START, self.END
        cases = {
            (1, 2, S, 3): True,  # Qwen thinking: "...assistant\n<think>\n"
            (1, 2, S, 3, 3, E, 3, 3): False,  # Qwen non-thinking: "<think>\n\n</think>\n\n"
            (1, S, 4, E, 5, 2, S): True,  # earlier turn closed, new turn opens (V4.1 "<think>")
            (1, 2, S, 4, E): False,  # V4.1 chat mode ends with "</think>"
            (1, 2, 3): False,  # no marker at all
        }
        for ids, want in cases.items():
            self.assertEqual(prompt_opens_thinking(list(ids), think_start=S, think_end=E), want, ids)

    def test_mixed_batch_matches_two_single_table_runs(self):
        torch.manual_seed(0)
        gamma = 7
        on, off = self._tables(gamma)
        entropy = torch.rand(6, gamma) * 3
        thinking = torch.tensor([True, False, True, False, False, True])
        mixed = entropy_confidence(entropy=entropy, calibrator=on, calibrator_nothink=off, thinking=thinking)
        only_on = entropy_confidence(entropy=entropy, calibrator=on)
        only_off = entropy_confidence(entropy=entropy, calibrator=off)
        torch.testing.assert_close(mixed[thinking], only_on[thinking])
        torch.testing.assert_close(mixed[~thinking], only_off[~thinking])
        self.assertFalse(torch.allclose(only_on, only_off))  # the two tables do differ
        # One table configured: unchanged path, the flag is ignored.
        torch.testing.assert_close(
            entropy_confidence(entropy=entropy, calibrator=on, thinking=thinking), only_on
        )

    def _selector_and_pool(self, gamma):
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
        from sglang.srt.speculative.spec_confidence import DraftConfidenceSelector

        on, off = self._tables(gamma)
        sel = DraftConfidenceSelector(gamma=gamma, device="cpu", default_source=ConfidenceSource.DRAFT_PROB, has_head=False)
        sel.calibrator, sel.calibrator_nothink = on, off
        pool = ReqToTokenPool(size=8, max_context_len=16, device="cpu", enable_memory_saver=False)
        tok = types.SimpleNamespace(get_vocab=lambda: {"<think>": self.START, "</think>": self.END})
        # The pool does not exist when the worker builds the selector: a getter.
        sel.configure_thinking(tokenizer=tok, req_to_token_pool_getter=lambda: pool)
        return sel, pool, on, off

    def test_flag_lives_on_the_pool_row_through_reshuffles(self):
        sel, pool, _, _ = self._selector_and_pool(7)
        S, E = self.START, self.END
        reqs = [types.SimpleNamespace(kv=_Kv(), origin_input_ids=ids)
                for ids in ([1, S, 2], [1, S, 2, E, 3], [S], [4, 5])]
        rows = pool.alloc(reqs)
        sel.mark_prompts(reqs)  # their prefill
        self.assertEqual(sel.thinking_rows(torch.tensor(rows)).tolist(), [True, False, True, False])
        # Chunked prefill: same row, marked again with the same value.
        reqs[0].kv.kv_allocated_len = 4
        self.assertEqual(pool.alloc([reqs[0]]), [rows[0]])
        sel.mark_prompts([reqs[0]])
        self.assertTrue(bool(sel.thinking_rows(torch.tensor([rows[0]]))))
        # Retract / preempt: the row is freed and taken by another request (marked at
        # its prefill); the retracted request is re-prefilled on a new row.
        pool.free(reqs[1])
        other = types.SimpleNamespace(kv=_Kv(), origin_input_ids=[9, S])
        other_row = pool.alloc([other])[0]
        sel.mark_prompts([other])
        new_row = pool.alloc([reqs[1]])[0]
        sel.mark_prompts([reqs[1]])
        self.assertTrue(bool(sel.thinking_rows(torch.tensor([other_row]))))
        self.assertFalse(bool(sel.thinking_rows(torch.tensor([new_row]))))
        self.assertEqual(sel.thinking_rows(torch.tensor([rows[0], rows[2], rows[3]])).tolist(), [True, True, False])

    def test_selector_reads_the_pool_flags(self):
        gamma = 7
        sel, pool, on, off = self._selector_and_pool(gamma)
        reqs = [types.SimpleNamespace(kv=_Kv(), origin_input_ids=ids)
                for ids in ([1, self.START], [2, self.START, 3, self.END], [3, self.START])]
        rows = torch.tensor(pool.alloc(reqs))
        sel.mark_prompts(reqs)
        entropy = torch.rand(3, gamma) * 3
        got = sel.entropy_confidence(entropy=entropy, req_pool_indices=rows)
        want = entropy_confidence(entropy=entropy, calibrator=on, calibrator_nothink=off,
                                  thinking=torch.tensor([True, False, True]))
        torch.testing.assert_close(got, want)

    def test_single_table_configures_nothing(self):
        from sglang.srt.speculative.spec_confidence import DraftConfidenceSelector

        sel = DraftConfidenceSelector(gamma=7, device="cpu", default_source=ConfidenceSource.DRAFT_PROB, has_head=False)
        sel.calibrator, _ = self._tables(7)
        sel.configure_thinking(tokenizer=None, req_to_token_pool_getter=lambda: None)  # no NOTHINK table
        sel.mark_prompts([types.SimpleNamespace(kv=_Kv(), origin_input_ids=[1])])  # no-op
        entropy = torch.rand(2, 7) * 3
        self.assertIsNone(sel.thinking_rows(torch.tensor([1, 2])))
        torch.testing.assert_close(sel.entropy_confidence(entropy=entropy, req_pool_indices=torch.tensor([1, 2])),
                                   entropy_confidence(entropy=entropy, calibrator=sel.calibrator))


class TestDcutSignalAndPolicy(CustomTestCase):
    """D-cut (arXiv 2607.14647): shifted draft-prob score and the ratio policy."""

    def test_shifted_score_never_reads_the_slot_own_token(self):
        """Slot k's survival is prod_{t<k} q(z_t): changing the token drawn at k
        must not move slot k's survival (App. B; otherwise RS is biased)."""
        q = torch.tensor([[0.9, 0.5, 0.8, 0.3]])
        surv = torch.cumprod(shifted_draft_prob(q), 1)
        torch.testing.assert_close(surv, torch.tensor([[1.0, 0.9, 0.45, 0.36]]))
        q2 = q.clone()
        q2[0, 2] = 0.01  # a different token drawn at slot 3
        surv2 = torch.cumprod(shifted_draft_prob(q2), 1)
        torch.testing.assert_close(surv2[:, :3], surv[:, :3])

    def test_drawn_prob_gathers(self):
        logits = torch.tensor([[[0.0, 1.0, 2.0], [2.0, 0.0, 0.0]]])
        tokens = torch.tensor([[2, 0]])
        q = torch.softmax(logits, -1)
        torch.testing.assert_close(
            drawn_prob_from_logits(logits, None, tokens), torch.stack([q[0, 0, 2], q[0, 1, 0]])[None]
        )
        cand = torch.tensor([[[5, 7, 9], [3, 4, 6]]])
        q_rows = torch.tensor([[[0.2, 0.5, 0.3], [0.6, 0.3, 0.1]]])
        got = drawn_prob_from_candidates(q_rows=q_rows, candidate_ids=cand, tokens=torch.tensor([[7, 6]]))
        torch.testing.assert_close(got, torch.tensor([[0.5, 0.1]]))

    def test_ratio_budgets_follow_k_rho(self):
        """K_rho = max(B, ceil(rho * B * (D + 1))); budget = K_rho - B (anchors free)."""
        # B = 3 requests, D + 1 = 8: K = max(3, ceil(.25*24)=6) .. 24 -> budgets 3, 9, 15, 21.
        self.assertEqual(ratio_policy_budgets(num_requests=3, max_verify_len=8, num_candidates=21), [3, 9, 15, 21])
        # B = 1, D + 1 = 2: rho 0.25 gives ceil(0.5) = 1 = B -> budget 0.
        self.assertEqual(ratio_policy_budgets(num_requests=1, max_verify_len=2, num_candidates=1), [0, 1])
        self.assertEqual(ratio_policy_budgets(num_requests=3, max_verify_len=8, num_candidates=10), [3, 9, 10])

    def test_ratio_policy_is_argmax_over_the_four_points(self):
        torch.manual_seed(0)
        surv = torch.cumprod(torch.rand(4, 7) * 0.5 + 0.5, 1)
        table = SpsCostTable(sample_batch_tokens=[4, 8, 16, 24, 32], sample_steps_per_sec=[100.0, 95.0, 70.0, 50.0, 40.0], max_batch_tokens=32)
        marg = compute_verify_token_budget(history_survival_probs=surv, sps_table=table, cfg=DSparkScheduleConfig(gamma=7))
        ratio = compute_verify_token_budget(
            history_survival_probs=surv, sps_table=table, cfg=DSparkScheduleConfig(gamma=7, budget_policy="ratio")
        )
        allowed = ratio_policy_budgets(num_requests=4, max_verify_len=8, num_candidates=28)
        self.assertIn(ratio.budget, allowed)
        vals = torch.sort(surv.flatten(), descending=True).values
        tau = lambda b: 4 + float(vals[:b].sum())
        sps = lambda n: [100.0, 95.0, 70.0, 50.0, 40.0][max(i for i, p in enumerate([4, 8, 16, 24, 32]) if p <= n)]
        self.assertEqual(ratio.budget, max(allowed, key=lambda b: tau(b) * sps(4 + b)))
        self.assertGreaterEqual(marg.predicted_theta, ratio.predicted_theta - 1e-9)

    def test_bad_policy_rejected(self):
        with self.assertRaises(ValueError):
            DSparkScheduleConfig(gamma=7, budget_policy="dcut").validate()


class TestFixedVerifyLen(CustomTestCase):
    def _planner(self, block=8):
        p = object.__new__(DSparkVerifyPlanner)
        p.verify_num_draft_tokens = block
        p._schedule_cfg = DSparkScheduleConfig(gamma=block - 1)
        p._budget_planner = types.SimpleNamespace(note_non_decode_step=lambda: None)
        p._fixed_verify_len = None
        p._tp_sync = _SyncStub()
        return p

    def test_fixed_len_overrides_confidence_and_budget(self):
        """The fixed-k arm verifies exactly k per request and sizes the graph
        tier budget to match, whatever the confidence says."""
        p = self._planner()
        p.set_fixed_verify_len(3)
        req = torch.arange(5)
        lens = p._schedule_verify_lens(
            req_pool_indices=req,
            prefix_lens=torch.zeros(5),
            device="cpu",
            confidence=torch.zeros(5, 7),
            budget=0,
        )
        self.assertEqual(lens.tolist(), [3] * 5)
        budget = p._budget_from_resolved(
            resolved=types.SimpleNamespace(), req_pool_indices_cpu=req
        )
        self.assertEqual(budget, 5 * (3 - 1))

    def test_fixed_len_needs_no_confidence(self):
        """With a pinned length the worker skips the confidence kernels: lens and
        budget must come out without confidence, a budget or a relayed value."""
        p = self._planner()
        p.set_fixed_verify_len(8)
        req = torch.arange(4)
        lens = p._schedule_verify_lens(
            req_pool_indices=req, prefix_lens=torch.zeros(4), device="cpu", confidence=None, budget=None
        )
        self.assertEqual(lens.tolist(), [8] * 4)
        self.assertEqual(p._budget_from_resolved(resolved=None, req_pool_indices_cpu=req), 4 * 7)
        p.set_fixed_verify_len(None)
        self.assertIsNone(p._budget_from_resolved(resolved=None, req_pool_indices_cpu=req))

    def test_worker_skips_confidence_under_fixed_len(self):
        w = object.__new__(DFlashWorkerV2)
        w._confidence_selector = object()  # would raise if the entropy path ran
        w._step_trace = types.SimpleNamespace(enabled=False)
        w._selector_step_q = (torch.zeros(2, 7, 16, dtype=torch.int64), torch.rand(2, 7, 16))
        w._verify_planner = types.SimpleNamespace(fixed_verify_len=8)
        self.assertEqual(
            w._step_confidence(drafted=torch.zeros(2, 7), req_pool_indices=torch.zeros(2, dtype=torch.long)),
            (None, None),
        )
        w._step_trace = types.SimpleNamespace(enabled=True)  # the trace still records the entropy
        with self.assertRaises(AttributeError):
            w._step_confidence(drafted=torch.zeros(2, 7), req_pool_indices=torch.zeros(2, dtype=torch.long))

    def test_fixed_len_bounds(self):
        p = self._planner()
        for bad in (0, 9):
            with self.assertRaises(ValueError):
                p.set_fixed_verify_len(bad)
        p.set_fixed_verify_len(8)
        p.set_fixed_verify_len(None)
        self.assertIsNone(p.fixed_verify_len)


class TestGreedyAcceptCutoff(CustomTestCase):
    def test_cutoff_caps_and_bonus_from_last_verified_row(self):
        """With verify_len v a request accepts at most v - 1 drafts and its bonus
        is the target argmax at the row of its last accepted draft."""
        bs, block, vocab = 4, 8, 32
        torch.manual_seed(0)
        target_predict = torch.randint(0, vocab, (bs, block))
        candidates = torch.empty(bs, block, dtype=torch.int64)
        candidates[:, 0] = 1
        candidates[:, 1:] = target_predict[:, :-1]  # every draft would match
        candidates[2, 4] = (target_predict[2, 3] + 1) % vocab  # row 2 misses at 3
        logits = torch.full((bs * block, vocab), -10.0)
        logits[torch.arange(bs * block), target_predict.view(-1)] = 10.0
        verify_lens = torch.tensor([8, 1, 6, 3])
        correct, bonus, trim = AcceptGreedy.torch(
            candidates=candidates,
            target_logits=logits,
            verify_num_draft_tokens=block,
            cutoff_verify_lens=verify_lens,
        )
        self.assertEqual(correct.tolist(), [7, 0, 3, 2])
        self.assertEqual(trim.tolist(), [0, 7, 0, 5])
        expect_bonus = [int(target_predict[i, c]) for i, c in enumerate(correct.tolist())]
        self.assertEqual(bonus.tolist(), expect_bonus)


@unittest.skipUnless(torch.cuda.is_available(), "accept_sampling is a Triton kernel")
class TestSparseQAcceptCutoffIsLossless(CustomTestCase):
    """DFlash2 at T=1: the selector's q lives on top-k candidates, scattered into a
    dense [bs, gamma, V] buffer for DSpark's accept_sampling. With a per-request cutoff
    every emitted position must still be distributed exactly as the target p of its row
    (the target rows here do not depend on the prefix, so each row's p is fixed)."""

    def test_emitted_positions_follow_target(self):
        from sglang.kernels.ops.speculative.dspark.dspark_accept import accept_sampling

        torch.manual_seed(0)
        dev, n, gamma, vocab, k = "cuda", 200_000, 3, 12, 4
        block = gamma + 1
        p_logits = torch.randn(block, vocab, device=dev) * 1.5
        p = torch.softmax(p_logits, -1)
        cand = torch.stack([torch.randperm(vocab, device=dev)[:k] for _ in range(gamma)])  # [gamma, K]
        q_sparse = torch.softmax(torch.randn(gamma, k, device=dev), -1)
        dense_q = torch.zeros(gamma, vocab, device=dev)
        dense_q.scatter_(1, cand, q_sparse)
        draws = torch.multinomial(q_sparse, n, replacement=True).T  # [n, gamma]
        drafts = torch.gather(cand.unsqueeze(0).expand(n, -1, -1), 2, draws.unsqueeze(-1)).squeeze(-1)
        candidates = torch.cat([torch.zeros(n, 1, dtype=torch.int64, device=dev), drafts], 1)
        info = types.SimpleNamespace(
            temperatures=torch.ones(n, 1, device=dev),
            need_top_k_sampling=False, need_top_p_sampling=False,
        )
        for v in (1, 2, 3, 4):
            correct, bonus, _ = accept_sampling(
                candidates=candidates,
                target_logits=p_logits.repeat(n, 1),
                draft_probs=dense_q.unsqueeze(0).expand(n, -1, -1).contiguous(),
                sampling_info=info,
                draft_input=types.SimpleNamespace(max_top_k=1, uniform_top_k_value=None),
                gamma=gamma,
                verify_num_draft_tokens=block,
                cutoff_verify_lens=torch.full((n,), v, device=dev, dtype=torch.int32),
            )
            self.assertLessEqual(int(correct.max()), v - 1)
            out = torch.cat([drafts, torch.zeros(n, 1, dtype=torch.int64, device=dev)], 1)
            out.scatter_(1, correct.long().view(-1, 1), bonus.view(-1, 1))
            for pos in range(v):  # every emitted position up to the bonus
                emitted = correct >= pos
                hist = torch.bincount(out[emitted, pos], minlength=vocab).float()
                tv = 0.5 * (hist / hist.sum() - p[pos]).abs().sum().item()
                # Deeper positions see only the draws accepted so far (~10k at pos 1, ~200 at
                # pos 3 here), so the bound scales with the multinomial noise of that count.
                floor = 0.5 * (2 * p[pos] * (1 - p[pos]) / (math.pi * hist.sum())).sqrt().sum().item()
                self.assertLess(tv, 0.002 + 4 * floor, f"v={v} pos={pos} TV={tv:.4f} noise={floor:.4f}")


@unittest.skipUnless(torch.cuda.is_available(), "compact_verify_ids_triton is a Triton kernel")
class TestCompactVerifyIdsAnchorStride(CustomTestCase):
    """The compact verify window takes each request's anchor from column 0 of its
    draft_block_ids row. DFLASH passes rows of gamma + 1 (anchor + drafts), DSpark rows
    of gamma; every request after the first must get its own anchor either way."""

    def test_triton_matches_torch_for_both_row_widths(self):
        from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
            compact_verify_ids,
            compact_verify_ids_triton,
        )

        torch.manual_seed(0)
        bs, gamma = 5, 7
        # 22 real tokens padded to a 24-token tier: the pad rows must come back as 0.
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 3, 1, 6, 4], device="cuda", grid=[24]
        )
        for cols in (gamma, gamma + 1):
            block = torch.randint(1000, (bs, cols), device="cuda")
            drafts = torch.randint(1000, (bs, gamma), device="cuda")
            ref = compact_verify_ids(draft_block_ids=block, draft_tokens=drafts, layout=layout, device="cuda")
            got = compact_verify_ids_triton(draft_block_ids=block, draft_tokens=drafts, layout=layout, device="cuda")
            torch.testing.assert_close(got, ref, msg=f"row width {cols}")


class TestDraftDependentCutoffIsLossless(CustomTestCase):
    """T=1 rejection sampling with the served cutoff rule: survivals from the walk's q
    rows, K from a lagged step, per-request top-K admission. Every emitted position
    must follow the target. K taken in-step from the current block's scores reads the
    slot's own token through its descendants' entropy (D-cut App. B) and must not."""

    B, GAMMA, V = 2, 3, 4

    def setUp(self):
        torch.manual_seed(0)
        self.q0 = torch.tensor([0.4, 0.2, 0.2, 0.2])
        # Near-certain after token 0, uniform elsewhere: slot k+1's entropy depends on x_k.
        self.trans = torch.full((self.V, self.V), 1.0 / self.V)
        self.trans[0] = torch.tensor([0.97, 0.01, 0.01, 0.01])
        self.p = torch.tensor([0.1, 0.3, 0.3, 0.3])  # every target row, prefix-independent
        # A step SPS table: past the first edge every slot is free, so K jumps from 0
        # to all slots on the batch's total survival, the case where in-step K leaks.
        table = SpsCostTable(
            sample_batch_tokens=[self.B, self.B + 1],
            sample_steps_per_sec=[1.0, 0.7],
            max_batch_tokens=self.B * (self.GAMMA + 1),
        )
        cfg = DSparkScheduleConfig(gamma=self.GAMMA)
        # A request's survivals depend on its walk prefix (x_0 .. x_{gamma-2}) only,
        # so the real planner is tabulated over every prefix of every request.
        prefixes = torch.cartesian_prod(*[torch.arange(self.V)] * (self.GAMMA - 1))
        surv = raw_survival_chain(torch.exp(-entropy_from_probs(self._q_rows(prefixes))))
        n_pre, n_k = len(prefixes), self.B * self.GAMMA + 1
        self.k_tab = torch.empty(n_pre, n_pre, dtype=torch.long)
        self.len_tab = torch.empty(n_pre, n_pre, n_k, self.B, dtype=torch.long)
        for a in range(n_pre):
            for b in range(n_pre):
                s = surv[[a, b]]
                self.k_tab[a, b] = compute_verify_token_budget(
                    history_survival_probs=s, sps_table=table, cfg=cfg
                ).budget
                for k in range(n_k):
                    self.len_tab[a, b, k] = schedule_verify_lens_topk_from_survival(
                        survival_probs=s, budget=k, cfg=cfg
                    )

    def _q_rows(self, prefix):
        return torch.stack(
            [self.q0.expand(len(prefix), -1)] + [self.trans[prefix[:, j]] for j in range(self.GAMMA - 1)], 1
        )

    def _walk(self, m):
        x = torch.empty(m, self.GAMMA, dtype=torch.long)
        q = self.q0.expand(m, -1)
        for j in range(self.GAMMA):
            x[:, j] = torch.multinomial(q, 1)[:, 0]
            q = self.trans[x[:, j]]
        return x

    def _prefix_id(self, x):
        idx = torch.zeros(len(x), dtype=torch.long)
        for j in range(self.GAMMA - 1):
            idx = idx * self.V + x[:, j]
        return idx

    def _emit(self, x, lens):
        """Chain rejection sampling with verify_len = lens (anchor included)."""
        m = len(x)
        q = self._q_rows(x)
        p = self.p.expand(m, -1)
        out = torch.zeros(m, self.GAMMA + 1, dtype=torch.long)
        num_out = torch.zeros(m, dtype=torch.long)
        alive = torch.ones(m, dtype=torch.bool)
        for j in range(self.GAMMA + 1):
            drafted = alive & (j < lens - 1)
            emitted = drafted | (alive & (j == lens - 1))
            tok = torch.multinomial(p, 1)[:, 0]  # bonus
            if j < self.GAMMA:
                xj, qj = x[:, j], q[:, j]
                accept = torch.rand(m) < (self.p[xj] / qj.gather(1, xj[:, None])[:, 0]).clamp(max=1)
                residual = torch.multinomial((p - qj).clamp(min=0), 1)[:, 0]
                tok = torch.where(drafted, torch.where(accept, xj, residual), tok)
                alive = drafted & accept
            out[:, j] = torch.where(emitted, tok, out[:, j])
            num_out += emitted.long()
        return out, num_out

    def _tv_per_position(self, k_rule):
        n = 100_000
        x = self._walk(n * self.B)
        pre = self._prefix_id(x).view(n, self.B)
        if k_rule == "verify_all":
            lens = torch.full((n * self.B,), self.GAMMA + 1)
        else:
            src = pre if k_rule == "in_step" else self._prefix_id(self._walk(n * self.B)).view(n, self.B)
            k = self.k_tab[src[:, 0], src[:, 1]]
            lens = self.len_tab[pre[:, 0], pre[:, 1], k].view(-1)
        out, num_out = self._emit(x, lens)
        tv = []
        for j in range(self.GAMMA + 1):
            sel = num_out > j
            if int(sel.sum()) >= 10_000:
                hist = torch.bincount(out[sel, j], minlength=self.V).float()
                tv.append(0.5 * float((hist / hist.sum() - self.p).abs().sum()))
        return tv

    def test_lagged_k_is_lossless_and_in_step_k_is_not(self):
        for rule in ("verify_all", "lagged"):
            tv = self._tv_per_position(rule)
            self.assertLess(max(tv), 0.01, f"{rule}: TV per position {tv}")
        tv = self._tv_per_position("in_step")
        self.assertGreater(max(tv), 0.025, f"in-step K should leak: TV per position {tv}")


class TestSpecMropeRagged(CustomTestCase):
    def _fb(self, bs):
        fb = object.__new__(ForwardBatch)
        fb.seq_lens = torch.zeros(bs, dtype=torch.int64)
        return fb

    def _batch(self, positions, mm_deltas, layout=None):
        mm = [
            None if d is None else types.SimpleNamespace(
                mrope_position_delta=torch.tensor([[d]])
            )
            for d in mm_deltas
        ]
        spec = types.SimpleNamespace(positions=positions, ragged_verify_layout=layout)
        return types.SimpleNamespace(multimodal_inputs=mm, spec_info=spec)

    def test_ragged_text_only_does_not_reshape_by_bs(self):
        """Regression: a ragged verify of 3 requests carries 7 padded rows, which
        view(bs, -1) rejected (the Qwen3.8 compact-verify crash)."""
        fb = self._fb(3)
        positions = torch.tensor([10, 11, 20, 30, 31, 32, 0])
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[2, 1, 3], device="cpu", grid=[7]
        )
        fb.compute_spec_mrope_positions(
            types.SimpleNamespace(device="cpu"),
            self._batch(positions, [None] * 3, layout),
        )
        torch.testing.assert_close(fb.mrope_positions, positions.repeat(3, 1))

    def test_ragged_deltas_follow_each_request(self):
        deltas = _spec_mrope_token_deltas(
            deltas=torch.tensor([5, 0, -2]),
            num_tokens=8,
            ragged_layout=types.SimpleNamespace(verify_lens=torch.tensor([2, 1, 3])),
        )
        self.assertEqual(deltas.tolist(), [5, 5, 0, -2, -2, -2, 0, 0])

    def test_uniform_matches_base(self):
        base = _load_base(
            "base_forward_batch_info", "sglang/srt/model_executor/forward_batch_info.py"
        )
        if base is None:
            self.skipTest("SGLANG_BASE_SRC not set")
        positions = torch.arange(12, dtype=torch.int64) + 100
        for mm in ([None, None, None], [None, 7, -3]):
            ours, ref = self._fb(3), object.__new__(base.ForwardBatch)
            ref.seq_lens = torch.zeros(3, dtype=torch.int64)
            runner = types.SimpleNamespace(device="cpu")
            ours.compute_spec_mrope_positions(runner, self._batch(positions, mm))
            ref.compute_spec_mrope_positions(runner, self._batch(positions, mm))
            torch.testing.assert_close(ours.mrope_positions, ref.mrope_positions)


class TestStaticPathUnchanged(CustomTestCase):
    """The static (verify-all) DFLASH path is the baseline arm: it must keep the
    base commit's gates and accept results."""

    def test_gates_closed_in_static_mode(self):
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            self.assertFalse(SpeculativeAlgorithm.DFLASH.supports_ragged_verify())
            self.assertTrue(SpeculativeAlgorithm.DSPARK.supports_ragged_verify())
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
            self.assertTrue(SpeculativeAlgorithm.DFLASH.supports_ragged_verify())

    def setUp(self):
        # The worker reads its TP rank from the runtime context on this tree.
        ctx = get_parallel().override(tp_rank=0)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)

    def _worker_stub(self):
        w = object.__new__(DFlashWorkerV2)
        w.model_runner = types.SimpleNamespace(tp_rank=0)
        w.selector = object()
        w.block_size = 8
        w.device = "cpu"
        w._tp_sync = _SyncStub()
        w._selector_sample = None
        w._use_triton_accept_bonus = False
        return w

    def test_static_mode_builds_no_planner(self):
        w = self._worker_stub()
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"), (
            envs.SGLANG_SPEC_STEP_TRACE_PATH.override("")
        ):
            w._init_dynamic_verify()
        self.assertIsNone(w._verify_planner)
        self.assertIsNone(w._confidence_selector)
        self.assertFalse(w.carries_confidence)
        self.assertIsNone(w.get_confidence_budget_prepare())
        w._selector_step_q = (torch.zeros(2, 7, 16, dtype=torch.int64), torch.rand(2, 7, 16))
        self.assertEqual(w._step_confidence(drafted=torch.zeros(2, 7), req_pool_indices=torch.zeros(2, dtype=torch.long)), (None, None))
        self.assertEqual(
            w._plan_verify(batch=None, draft_input=None, confidence=None, prefix_lens=None),
            (None, None),
        )

    def test_ragged_mode_needs_a_selector_draft(self):
        w = self._worker_stub()
        w.selector = None
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
            with self.assertRaises(ValueError):
                w._init_dynamic_verify()

    def test_greedy_accept_matches_base(self):
        base = _load_base(
            "base_dflash_worker_v2", "sglang/srt/speculative/dflash_worker_v2.py"
        )
        if base is None:
            self.skipTest("SGLANG_BASE_SRC not set")
        torch.manual_seed(0)
        bs, block, vocab = 5, 8, 40
        candidates = torch.randint(0, vocab, (bs, block))
        logits = torch.randn(bs * block, vocab)
        # Make some drafts match so accept lengths vary.
        top = logits.view(bs, block, vocab).argmax(-1)
        candidates[:, 1:4] = top[:, :3]
        kwargs = dict(
            candidates=candidates,
            next_token_logits=logits,
            sampling_info=None,
            draft_input=None,
            prefix_lens=torch.full((bs,), 100),
            bs=bs,
        )
        ours = DFlashWorkerV2._accept_block(self._worker_stub(), **kwargs)
        ref = base.DFlashWorkerV2._accept_block(self._worker_stub(), **kwargs)
        # This tree's _accept_block returns (accept, commit, bonus, out, new_seq_lens)
        # and we append cap_trim_lens, None on the static path.
        self.assertEqual(len(ours), 6)
        self.assertEqual(len(ref), 5)
        self.assertIsNone(ours[5])
        for a, b in zip(ours[:5], ref):
            if a is None or b is None:
                self.assertIs(a, b)
            else:
                torch.testing.assert_close(a, b)


class TestDSparkEntropySource(CustomTestCase):
    def _worker(self, corrected):
        w = object.__new__(DSparkWorkerV2)
        rng = np.random.default_rng(2)
        tx = np.minimum.accumulate(np.cumprod(rng.uniform(0, 1, (1000, 4)), 1), 1)
        cal = EntropyCalibrator(table=_export_table(tx, tx), gamma=4, device="cpu")
        w._confidence_selector = types.SimpleNamespace(
            uses_entropy=True, uses_draft_prob=False, calibrator=cal, source=ConfidenceSource.ENTROPY,
            entropy_confidence=lambda *, entropy, req_pool_indices: entropy_confidence(entropy=entropy, calibrator=cal),
        )
        block = types.SimpleNamespace(
            corrected_logits=corrected, temperatures=torch.ones(2)
        )
        return w, types.SimpleNamespace(draft_block=block), cal

    def test_entropy_is_the_probe_signal(self):
        """Served entropy = entropy of softmax(corrected_logits / T), the signal the
        offline tables were fitted on (dspark_observability q_ent)."""
        torch.manual_seed(0)
        corrected = torch.randn(2, 4, 30)
        w, proposal, cal = self._worker(corrected)
        conf, ent = w._confidence_for_budget(
            proposal=proposal, draft_block_ids=None, draft_tokens=None,
                req_pool_indices=torch.zeros(2, dtype=torch.long),
        )
        q = torch.softmax(corrected, -1)
        torch.testing.assert_close(ent, -(q * q.log()).sum(-1))
        torch.testing.assert_close(conf, entropy_confidence(entropy=ent, calibrator=cal))

    def test_missing_logits_raise(self):
        w, proposal, _ = self._worker(None)
        with self.assertRaises(RuntimeError):
            w._confidence_for_budget(
                proposal=proposal, draft_block_ids=None, draft_tokens=None,
                req_pool_indices=torch.zeros(2, dtype=torch.long),
            )


if __name__ == "__main__":
    assert not torch.cuda.is_available(), "CPU-only test run must not see a GPU"
    unittest.main(verbosity=2)
