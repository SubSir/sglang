"""Dynamic verify (entropy confidence, DFLASH ragged verify): CPU checks.

Base-parity cases load the unmodified worker/forward-batch sources from
SGLANG_BASE_SRC (the image's /sgl-workspace/sglang/python) and are skipped
without it.
"""

import importlib.util
import os
import types
import unittest

import numpy as np
import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptGreedy
from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    ScheduleVerifyLensTopk,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    _spec_mrope_token_deltas,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    DSparkVerifyPlanner,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.spec_confidence import (
    ConfidenceSource,
    EntropyCalibrationTable,
    EntropyCalibrator,
    conditional_from_survival,
    entropy_confidence,
    entropy_from_logits,
    raw_survival_chain,
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
        self.assertEqual(w._step_confidence(), (None, None))
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
        self.assertEqual(len(ours), 7)
        self.assertIsNone(ours[6])
        for a, b in zip(ours[:6], ref):
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
            uses_entropy=True, calibrator=cal, source=ConfidenceSource.ENTROPY
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
            proposal=proposal, draft_block_ids=None, draft_tokens=None
        )
        q = torch.softmax(corrected, -1)
        torch.testing.assert_close(ent, -(q * q.log()).sum(-1))
        torch.testing.assert_close(conf, entropy_confidence(entropy=ent, calibrator=cal))

    def test_missing_logits_raise(self):
        w, proposal, _ = self._worker(None)
        with self.assertRaises(RuntimeError):
            w._confidence_for_budget(
                proposal=proposal, draft_block_ids=None, draft_tokens=None
            )


if __name__ == "__main__":
    assert not torch.cuda.is_available(), "CPU-only test run must not see a GPU"
    unittest.main(verbosity=2)
