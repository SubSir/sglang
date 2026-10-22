"""GDN ReplaySSM spec-verify ring: configuration checks and commit routing.

Under ``--enable-linear-replayssm-spec`` a GDN verify appends the draft window
to a compact request ring of ``--linear-replayssm-cache-len`` positions and the
commit folds the accepted prefix into the checkpoint.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_KERNELS = "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode"
DRAFT_TOKENS = 8


def _gdn_cache_params(num_layers=2):
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=64,
        n_groups=2,
        num_heads=4,
        head_dim=16,
        state_size=16,
        conv_kernel=4,
    )
    dtype = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.bfloat16)
    return Mamba2CacheParams(shape=shape, dtype=dtype, layers=list(range(num_layers)))


def _spec_state(*, intermediate_ssm=None, temporal_dtype=torch.bfloat16):
    temporal = torch.zeros(1, 8, 2, 4, 4, dtype=temporal_dtype)
    return MambaPool.SpeculativeState(
        conv=[torch.zeros(1, 8, 3, 2)],
        temporal=temporal,
        intermediate_ssm=intermediate_ssm,
        intermediate_conv_window=[torch.zeros(1, 4, DRAFT_TOKENS, 3, 2)],
        replayssm_d=torch.zeros(1, 4, 2, 16, 4),
    )


def _gdn_pool():
    return SimpleNamespace(
        replayssm_is_kda=False,
        replayssm_spec_fold=False,
        replayssm_spec_write_pos=torch.zeros(40, dtype=torch.int32),
        replayssm_cache_base=torch.zeros(40, dtype=torch.int32),
        replayssm_is_flush=torch.zeros(40, dtype=torch.int8),
    )


class TestGDNReplaySSMRingLength(CustomTestCase):
    @staticmethod
    def _pool(ring_len, draft_tokens=DRAFT_TOKENS):
        return MambaPool(
            size=4,
            spec_state_size=3,
            cache_params=_gdn_cache_params(),
            mamba_layer_ids=[0, 1],
            device="cpu",
            speculative_num_draft_tokens=draft_tokens,
            speculative_eagle_topk=None,
            linear_replayssm_cache_len=ring_len,
            enable_linear_replayssm_spec=True,
        )

    def test_ring_must_hold_the_draft_window(self):
        with self.assertRaisesRegex(ValueError, "ring too small"):
            self._pool(ring_len=16, draft_tokens=24)
        # The compact commit needs at least 16 ring positions.
        with self.assertRaisesRegex(ValueError, "ring too small"):
            self._pool(ring_len=DRAFT_TOKENS)
        with self.assertRaisesRegex(ValueError, "power of two"):
            self._pool(ring_len=12)

    def test_ring_without_snapshots(self):
        for ring_len in (16, 32):
            with self.subTest(ring_len=ring_len):
                pool = self._pool(ring_len=ring_len)
                state = pool.get_speculative_mamba2_params_all_layers()
                hv, v_dim, _ = state.temporal.shape[2:]
                self.assertIsNone(state.intermediate_ssm)
                # Ring rows follow request rows (spec_state_size + 1).
                self.assertEqual(
                    tuple(state.replayssm_d.shape), (2, 4, hv, ring_len, v_dim)
                )
                self.assertEqual(pool.replayssm_cache_base.shape[0], 4)


class TestGDNReplaySSMGenericCommit(CustomTestCase):
    def test_cursor_advance_gets_the_state_slots(self):
        """The cursor advance skips the rows the verify and fold skip (state
        slot -1), so it gets the same state slots as the fold."""
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        target_worker = MagicMock()
        req_pool = target_worker.model_runner.req_to_token_pool
        req_pool.mamba_pool = _gdn_pool()
        slots = torch.tensor([5, -1])
        req_pool.get_mamba_indices.return_value = slots
        req_pool.get_speculative_mamba2_params_all_layers.return_value = _spec_state()
        batch = MagicMock()
        batch.forward_mode.is_idle.return_value = False
        batch.req_pool_indices = torch.tensor([17, 23])
        batch.mamba_track_indices = None
        accept_lens = torch.tensor([2, 1], dtype=torch.int32)
        accept_index = torch.tensor([[0, 1, -1], [3, -1, -1]], dtype=torch.int32)

        with (
            patch(
                "sglang.srt.speculative.spec_utils.mambaish_config",
                return_value={"some": "config"},
            ),
            patch(f"{_KERNELS}.commit_gdn_replayssm_spec") as advance,
            patch(f"{_KERNELS}.commit_gdn_replayssm_circular") as fold,
            patch(
                "sglang.kernels.ops.mamba.mamba_state_scatter_triton."
                "fused_conv_window_scatter_with_mask"
            ),
        ):
            commit_mamba_states_after_verify(
                target_worker,
                batch,
                accept_lens,
                accept_index,
                draft_token_num=3,
            )

        self.assertIs(advance.call_args.kwargs["state_batch_indices"], slots)
        self.assertIs(fold.call_args.kwargs["state_batch_indices"], slots)


if __name__ == "__main__":
    unittest.main()
