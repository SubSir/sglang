"""GDN ReplaySSM spec-verify request cursors.

Each request row keeps a ring origin (``cache_base``), the number of committed
records past it (``write_pos``) and whether the next commit folds them
(``is_flush``). The commit advances them only for rows that have a state slot,
as the verify, the fold and the finish kernels do.
"""

import unittest

import torch

from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_spec,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")

WIDTH, RING = 8, 16
NUM_ROWS = 12  # request rows: spec_state_size + 1


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGDNReplaySSMSpecCursors(CustomTestCase):
    def test_rows_without_state_slot_keep_their_cursors(self):
        def cursor(values, dtype):
            out = torch.zeros(NUM_ROWS, device="cuda", dtype=dtype)
            for row, value in values.items():
                out[row] = value
            return out

        # Rows 3 and 7 have slots (7 folds this commit); 5 and 9 have none.
        write_pos = cursor({3: 2, 5: 1, 7: 4, 11: 6}, torch.int32)
        cache_base = cursor({3: 5, 5: 3, 7: 14, 9: 9, 11: 1}, torch.int32)
        is_flush = cursor({5: 1, 7: 1, 11: 1}, torch.int8)
        before = [t.clone() for t in (write_pos, cache_base, is_flush)]
        commit_gdn_replayssm_spec(
            write_pos=write_pos,
            cache_base=cache_base,
            is_flush=is_flush,
            num_accepted=torch.tensor([3, 2, 8, 1], device="cuda"),
            replay_indices=torch.tensor([3, 5, 7, 9], device="cuda"),
            state_batch_indices=torch.tensor(
                [2, -1, 4, -1], device="cuda", dtype=torch.int32
            ),
            max_cache_len=RING,
            max_spec_len=WIDTH,
            fold_every_commit=True,
            null_block_id=-1,
        )

        expected = [t.clone() for t in before]
        # Row 3 appends its 3 accepted records; row 7 first retires its folded
        # 4 records (origin 14 + 4 wraps to 2), then holds the 8 new ones.
        for cursor_t, row3, row7 in zip(expected, (5, 5, 1), (8, 2, 1)):
            cursor_t[3], cursor_t[7] = row3, row7
        for name, actual, want in zip(
            ("write_pos", "cache_base", "is_flush"),
            (write_pos, cache_base, is_flush),
            expected,
        ):
            self.assertEqual(actual.tolist(), want.tolist(), name)


if __name__ == "__main__":
    unittest.main()
