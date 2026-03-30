# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Alignment tests: DFlash tree-verify Triton fused kernel vs original PyTorch loop."""

import os
import time
import unittest

import torch

from sglang.srt.speculative.dflash_utils import build_tree_verify_tokens
from sglang.srt.speculative.triton_ops.dflash_tree_expand_topk import (
    dflash_expand_topk4,
    dflash_tree_verify_select_topk4_fused,
)
from sglang.srt.utils import is_cuda
from sglang.srt.utils.common import fast_topk
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, suite="stage-b-test-small-1-gpu")

TOPK = 4


def _reference_tree_intermediates(
    topk_probs: torch.Tensor,
    topk_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same tensors as ``build_tree_verify_tokens`` before global top-k (pure PyTorch)."""
    bs, num_steps, k = topk_probs.shape
    assert k == TOPK
    device = topk_probs.device
    expanded_topk_probs = topk_probs[:, 1:].repeat_interleave(TOPK, dim=0)
    expanded_topk_ids = topk_ids[:, 1:].repeat_interleave(TOPK, dim=0)
    score_list: list[torch.Tensor] = []
    token_list: list[torch.Tensor] = []
    parents_list: list[torch.Tensor] = []
    scores: torch.Tensor | None = None

    for i in range(num_steps):
        if i == 0:
            step_topk_p = topk_probs[:, 0]
            step_topk_ids = topk_ids[:, 0]
        else:
            step_topk_p = expanded_topk_probs[:, i - 1]
            step_topk_ids = expanded_topk_ids[:, i - 1]

        if i == 0:
            scores = step_topk_p
            tree_info = (
                step_topk_p.unsqueeze(1),
                step_topk_ids,
                torch.arange(-1, TOPK, dtype=torch.long, device=device)
                .unsqueeze(0)
                .repeat(bs, 1),
            )
        else:
            assert scores is not None
            expand_scores = torch.mul(
                scores.unsqueeze(2), step_topk_p.reshape(-1, TOPK, TOPK)
            )
            topk_cs_p, topk_cs_index = fast_topk(
                expand_scores.flatten(start_dim=1), TOPK, dim=-1
            )
            scores = topk_cs_p
            tok = step_topk_ids.reshape(-1, TOPK * TOPK)
            tree_info = (
                expand_scores,
                tok,
                topk_cs_index + (TOPK * TOPK * (i - 1) + TOPK),
            )

        score_list.append(tree_info[0])
        token_list.append(tree_info[1])
        parents_list.append(tree_info[2])

    score_list_cat = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        parent_list = torch.empty((bs, 0), dtype=torch.long, device=device)
    return score_list_cat, ss_token_list, parent_list


def _reference_build_tree_verify(
    verified_id: torch.Tensor,
    topk_probs: torch.Tensor,
    topk_ids: torch.Tensor,
    num_draft_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    score_list_cat, ss_token_list, parent_list = _reference_tree_intermediates(
        topk_probs, topk_ids
    )
    top_scores = fast_topk(score_list_cat, num_draft_tokens - 1, dim=-1)
    top_scores_index = torch.sort(top_scores.indices).values
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)
    draft_tokens = torch.cat([verified_id[:, None], draft_tokens], dim=1).flatten()
    return draft_tokens, parent_list, top_scores_index


@unittest.skipUnless(is_cuda(), "CUDA required for Triton DFlash tree-verify tests")
class TestDflashTreeVerifyTriton(CustomTestCase):
    def test_fused_kernel_matches_reference_intermediates(self):
        device = "cuda"
        for num_steps in (1, 2, 5, 14):
            for batch_size in (1, 8, 32):
                torch.manual_seed(num_steps * 17 + batch_size)
                probs = torch.rand(batch_size, num_steps, TOPK, device=device, dtype=torch.float32)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                ids = torch.randint(
                    0,
                    32000,
                    (batch_size, num_steps, TOPK),
                    device=device,
                    dtype=torch.long,
                )
                ref_sc, ref_tok, ref_par = _reference_tree_intermediates(probs, ids)
                tri_sc, tri_tok, tri_par = dflash_tree_verify_select_topk4_fused(probs, ids)

                self.assertEqual(ref_sc.shape, tri_sc.shape)
                self.assertEqual(ref_tok.shape, tri_tok.shape)
                self.assertEqual(ref_par.shape, tri_par.shape)
                max_sc = (ref_sc - tri_sc).abs().max().item()
                self.assertLess(
                    max_sc,
                    1e-4,
                    msg=f"score mismatch num_steps={num_steps} bs={batch_size} max_abs={max_sc}",
                )
                self.assertTrue(
                    torch.equal(ref_tok, tri_tok),
                    msg=f"token ids num_steps={num_steps} bs={batch_size}",
                )
                self.assertTrue(
                    torch.equal(ref_par, tri_par),
                    msg=f"parents num_steps={num_steps} bs={batch_size}",
                )

    def test_fused_kernel_matches_reference_bfloat16(self):
        device = "cuda"
        torch.manual_seed(0)
        batch_size, num_steps = 2, 4
        probs = torch.rand(batch_size, num_steps, TOPK, device=device, dtype=torch.bfloat16)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        ids = torch.randint(
            0, 32000, (batch_size, num_steps, TOPK), device=device, dtype=torch.long
        )
        ref_sc, ref_tok, ref_par = _reference_tree_intermediates(probs, ids)
        tri_sc, tri_tok, tri_par = dflash_tree_verify_select_topk4_fused(probs, ids)
        max_sc = (ref_sc.float() - tri_sc.float()).abs().max().item()
        self.assertLess(max_sc, 5e-3)
        self.assertTrue(torch.equal(ref_tok, tri_tok))
        self.assertTrue(torch.equal(ref_par, tri_par))

    def test_expand_topk4_single_step_matches_torch(self):
        device = "cuda"
        torch.manual_seed(1)
        batch_size = 5
        scores = torch.rand(batch_size, TOPK, device=device, dtype=torch.float32).abs()
        topk_p = torch.rand(batch_size * TOPK, TOPK, device=device, dtype=torch.float32)
        topk_p = topk_p.softmax(dim=-1)
        ex_t, tv_t, ti_t = dflash_expand_topk4(scores, topk_p)
        ref_ex = scores.unsqueeze(2) * topk_p.view(batch_size, TOPK, TOPK)
        ref_v, ref_i = torch.topk(ref_ex.flatten(1), TOPK, dim=-1)
        self.assertTrue(torch.allclose(tv_t, ref_v, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.equal(ti_t, ref_i))
        self.assertLess((ex_t - ref_ex).abs().max().item(), 1e-6)

    def test_build_tree_verify_tokens_matches_reference(self):
        device = "cuda"
        for num_steps in (2, 4):
            batch_size = 2
            torch.manual_seed(40 + num_steps)
            verified = torch.zeros(batch_size, dtype=torch.long, device=device)
            probs = torch.rand(batch_size, num_steps, TOPK, device=device, dtype=torch.float32)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            ids = torch.randint(
                0, 32000, (batch_size, num_steps, TOPK), device=device, dtype=torch.long
            )
            max_cand = TOPK + TOPK * TOPK * (num_steps - 1)
            num_draft_tokens = min(8, max_cand) + 1

            got = build_tree_verify_tokens(
                verified_id=verified,
                topk_probs=probs.clone(),
                topk_ids=ids.clone(),
                topk=TOPK,
                num_draft_tokens=num_draft_tokens,
            )
            ref = _reference_build_tree_verify(verified, probs, ids, num_draft_tokens)
            self.assertTrue(torch.equal(got[0], ref[0]), msg=f"draft_tokens n={num_steps}")
            self.assertTrue(torch.equal(got[1], ref[1]), msg=f"parent_list n={num_steps}")
            self.assertTrue(torch.equal(got[2], ref[2]), msg=f"selected_index n={num_steps}")

    @unittest.skipIf(
        os.environ.get("SGLANG_SKIP_SPEED_TESTS", ""),
        "Set SGLANG_SKIP_SPEED_TESTS to skip timing assertions (e.g. noisy CI).",
    )
    def test_fused_tree_intermediates_speed_vs_pytorch_reference(self):
        """Fused Triton kernel should beat the Python + fast_topk loop on a non-trivial batch."""
        device = "cuda"
        batch_size = 128
        num_steps = 14
        n_warmup = 8
        n_iter = 64

        torch.manual_seed(12345)
        probs = torch.rand(batch_size, num_steps, TOPK, device=device, dtype=torch.float32)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        ids = torch.randint(
            0, 32000, (batch_size, num_steps, TOPK), device=device, dtype=torch.long
        )

        for _ in range(n_warmup):
            _reference_tree_intermediates(probs, ids)
            dflash_tree_verify_select_topk4_fused(probs, ids)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iter):
            _reference_tree_intermediates(probs, ids)
        torch.cuda.synchronize()
        t_ref = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n_iter):
            dflash_tree_verify_select_topk4_fused(probs, ids)
        torch.cuda.synchronize()
        t_fused = time.perf_counter() - t0

        ms_ref = 1e3 * t_ref / n_iter
        ms_fused = 1e3 * t_fused / n_iter
        speedup = t_ref / t_fused if t_fused > 0 else float("inf")
        print(
            f"[dflash tree-verify speed] bs={batch_size} num_steps={num_steps} "
            f"ref={ms_ref:.4f} ms/it fused={ms_fused:.4f} ms/it speedup={speedup:.2f}x"
        )

        self.assertGreater(
            speedup,
            1.15,
            msg=(
                f"Expected fused kernel >1.15x faster than PyTorch reference; "
                f"got speedup={speedup:.2f}x (ref {ms_ref:.4f} ms/it vs fused {ms_fused:.4f} ms/it). "
                f"Set SGLANG_SKIP_SPEED_TESTS=1 to skip."
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
