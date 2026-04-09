from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dflash_utils import (
    compute_dflash_accept_len_and_bonus,
    compute_dflash_sampling_accept_len_and_bonus,
    is_dflash_sampling_verify_available,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func


def _compute_paged_keep_slots(
    *,
    prefix_lens: torch.Tensor,
    commit_lens: torch.Tensor,
    draft_token_num: int,
    page_size: int,
) -> torch.Tensor:
    """Compute how many draft slots per request must remain allocated.

    The allocator frees at page granularity for paged mode, so we can only release
    full pages from the tail after verify.
    """

    if page_size <= 1:
        raise ValueError(f"Expected page_size > 1, got {page_size}.")

    seq_dtype = prefix_lens.dtype
    extended_lens = prefix_lens + int(draft_token_num)
    new_lens = prefix_lens + commit_lens.to(seq_dtype)
    aligned_new_lens = ((new_lens + page_size - 1) // page_size) * page_size
    keep_lens = torch.minimum(aligned_new_lens, extended_lens)
    keep_slots = (keep_lens - prefix_lens).to(torch.int64)
    keep_slots.clamp_(min=0, max=int(draft_token_num))
    return keep_slots


@dataclass
class DFlashDraftInput(SpecInput):
    """Per-batch DFlash draft state for spec-v1 (non-overlap) scheduling.

    This object is stored on `ScheduleBatch.spec_info` between decode iterations.
    It is NOT sent to model attention backends; the DFlash worker uses it to run
    the draft model and to track draft-side cache progress.

    When draft windowing is disabled, `draft_seq_lens` matches the committed target
    prefix length already materialized in the draft KV cache. When windowing is
    enabled, `draft_seq_lens` is the logical resident length in the draft worker's
    compact req-to-token mapping. In paged mode this may exceed the requested
    window by up to `page_size - 1` so the local page table remains valid. `ctx_lens`
    tracks newly committed target tokens that still need draft KV materialization.
    """

    # Current token to start the next DFlash block (one per request).
    verified_id: torch.Tensor

    # Flattened context features for tokens that need to be appended into the draft cache.
    # Shape: [sum(ctx_lens), K * hidden_size], where K is the number of target-layer
    # hidden-state features concatenated per token (len(dflash_config.target_layer_ids),
    # or default K == draft_num_layers for existing checkpoints).
    target_hidden: torch.Tensor

    # Context lengths per request, used to slice `target_hidden`. Device tensor (int32).
    ctx_lens: torch.Tensor

    # How many committed tokens are visible to the draft worker per request.
    draft_seq_lens: torch.Tensor

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # Draft state does not change token accounting.
        return (1, 1)

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        old_ctx_lens = self.ctx_lens
        old_target_hidden = self.target_hidden

        self.verified_id = self.verified_id[new_indices]
        self.ctx_lens = old_ctx_lens[new_indices]
        self.draft_seq_lens = self.draft_seq_lens[new_indices]

        if old_target_hidden is None or old_target_hidden.numel() == 0:
            self.target_hidden = old_target_hidden
            return

        # Rebuild target_hidden for the filtered batch using vectorized indexing.
        old_bs = int(old_ctx_lens.shape[0])
        offsets = torch.zeros(
            (old_bs + 1,), dtype=torch.int64, device=old_ctx_lens.device
        )
        offsets[1:].copy_(old_ctx_lens.to(torch.int64).cumsum(0))

        start = offsets[:-1]
        seg_start = start[new_indices]
        seg_lens = old_ctx_lens[new_indices].to(torch.int64)

        max_len = int(seg_lens.max().item()) if seg_lens.numel() > 0 else 0
        if max_len <= 0:
            self.target_hidden = old_target_hidden[:0]
            return

        r = torch.arange(max_len, device=old_ctx_lens.device, dtype=torch.int64)[
            None, :
        ]
        pos2d = seg_start[:, None] + r
        mask = r < seg_lens[:, None]
        flat_pos = pos2d[mask]
        self.target_hidden = (
            old_target_hidden.index_select(0, flat_pos)
            if flat_pos.numel() > 0
            else old_target_hidden[:0]
        )

    def merge_batch(self, spec_info: "DFlashDraftInput"):
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], dim=0)
        self.ctx_lens = torch.cat([self.ctx_lens, spec_info.ctx_lens], dim=0)
        self.draft_seq_lens = torch.cat(
            [self.draft_seq_lens, spec_info.draft_seq_lens], dim=0
        )
        if self.target_hidden is None or self.target_hidden.numel() == 0:
            self.target_hidden = spec_info.target_hidden
        elif (
            spec_info.target_hidden is not None and spec_info.target_hidden.numel() > 0
        ):
            self.target_hidden = torch.cat(
                [self.target_hidden, spec_info.target_hidden], dim=0
            )


@dataclass
class DFlashVerifyInput(SpecInput):
    """Inputs for a target-model verify forward in DFlash (spec-v1).

    The verify forward is run with `ForwardMode.TARGET_VERIFY` so that the target
    model returns logits for all tokens in the block, enabling accept-length
    computation.
    """

    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    # Kept for compatibility with attention backends that gate tree metadata by `topk > 1`.
    # DFLASH verify is linear (non-tree), so this is always 1.
    topk: int = 1
    # Custom attention "allow mask" for TARGET_VERIFY in backends that require it (e.g. triton).
    # Semantics follow SGLang speculative conventions: True means the (q, k) pair is allowed.
    custom_mask: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Per-request actual verify block sizes for dynamic VBS (tight packing).
    # Shape [bs], int32. None when using fixed block size.
    per_request_vbs: Optional[torch.Tensor] = None
    # Cumulative sum of per_request_vbs, shape [bs+1], starts from 0.
    per_request_vbs_cumsum: Optional[torch.Tensor] = None
    # Total number of real (non-padding) tokens in the packed layout.
    total_real_tokens: int = -1

    # Shape info for padding (e.g., DP attention / CUDA graph).
    num_tokens_per_batch: int = -1

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)
        if self.num_tokens_per_batch == -1:
            self.num_tokens_per_batch = int(self.draft_token_num)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def _is_tight_packed(self) -> bool:
        """Whether tokens are tightly packed (variable per-request sizes)."""
        return self.per_request_vbs_cumsum is not None

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        page_size: int,
        *,
        build_custom_mask: bool = True,
    ):
        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token
        tight = self._is_tight_packed()

        # Per-request end offsets for KV cache allocation.
        if tight:
            end_offset = batch.seq_lens + self.per_request_vbs.to(batch.seq_lens.dtype)
        else:
            end_offset = batch.seq_lens + self.draft_token_num

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache, len(batch.input_ids)
            )
        else:
            if tight:
                end_offset_cpu = batch.seq_lens_cpu + self.per_request_vbs.cpu().to(
                    batch.seq_lens_cpu.dtype
                )
            else:
                end_offset_cpu = batch.seq_lens_cpu + self.draft_token_num
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )
            self.last_loc = last_loc

        bs = batch.batch_size()
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        if not build_custom_mask:
            self.custom_mask = None
            return

        if self.draft_token_num <= 0:
            raise ValueError(
                f"DFLASH draft_token_num must be positive, got {self.draft_token_num}."
            )

        # Build custom attention mask.
        mask_chunks: List[torch.Tensor] = []
        per_vbs_cpu = (
            self.per_request_vbs.cpu().tolist()
            if self.per_request_vbs is not None
            else None
        )
        for idx, prefix_len in enumerate(batch.seq_lens_cpu.tolist()):
            prefix_len_i = int(prefix_len)
            # With tight packing, each request has its own q_len.
            if per_vbs_cpu is not None:
                req_q_len = int(per_vbs_cpu[idx])
            else:
                req_q_len = int(self.draft_token_num)

            kv_len = prefix_len_i + req_q_len
            q_idx = torch.arange(req_q_len, device=batch.device, dtype=torch.int32).unsqueeze(1)
            k_idx = torch.arange(kv_len, device=batch.device, dtype=torch.int32).unsqueeze(0)
            allow = k_idx <= (prefix_len_i + q_idx)
            mask_chunks.append(allow.flatten())

        self.custom_mask = (
            torch.cat(mask_chunks, dim=0)
            if mask_chunks
            else torch.empty((0,), dtype=torch.bool, device=batch.device)
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        bs = len(req_pool_indices)
        tight = self._is_tight_packed()

        if tight and self.per_request_vbs is not None:
            # Variable per-request query lengths.
            # Pad per_vbs to bs (CUDA graph may pad bs beyond real batch size).
            raw_vbs = self.per_request_vbs
            if raw_vbs.shape[0] < bs:
                per_vbs = torch.zeros(bs, dtype=raw_vbs.dtype, device=device)
                per_vbs[:raw_vbs.shape[0]] = raw_vbs
            else:
                per_vbs = raw_vbs[:bs]

            # Build variable qo_indptr from per-request VBS cumsum.
            qo_cumsum = torch.cumsum(per_vbs.to(torch.int32), dim=0)
            qo_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
            qo_indptr[1:] = qo_cumsum

            # Per-request KV lengths = prefix + per_req_vbs.
            paged_kernel_lens = paged_kernel_lens + per_vbs.to(paged_kernel_lens.dtype)
        else:
            qo_indptr = torch.arange(
                0,
                (bs + 1) * self.draft_token_num,
                step=self.draft_token_num,
                dtype=torch.int32,
                device=device,
            )
            paged_kernel_lens = paged_kernel_lens + self.draft_token_num

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        total_kv = int(cum_kv_seq_len[-1].item())
        kv_indices = torch.empty(total_kv, dtype=torch.int32, device=device)
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )

        mask = self.custom_mask
        if mask is not None:
            # Compute expected mask size: sum over requests of (kv_len_i * q_len_i).
            if tight and self.per_request_vbs is not None:
                # With variable q_len, mask_numel = sum(kv_len_i * q_len_i)
                # = sum((prefix_i + vbs_i) * vbs_i)
                # Approximate upper bound for padding check:
                mask_numel = int((paged_kernel_lens * per_vbs.to(paged_kernel_lens.dtype)).sum().item())
            else:
                mask_numel = (
                    paged_kernel_lens_sum * self.draft_token_num
                    + (self.draft_token_num**2) * bs
                )
            if mask.numel() < mask_numel:
                mask = torch.cat(
                    [
                        mask,
                        torch.full(
                            (mask_numel - mask.numel(),),
                            True,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
                self.custom_mask = mask
        return kv_indices, cum_kv_seq_len, qo_indptr, mask

    def verify(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """DFlash verification for greedy and non-greedy sampling.

        Returns:
            new_verified_id: int64 tensor [bs] (the new current token per request)
            commit_lens: int32 tensor [bs] (how many verify-input tokens are committed)
            next_target_hidden: tensor [sum(commit_lens), feature_dim]
            accept_length_per_req_cpu: list[int] (accepted draft tokens per request)
        """
        if batch.forward_mode.is_idle():
            empty = torch.empty((0,), dtype=torch.int64, device=batch.device)
            return empty, empty.to(torch.int32), empty, []

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device
        tight = self._is_tight_packed()

        sampling_info = batch.sampling_info
        if sampling_info is not None:
            if len(sampling_info) != bs:
                raise RuntimeError(
                    "DFLASH verify sampling_info size mismatch: "
                    f"len(sampling_info)={len(sampling_info)}, bs={bs}."
                )

            if sampling_info.has_custom_logit_processor:
                apply_custom_logit_processor(
                    logits_output.next_token_logits,
                    sampling_info,
                    num_tokens_in_batch=self.draft_token_num,
                )

            if (
                sampling_info.penalizer_orchestrator.is_required
                or sampling_info.logit_bias is not None
            ):
                linear_penalty = torch.zeros(
                    (bs, logits_output.next_token_logits.shape[1]),
                    dtype=torch.float32,
                    device=device,
                )
                sampling_info.apply_logits_bias(linear_penalty)
                logits_output.next_token_logits.add_(
                    torch.repeat_interleave(linear_penalty, self.draft_token_num, dim=0)
                )

        if tight:
            # Unpack from tight packing to (bs, max_vbs) for acceptance computation.
            cumsum = self.per_request_vbs_cumsum
            total_real = self.total_real_tokens

            # Packed logits: [bs * effective_tpbs, vocab_size]
            # Real logits are the first total_real entries (tight packed).
            all_logits = logits_output.next_token_logits
            packed_predict = torch.argmax(all_logits[:total_real], dim=-1)  # [total_real]

            if bs == 1:
                max_vbs = total_real
                candidates_2d = self.draft_token[:total_real].view(1, max_vbs)
                target_predict_2d = packed_predict.view(1, max_vbs)
            else:
                max_vbs = int(self.per_request_vbs.max().item())

                # Vectorized scatter: build row/col indices for unpacking.
                row_idx = torch.repeat_interleave(
                    torch.arange(bs, device=device), self.per_request_vbs
                )
                col_idx = torch.arange(total_real, device=device) - torch.repeat_interleave(
                    cumsum[:bs], self.per_request_vbs
                )

                candidates_2d = torch.zeros(
                    bs, max_vbs, dtype=self.draft_token.dtype, device=device
                )
                target_predict_2d = torch.zeros(
                    bs, max_vbs, dtype=torch.int64, device=device
                )
                candidates_2d[row_idx, col_idx] = self.draft_token[:total_real]
                target_predict_2d[row_idx, col_idx] = packed_predict

            # Greedy acceptance on the unpacked 2D layout.
            accept_len, bonus = compute_dflash_accept_len_and_bonus(
                candidates=candidates_2d,
                target_predict=target_predict_2d,
            )

            # Clamp to per-request VBS.
            max_possible = (self.per_request_vbs - 1).to(accept_len.dtype).to(device)
            accept_len = torch.minimum(accept_len, max_possible)
            row_ids = torch.arange(bs, device=device)
            bonus = target_predict_2d[row_ids, accept_len]

            # Build packed for D2H transfer: candidates[1:max_vbs] + accept_len + bonus
            packed_cpu = torch.cat(
                [candidates_2d[:, 1:], accept_len.unsqueeze(1), bonus.unsqueeze(1)], dim=1
            ).cpu()
            max_acc = max_vbs - 1
        else:
            candidates = self.draft_token.view(bs, self.draft_token_num)
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                bs, self.draft_token_num
            )
            if (
                sampling_info is not None
                and not sampling_info.is_all_greedy
                and is_dflash_sampling_verify_available()
            ):
                accept_len, bonus = compute_dflash_sampling_accept_len_and_bonus(
                    candidates=candidates,
                    next_token_logits=logits_output.next_token_logits,
                    sampling_info=sampling_info,
                )
            else:
                accept_len, bonus = compute_dflash_accept_len_and_bonus(
                    candidates=candidates,
                    target_predict=target_predict,
                )

            if self.per_request_vbs is not None:
                max_possible = (self.per_request_vbs - 1).to(accept_len.dtype).to(device)
                needs_clamp = accept_len > max_possible
                if needs_clamp.any():
                    accept_len = torch.minimum(accept_len, max_possible)
                    row_ids = torch.arange(bs, device=device)
                    new_bonus = target_predict[row_ids, accept_len]
                    bonus = torch.where(needs_clamp, new_bonus, bonus)

            packed_cpu = torch.cat(
                [candidates[:, 1:], accept_len.unsqueeze(1), bonus.unsqueeze(1)], dim=1
            ).cpu()
            max_acc = self.draft_token_num - 1

        accept_length_per_req_cpu: List[int] = []
        commit_lens_cpu: List[int] = []
        new_verified_list: List[int] = []

        for i, req in enumerate(batch.reqs):
            acc_len = int(packed_cpu[i, max_acc].item())
            proposed = packed_cpu[i, :acc_len].tolist() + [
                int(packed_cpu[i, max_acc + 1].item())
            ]

            appended = 0
            for token_id in proposed:
                token_id = int(token_id)
                req.output_ids.append(token_id)
                appended += 1
                req.check_finished()
                if req.finished():
                    break
                if req.grammar is not None:
                    req.grammar.accept_token(token_id)

            if req.output_ids:
                new_verified_token = int(req.output_ids[-1])
            elif req.origin_input_ids:
                new_verified_token = int(req.origin_input_ids[-1])
            else:
                raise RuntimeError(
                    "DFLASH verify cannot determine current token: both output_ids and origin_input_ids are empty."
                )

            commit_lens_cpu.append(appended)
            new_verified_list.append(new_verified_token)
            accept_length_per_req_cpu.append(max(0, appended - 1))
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += accept_length_per_req_cpu[-1]

        commit_lens = torch.tensor(commit_lens_cpu, dtype=torch.int32, device=device)
        new_verified_id = torch.tensor(
            new_verified_list, dtype=torch.int64, device=device
        )

        # Free uncommitted KV cache slots and compact out_cache_loc.
        if tight:
            # With tight packing, out_cache_loc has bs * effective_tpbs entries.
            # The first total_real are assigned to requests (in packed order).
            # The remaining are padding slots that need to be freed.
            total_alloc = batch.out_cache_loc.shape[0]
            total_real = self.total_real_tokens

            # Free padding cache slots (beyond real tokens).
            if total_real < total_alloc:
                batch.token_to_kv_pool_allocator.free(
                    batch.out_cache_loc[total_real:total_alloc]
                )

            # Vectorized per-request KV freeing within real tokens.
            real_cache_loc = batch.out_cache_loc[:total_real]
            if bs == 1:
                commit_len0 = int(commit_lens_cpu[0])
                committed = (
                    torch.arange(total_real, device=device) < commit_len0
                )
                if commit_len0 < total_real:
                    batch.token_to_kv_pool_allocator.free(real_cache_loc[commit_len0:])
                batch.out_cache_loc = real_cache_loc[:commit_len0]
            else:
                # Build offset-within-request for each packed token.
                offset_in_req = torch.arange(total_real, device=device) - torch.repeat_interleave(
                    self.per_request_vbs_cumsum[:bs], self.per_request_vbs
                )
                # Each token is committed if its offset < commit_lens[request_id].
                committed = offset_in_req < torch.repeat_interleave(
                    commit_lens, self.per_request_vbs
                )

                batch.token_to_kv_pool_allocator.free(real_cache_loc[~committed])
                batch.out_cache_loc = real_cache_loc[committed]
        elif page_size == 1:
            out_cache_loc = batch.out_cache_loc.view(bs, self.draft_token_num)
            keep_mask = (
                torch.arange(self.draft_token_num, device=device)[None, :]
                < commit_lens[:, None]
            )
            batch.token_to_kv_pool_allocator.free(out_cache_loc[~keep_mask])
            batch.out_cache_loc = out_cache_loc[keep_mask]
        else:
            out_cache_loc = batch.out_cache_loc.view(bs, self.draft_token_num)
            row_offsets = torch.arange(self.draft_token_num, device=device)[None, :]
            keep_slots = _compute_paged_keep_slots(
                prefix_lens=batch.seq_lens,
                commit_lens=commit_lens,
                draft_token_num=self.draft_token_num,
                page_size=page_size,
            )
            free_mask = row_offsets >= keep_slots[:, None]
            batch.token_to_kv_pool_allocator.free(out_cache_loc[free_mask])

            keep_mask = row_offsets < commit_lens[:, None]
            batch.out_cache_loc = out_cache_loc[keep_mask]

        # Update req-level KV cache accounting.
        for req, commit_len in zip(batch.reqs, commit_lens_cpu, strict=True):
            req.kv_committed_len += commit_len
            req.kv_allocated_len = req.kv_committed_len

        # Update req_to_token pool mapping for newly committed tokens.
        end_offset = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        # Update batch seq lens.
        batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
        batch.seq_lens_cpu.add_(
            torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
        )
        batch.seq_lens_sum += sum(commit_lens_cpu)

        # Build next-step context features from the committed verify-input tokens.
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )

        if tight:
            # Vectorized gather of committed hidden states from tight-packed layout.
            # Reuse the 'committed' mask from KV freeing (same logic).
            hidden_flat = hidden[:total_real]  # [total_real, feature_dim]
            next_target_hidden = hidden_flat[committed]
        else:
            hidden = hidden.view(bs, self.draft_token_num, -1)
            offsets = torch.arange(self.draft_token_num, device=device)[None, :]
            gather_mask = offsets < commit_lens[:, None]
            next_target_hidden = hidden[gather_mask]

        logits_output.hidden_states = None

        return (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
        )
