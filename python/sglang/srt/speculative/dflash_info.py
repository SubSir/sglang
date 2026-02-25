from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import os
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dflash_utils import compute_dflash_accept_len_and_bonus
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func


@dataclass
class DFlashDraftInput(SpecInput):
    """Per-batch DFlash draft state for spec-v1 (non-overlap) scheduling.

    This object is stored on `ScheduleBatch.spec_info` between decode iterations.
    It is NOT sent to model attention backends; the DFlash worker uses it to run
    the draft model and to track draft-side cache progress.

    Invariant (per request):
      - `draft_seq_len + ctx_len == batch.seq_lens[i]`
        where `ctx_len` is the number of target context-feature tokens carried in
        `target_hidden` for that request.
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

    # How many tokens are already in the draft KV cache per request.
    # The next draft step appends ctx_lens[i] tokens starting at draft_seq_lens[i].
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

    支持两种 verify 形态：
    - block verify（固定 block_size）：draft_token 是 [bs * block]
    - ragged verify（每个 req 不同 verify_len）：draft_token 是 flatten 后的 [sum(verify_lens)]
    """

    draft_token: torch.Tensor
    draft_token_num: int

    # 兼容旧路径（block verify）
    positions: Optional[torch.Tensor] = None

    # ragged verify：每个 req 实际 verify 的 token 数（包含 verified_id 在内）
    verify_token_lens: Optional[torch.Tensor] = None
    # ragged verify：CPU 上的 per-req 起始 offset（len == bs）
    verify_start_offsets_cpu: Optional[List[int]] = None

    # ragged verify: the extend_start_loc calculated by ForwardBatchInfo, used to slice logits
    verify_extend_start_loc: Optional[torch.Tensor] = None

    # Kept for compatibility with attention backends that gate tree metadata by `topk > 1`.
    # DFLASH verify is linear (non-tree), so this is always 1.
    topk: int = 1

    custom_mask: Optional[torch.Tensor] = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Shape info for padding (e.g., DP attention / CUDA graph).
    # 对 block verify：= draft_token_num；对 ragged verify：可保持为 draft_token_num（DP padding 需要）
    num_tokens_per_batch: int = -1

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)
        if self.verify_token_lens is None:
            self.verify_token_lens = torch.empty((0,), dtype=torch.int32)
        if self.num_tokens_per_batch == -1:
            self.num_tokens_per_batch = int(self.draft_token_num) if int(self.draft_token_num) > 0 else 1

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def _is_ragged_verify(self) -> bool:
        """判断是否为 ragged verify（verify_token_lens 非空且长度 == bs）。"""
        return self.verify_token_lens is not None and self.verify_token_lens.numel() > 0

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        page_size: int,
        *,
        build_custom_mask: bool = True,
    ):
        if batch.forward_mode.is_idle():
            return

        # ragged verify 目前只支持 page_size == 1
        if self._is_ragged_verify() and page_size != 1:
            raise NotImplementedError("DFLASH ragged verify currently supports page_size==1 only.")

        bs = batch.batch_size()
        batch.input_ids = self.draft_token

        # ragged verify：不覆盖 batch.positions，让 ForwardBatchInfo 用 extend_* 字段计算
        # if self.positions is not None:
        batch.positions = self.positions

        # Defensive checks: if scheduler enters DFLASH_VERIFY mode, ragged metadata must be present.
        if batch.forward_mode.name == "DFLASH_VERIFY":
            if not self._is_ragged_verify():
                raise RuntimeError(
                    "DFLASH_VERIFY forward requires ragged metadata, but verify_token_lens is empty. "
                    "This can lead to invalid attention indices and GPU faults."
                )
            if self.verify_start_offsets_cpu is None or len(self.verify_start_offsets_cpu) != bs:
                raise RuntimeError(
                    "DFLASH_VERIFY forward requires verify_start_offsets_cpu with length == batch_size."
                )

        if self._is_ragged_verify():
            # --- ragged verify path
            if self.verify_token_lens.numel() != bs:
                raise RuntimeError(
                    f"DFLASH verify_token_lens shape mismatch: got {self.verify_token_lens.numel()} for bs={bs}."
                )
            batch.out_cache_loc = alloc_token_slots(batch.tree_cache, len(batch.input_ids))
            prefix_lens = batch.seq_lens
            end_offset = prefix_lens + self.verify_token_lens.to(prefix_lens.device, dtype=prefix_lens.dtype)
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                prefix_lens,
                end_offset,
                batch.out_cache_loc,
                bs,
            )
            # ragged verify 不构建 custom_mask（flashinfer/flashattention 后端不需要）
            self.custom_mask = None
            return

        # --- block verify path（原有逻辑）
        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache, len(batch.input_ids)
            )
            end_offset = batch.seq_lens + self.draft_token_num
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.draft_token_num
            end_offset_cpu = prefix_lens_cpu + self.draft_token_num
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
        mask_chunks: List[torch.Tensor] = []
        q_len = int(self.draft_token_num)
        q_idx = torch.arange(q_len, device=batch.device, dtype=torch.int32).unsqueeze(1)
        for prefix_len in batch.seq_lens_cpu.tolist():
            prefix_len_i = int(prefix_len)
            kv_len = prefix_len_i + q_len
            k_idx = torch.arange(
                kv_len, device=batch.device, dtype=torch.int32
            ).unsqueeze(0)
            # Allow attending to the full prefix and to tokens up to (and including) the
            # current query position within the verify block (standard causal masking).
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

        if bs == 0:
            return (
                torch.empty((0,), dtype=torch.int32, device=device),
                torch.zeros((1,), dtype=torch.int32, device=device),
                torch.zeros((1,), dtype=torch.int32, device=device),
                None,
            )

        # ragged verify：qo_indptr / kv_indices 按 verify_token_lens 构造
        if self._is_ragged_verify():
            vlens = self.verify_token_lens
            if vlens is None or vlens.numel() != bs:
                vlens = torch.full(
                    (bs,), int(self.draft_token_num), dtype=torch.int32, device=device
                )

            qo_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
            qo_indptr[1:].copy_(torch.cumsum(vlens.to(device), dim=0))

            cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
            total_kv_lens = paged_kernel_lens + vlens.to(device)
            cum_kv_seq_len[1:] = torch.cumsum(total_kv_lens, dim=0)

            kv_indices = torch.empty(
                int(total_kv_lens.sum().item()),
                dtype=torch.int32,
                device=device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                req_to_token,
                req_pool_indices,
                total_kv_lens,
                cum_kv_seq_len,
                None,
                kv_indices,
                req_to_token.size(1),
            )
            return kv_indices, cum_kv_seq_len, qo_indptr, None

        # --- block verify（原逻辑）
        qo_indptr = torch.arange(
            0,
            (bs + 1) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * bs,
            dtype=torch.int32,
            device=device,
        )
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
            mask_numel = (
                paged_kernel_lens_sum * self.draft_token_num
                + (self.draft_token_num**2) * bs
            )
            if mask.numel() < mask_numel:
                # FIXME(attn): temporary fix for custom mask padding with cuda graph
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
        """Greedy DFlash verification.

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

        # --- ragged verify path
        if self._is_ragged_verify():
            vlen_list = self.verify_token_lens.tolist() if self.verify_token_lens is not None else []
            start_offsets = self.verify_start_offsets_cpu
            if start_offsets is None or len(start_offsets) != bs:
                raise RuntimeError("DFLASH ragged verify requires verify_start_offsets_cpu.")

            logits_flat = logits_output.next_token_logits
            hidden_flat = logits_output.hidden_states

            accept_length_per_req_cpu: List[int] = []
            commit_lens_cpu: List[int] = []
            new_verified_cpu: List[int] = []
            segments_hidden: List[torch.Tensor] = []

            for i, req in enumerate(batch.reqs):
                vlen = int(vlen_list[i])
                offset = int(start_offsets[i])

                current_candidates = self.draft_token[offset : offset + vlen]

                # Use extend_start_loc to correctly slice logits for each request.
                # extend_start_loc[i] is the start index in the flattened logits tensor for request i.
                # Each token position t in the request corresponds to logits at index extend_start_loc[i] + t,
                # which predicts the token at position t+1.
                # if self.verify_extend_start_loc is not None:
                #     logit_off = int(self.verify_extend_start_loc[i].item())
                # else:
                    # Fallback to draft token offset if extend_start_loc is not available (should not happen in normal flow)
                logit_off = offset

                # A-scheme: mirror original DFlash spec.
                # logits_flat[logit_off + j] is the prediction made after seeing candidates[j].
                # We compare candidates[1:] with current_predict[:-1].
                logits_len = vlen
                current_logits = logits_flat[logit_off : logit_off + logits_len]

                if os.environ.get("SGLANG_DFLASH_NAN_GUARD", "1") == "1":
                    if torch.isnan(current_logits).any().item() or torch.isinf(current_logits).any().item():
                        # Dump more metadata for target logits NaN
                        sl_cpu = batch.seq_lens_cpu.tolist() if batch.seq_lens_cpu is not None else []
                        sl_dev = batch.seq_lens.tolist()
                        vlens = self.verify_token_lens.tolist() if self.verify_token_lens is not None else []
                        oc_min = int(batch.out_cache_loc.min().item()) if batch.out_cache_loc.numel() > 0 else -1
                        oc_max = int(batch.out_cache_loc.max().item()) if batch.out_cache_loc.numel() > 0 else -1
                        
                        # Add positions info if available
                        pos_info = ""
                        if self.positions is not None:
                            p_slice = self.positions[offset : offset + vlen]
                            pos_info = f"explicit_pos_slice=[{int(p_slice.min().item())},{int(p_slice.max().item())}], pos_head={p_slice[:min(8, vlen)].cpu().tolist()}"
                        else:
                            pos_info = "explicit_pos=None (recomputed by FBInfo)"

                        raise RuntimeError(
                            "DFLASH_NAN_GUARD: NaN/Inf in target logits during ragged verify. "
                            f"req={i}, vlen={vlen}, logit_off={logit_off}, logits_len={logits_len}, "
                            f"batch_seq_lens_cpu={sl_cpu}, batch_seq_lens_dev={sl_dev}, "
                            f"verify_token_lens={vlens}, out_cache_loc_range=[{oc_min},{oc_max}], "
                            f"{pos_info}, candidates_head={current_candidates[:min(8, vlen)].cpu().tolist()}"
                        )

                current_predict = torch.argmax(current_logits, dim=-1)  # shape [vlen]

                if os.environ.get("SGLANG_DFLASH_DEBUG", "1") == "1":
                    print(f"[DFLASH DEBUG] req={i}, vlen={vlen}, logit_off={logit_off}, logits_len={logits_len}")
                    print(f"[DFLASH DEBUG]   candidates={current_candidates.cpu().tolist()}")
                    print(f"[DFLASH DEBUG]   current_predict={current_predict.cpu().tolist()}")

                # Ensure target_predict has the same shape as candidates.
                # `current_predict` is already length vlen, so no extra padding is needed.
                target_predict = current_predict

                acc_len_t, bonus_t = compute_dflash_accept_len_and_bonus(
                    candidates=current_candidates.unsqueeze(0),
                    target_predict=target_predict.unsqueeze(0),
                )
                acc_len = int(acc_len_t.item())
                bonus = int(bonus_t.item())

                proposed: List[int] = []
                if acc_len > 0:
                    proposed.extend(current_candidates[1 : acc_len + 1].tolist())
                proposed.append(bonus)

                appended = 0
                if (
                    req.grammar is None
                    and not req.sampling_params.stop_strs
                    and not req.sampling_params.stop_regex_strs
                ):
                    remaining = int(req.sampling_params.max_new_tokens) - len(req.output_ids)
                    if remaining > 0:
                        tokens = proposed[:remaining]
                        if not req.sampling_params.ignore_eos:
                            stop_token_ids = req.sampling_params.stop_token_ids
                            eos_token_ids = req.eos_token_ids
                            tokenizer = req.tokenizer
                            tokenizer_eos = tokenizer.eos_token_id if tokenizer is not None else None
                            additional_stop = (
                                tokenizer.additional_stop_token_ids if tokenizer is not None else None
                            )
                            vocab_size = getattr(req, "vocab_size", None)
                            for j, token_id in enumerate(tokens):
                                if vocab_size is not None and (
                                    int(token_id) > int(vocab_size) or int(token_id) < 0
                                ):
                                    tokens = tokens[: j + 1]
                                    break
                                if stop_token_ids and token_id in stop_token_ids:
                                    tokens = tokens[: j + 1]
                                    break
                                if eos_token_ids and token_id in eos_token_ids:
                                    tokens = tokens[: j + 1]
                                    break
                                if tokenizer_eos is not None and int(token_id) == int(tokenizer_eos):
                                    tokens = tokens[: j + 1]
                                    break
                                if additional_stop and token_id in additional_stop:
                                    tokens = tokens[: j + 1]
                                    break

                        req.output_ids.extend(int(tok) for tok in tokens)
                        appended = len(tokens)
                        if appended > 0:
                            req.check_finished(new_accepted_len=appended)
                else:
                    for tok in proposed:
                        req.output_ids.append(int(tok))
                        appended += 1
                        req.check_finished()
                        if req.finished():
                            break
                        if req.grammar is not None:
                            req.grammar.accept_token(int(tok))

                if appended <= 0:
                    raise RuntimeError("DFLASH verify appended 0 tokens.")

                commit_lens_cpu.append(appended)
                new_verified_cpu.append(int(req.output_ids[-1]))

                acc_true = max(0, appended - 1)
                accept_length_per_req_cpu.append(acc_true)

                # Some Req implementations may not initialize speculative metrics; be defensive.
                if not hasattr(req, "spec_verify_ct"):
                    req.spec_verify_ct = 0
                if not hasattr(req, "spec_accepted_tokens"):
                    req.spec_accepted_tokens = 0
                if not hasattr(req, "spec_verify_tokens"):
                    req.spec_verify_tokens = 0

                req.spec_verify_ct += 1
                req.spec_accepted_tokens += acc_true
                req.spec_verify_tokens += vlen

                if hidden_flat is not None:
                    segments_hidden.append(hidden_flat[offset : offset + appended])

                # Update k-online stats if present
                if hasattr(self, "_k_online_token_nll") and getattr(self, "_k_online_token_nll") is not None:
                    if getattr(req, "k_online_sum_by_acc", None) is not None:
                        token_nll_i = self._k_online_token_nll[i]
                        num_pos = int(token_nll_i.shape[0])
                        acc_idx = min(int(acc_true), num_pos)
                        req.k_online_sum_by_acc[acc_idx, :] += token_nll_i
                        req.k_online_count_by_acc[acc_idx] += 1
                        req.k_online_step += 1

            commit_lens = torch.tensor(commit_lens_cpu, dtype=torch.int32, device=device)
            # clear cache
            if hasattr(self, "_k_online_token_nll"):
                self._k_online_token_nll = None

            if page_size == 1:
                out_cache_loc = batch.out_cache_loc
                # Consistency check
                if os.environ.get("SGLANG_DFLASH_NAN_GUARD", "1") == "1":
                    total_vlen_expected = sum(vlen_list)
                    if out_cache_loc.numel() != total_vlen_expected:
                        raise RuntimeError(
                            f"DFLASH KV Consistency Error: batch.out_cache_loc.numel() ({out_cache_loc.numel()}) "
                            f"!= sum(vlen_list) ({total_vlen_expected}). vlen_list={vlen_list}"
                        )

                to_free_chunks, kept_chunks = [], []
                curr = 0
                for i, vlen in enumerate(vlen_list):
                    committed = commit_lens_cpu[i]
                    kept_chunks.append(out_cache_loc[curr : curr + committed])
                    if vlen > committed:
                        to_free_chunks.append(out_cache_loc[curr + committed : curr + vlen])
                    curr += vlen
                
                if to_free_chunks:
                    batch.token_to_kv_pool_allocator.free(torch.cat(to_free_chunks))
                batch.out_cache_loc = torch.cat(kept_chunks) if kept_chunks else out_cache_loc[:0]
            else:
                raise NotImplementedError("DFLASH ragged verify page_size > 1 not supported.")

            for req, commit_len in zip(batch.reqs, commit_lens_cpu, strict=True):
                req.kv_committed_len += commit_len
                req.kv_allocated_len = req.kv_committed_len

            end_offset = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                batch.seq_lens,
                end_offset,
                batch.out_cache_loc,
                bs,
            )

            batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
            batch.seq_lens_cpu.add_(
                torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
            )
            batch.seq_lens_sum += sum(commit_lens_cpu)

            next_target_hidden = (
                torch.cat(segments_hidden, dim=0)
                if segments_hidden
                else (
                    hidden_flat[:0]
                    if hidden_flat is not None
                    else logits_flat[:0]
                )
            )
            logits_output.hidden_states = None
            new_verified_id = torch.tensor(new_verified_cpu, dtype=torch.int64, device=device)
            return new_verified_id, commit_lens, next_target_hidden, accept_length_per_req_cpu

        # --- block verify path
        candidates = self.draft_token.view(bs, self.draft_token_num)
        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
            bs, self.draft_token_num
        )
        accept_len, bonus = compute_dflash_accept_len_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )

        # Single D2H transfer: candidates[1:] + accept_len + bonus
        packed = torch.cat(
            [candidates[:, 1:], accept_len.unsqueeze(1), bonus.unsqueeze(1)], dim=1
        ).cpu()

        max_acc = self.draft_token_num - 1
        accept_length_per_req_cpu: List[int] = []
        commit_lens_cpu: List[int] = []
        new_verified_list: List[int] = []

        for i, req in enumerate(batch.reqs):
            acc_len = int(packed[i, max_acc].item())
            proposed = packed[i, :acc_len].tolist() + [
                int(packed[i, max_acc + 1].item())
            ]

            appended = 0
            if (
                req.grammar is None
                and not req.sampling_params.stop_strs
                and not req.sampling_params.stop_regex_strs
            ):
                remaining = int(req.sampling_params.max_new_tokens) - len(
                    req.output_ids
                )
                if remaining > 0:
                    tokens = proposed[:remaining]
                    if not req.sampling_params.ignore_eos:
                        stop_token_ids = req.sampling_params.stop_token_ids
                        eos_token_ids = req.eos_token_ids
                        tokenizer = req.tokenizer
                        tokenizer_eos = (
                            tokenizer.eos_token_id if tokenizer is not None else None
                        )
                        additional_stop = (
                            tokenizer.additional_stop_token_ids
                            if tokenizer is not None
                            else None
                        )
                        vocab_size = getattr(req, "vocab_size", None)

                        for j, token_id in enumerate(tokens):
                            if vocab_size is not None and (
                                int(token_id) > int(vocab_size) or int(token_id) < 0
                            ):
                                tokens = tokens[: j + 1]
                                break
                            if stop_token_ids and token_id in stop_token_ids:
                                tokens = tokens[: j + 1]
                                break
                            if eos_token_ids and token_id in eos_token_ids:
                                tokens = tokens[: j + 1]
                                break
                            if tokenizer_eos is not None and int(token_id) == int(
                                tokenizer_eos
                            ):
                                tokens = tokens[: j + 1]
                                break
                            if additional_stop and token_id in additional_stop:
                                tokens = tokens[: j + 1]
                                break

                    req.output_ids.extend(int(tok) for tok in tokens)
                    appended = len(tokens)
                    if appended > 0:
                        req.check_finished(new_accepted_len=appended)
            else:
                for tok in proposed:
                    req.output_ids.append(int(tok))
                    appended += 1
                    req.check_finished()
                    if req.finished():
                        break
                    if req.grammar is not None:
                        req.grammar.accept_token(int(tok))

            if req.output_ids:
                new_verified_token = int(req.output_ids[-1])
            elif req.origin_input_ids:
                # If no token was appended in this verify step, keep the current token unchanged.
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
        if page_size == 1:
            out_cache_loc = batch.out_cache_loc.view(bs, self.draft_token_num)
            keep_mask = (
                torch.arange(self.draft_token_num, device=device)[None, :]
                < commit_lens[:, None]
            )
            batch.token_to_kv_pool_allocator.free(out_cache_loc[~keep_mask])
            batch.out_cache_loc = out_cache_loc[keep_mask]
        else:
            # Page-size > 1 is not supported in the initial DFlash implementation.
            raise NotImplementedError(
                "DFLASH verify with page_size > 1 is not supported yet."
            )

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
        # Keep seq_lens_sum in sync; flashinfer indices updaters rely on this for buffer sizing.
        batch.seq_lens_sum += sum(commit_lens_cpu)

        # Build next-step context features from the committed verify-input tokens.
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        hidden = hidden.view(bs, self.draft_token_num, -1)
        segments: List[torch.Tensor] = []
        for i, ln in enumerate(commit_lens_cpu):
            if ln > 0:
                segments.append(hidden[i, :ln, :])
        next_target_hidden = torch.cat(segments, dim=0) if segments else hidden[:0]

        # Avoid confusing downstream consumers (spec-v1 decode doesn't use this).
        logits_output.hidden_states = None

        return (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
        )
