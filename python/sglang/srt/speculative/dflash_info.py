from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dflash_utils import compute_dflash_accept_len_and_bonus
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func


@dataclass
class DFlashDraftInput(SpecInput):
    """Per-batch DFlash draft state for spec-v1 (non-overlap) scheduling."""

    verified_id: torch.Tensor
    target_hidden: torch.Tensor
    ctx_lens: torch.Tensor
    draft_seq_lens: torch.Tensor

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
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
        elif spec_info.target_hidden is not None and spec_info.target_hidden.numel() > 0:
            self.target_hidden = torch.cat([self.target_hidden, spec_info.target_hidden], dim=0)


@dataclass
class DFlashVerifyInput(SpecInput):
    """Inputs for a target-model verify forward in DFlash (spec-v1)."""

    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    verify_token_lens: torch.Tensor
    verify_start_offsets_cpu: Optional[List[int]] = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL
    num_tokens_per_batch: int = -1

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)
        if self.num_tokens_per_batch == -1:
            self.num_tokens_per_batch = int(self.draft_token_num) if int(self.draft_token_num) > 0 else 1

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
        if batch.forward_mode.is_idle():
            return
        if page_size != 1:
            raise NotImplementedError("DFLASH verify currently supports page_size==1 only.")

        bs = batch.batch_size()
        if self.verify_token_lens.numel() != bs:
            raise RuntimeError(
                f"DFLASH verify_token_lens shape mismatch: got {self.verify_token_lens.numel()} for bs={bs}."
            )

        batch.input_ids = self.draft_token
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

        # Ensure verify_token_lens is at least [bs] to avoid shape mismatch in copy_
        # If it's shorter or empty, it means we are likely in a fixed-block mode (like draft forward)
        # where we should use draft_token_num for all requests.
        vlens = self.verify_token_lens
        if vlens.numel() != bs:
            vlens = torch.full((bs,), int(self.draft_token_num), dtype=torch.int32, device=device)

        qo_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        qo_indptr[1:].copy_(torch.cumsum(vlens.to(device), dim=0))

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        # paged_kernel_lens is prefix_lens, we need to add vlens for total KV length
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

    def verify(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        if batch.forward_mode.is_idle():
            empty = torch.empty((0,), dtype=torch.int64, device=batch.device)
            return empty, empty.to(torch.int32), empty, []

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device
        vlen_list = self.verify_token_lens.tolist()
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
            
            # current_candidates: [vlen], includes verified_id at index 0.
            # positions: [vlen], verified_id is at prefix_lens[i].
            current_candidates = self.draft_token[offset : offset + vlen]
            # current_logits[t] is prediction for current_candidates[t+1].
            current_logits = logits_flat[offset : offset + vlen]
            current_predict = torch.argmax(current_logits, dim=-1)

            # # Debug logs for Shift-1 alignment analysis
            # if i == 0:
            #     print(f"DEBUG: [Req 0] vlen={vlen}, offset={offset}")
            #     print(f"DEBUG: [Req 0] current_candidates[:5]={current_candidates[:5].tolist()}")
            #     print(f"DEBUG: [Req 0] current_predict[:5]={current_predict[:5].tolist()}")
            #     print(f"DEBUG: [Req 0] logits_flat.shape={logits_flat.shape}")
            
            # The rule in compute_dflash_accept_len_and_bonus is:
            #   accept while candidates[:, 1:] == target_predict[:, :-1]
            # In our ragged case, target_predict[t] is prediction for candidates[t+1].
            # So target_predict[:-1] are predictions for candidates[1:].
            # This matches exactly.
            acc_len_t, bonus_t = compute_dflash_accept_len_and_bonus(
                candidates=current_candidates.unsqueeze(0),
                target_predict=current_predict.unsqueeze(0),
            )
            
            acc_len = int(acc_len_t.item())
            bonus = int(bonus_t.item())
            
            # The tokens to append: accepted draft tokens (indices 1 to acc_len) + 1 bonus.
            proposed: List[int] = []
            if acc_len > 0:
                proposed.extend(current_candidates[1 : acc_len + 1].tolist())
            proposed.append(bonus)

            appended = 0
            if req.grammar is None and not req.sampling_params.stop_strs and not req.sampling_params.stop_regex_strs:
                remaining = int(req.sampling_params.max_new_tokens) - len(req.output_ids)
                if remaining > 0:
                    tokens = proposed[:remaining]
                    if not req.sampling_params.ignore_eos:
                        stop_token_ids = req.sampling_params.stop_token_ids
                        eos_token_ids = req.eos_token_ids
                        tokenizer = req.tokenizer
                        tokenizer_eos = tokenizer.eos_token_id if tokenizer is not None else None
                        additional_stop = tokenizer.additional_stop_token_ids if tokenizer is not None else None
                        vocab_size = getattr(req, "vocab_size", None)
                        for j, token_id in enumerate(tokens):
                            if vocab_size is not None and (int(token_id) > int(vocab_size) or int(token_id) < 0):
                                tokens = tokens[: j + 1]; break
                            if stop_token_ids and token_id in stop_token_ids:
                                tokens = tokens[: j + 1]; break
                            if eos_token_ids and token_id in eos_token_ids:
                                tokens = tokens[: j + 1]; break
                            if tokenizer_eos is not None and int(token_id) == int(tokenizer_eos):
                                tokens = tokens[: j + 1]; break
                            if additional_stop and token_id in additional_stop:
                                tokens = tokens[: j + 1]; break
                    req.output_ids.extend(int(tok) for tok in tokens)
                    appended = len(tokens)
                    if appended > 0: req.check_finished(new_accepted_len=appended)
            else:
                for tok in proposed:
                    req.output_ids.append(int(tok))
                    appended += 1
                    req.check_finished()
                    if req.finished(): break
                    if req.grammar is not None: req.grammar.accept_token(int(tok))

            if appended <= 0: raise RuntimeError("DFLASH verify appended 0 tokens.")
            commit_lens_cpu.append(appended)
            new_verified_cpu.append(req.output_ids[-1])
            
            # accept_length is the number of draft tokens accepted (proposed minus bonus).
            acc_true = max(0, appended - 1)
            accept_length_per_req_cpu.append(acc_true)
            
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += acc_true
            req.spec_verify_tokens += vlen
            
            if hidden_flat is not None:
                segments_hidden.append(hidden_flat[offset : offset + appended])
            
            # Update K-online stats if available.
            if hasattr(self, "_k_online_token_nll") and self._k_online_token_nll is not None:
                if req.k_online_sum_by_acc is not None:
                    # _k_online_token_nll was saved during draft as [bs, block_size-1].
                    token_nll_i = self._k_online_token_nll[i]
                    num_pos = int(token_nll_i.shape[0])
                    acc_idx = min(int(acc_true), num_pos)
                    req.k_online_sum_by_acc[acc_idx, :] += token_nll_i
                    req.k_online_count_by_acc[acc_idx] += 1
                    req.k_online_step += 1

        commit_lens = torch.tensor(commit_lens_cpu, dtype=torch.int32, device=device)
        self._k_online_token_nll = None
        
        if page_size == 1:
            out_cache_loc = batch.out_cache_loc
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

        for req, commit_len in zip(batch.reqs, commit_lens_cpu):
            req.kv_committed_len += commit_len
            req.kv_allocated_len = req.kv_committed_len
            
        end_offset = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)
        assign_req_to_token_pool_func(batch.req_pool_indices, batch.req_to_token_pool.req_to_token, batch.seq_lens, end_offset, batch.out_cache_loc, bs)
        
        batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
        batch.seq_lens_cpu.add_(torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype))
        batch.seq_lens_sum += sum(commit_lens_cpu)
        
        next_target_hidden = torch.cat(segments_hidden, dim=0) if segments_hidden else (hidden_flat[:0] if hidden_flat is not None else logits_flat[:0])
        logits_output.hidden_states = None
        new_verified_id = torch.tensor(new_verified_cpu, dtype=torch.int64, device=device)
        return new_verified_id, commit_lens, next_target_hidden, accept_length_per_req_cpu

