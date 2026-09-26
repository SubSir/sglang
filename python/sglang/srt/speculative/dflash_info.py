from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

_is_npu = is_npu()


@dataclass
class DFlashVerifyInput(SpecInput):
    """Inputs for a target-model verify forward in DFlash.

    The verify forward is run with `ForwardMode.TARGET_VERIFY` so that the target
    model returns logits for all tokens in the block, enabling accept-length
    computation.
    """

    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    # Kept for compatibility with attention backends that gate tree metadata by `topk > 1`.
    # 1 for the chain verify; the tree branching width under --speculative-dflash-tree-topk.
    topk: int = 1
    # Custom attention "allow mask" for TARGET_VERIFY in backends that require it.
    # Semantics follow SGLang speculative conventions: True means the (q, k) pair is allowed.
    custom_mask: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Shape info for padding (e.g., DP attention / CUDA graph).
    num_tokens_per_req: int = -1

    ragged_verify_layout: Optional[RaggedVerifyLayout] = None

    # Tree verify (EAGLE layout). Set with positions=None; prepare_for_verify then
    # builds the ancestor mask, the per-node positions and the retrieve pointers.
    tree_parent_list: Optional[torch.Tensor] = None
    tree_selected_index: Optional[torch.Tensor] = None
    tree_depth: int = 0
    tree_mask_buf: Optional[torch.Tensor] = None
    # Sliding-window layers see min(prefix, window) + tree keys per request, so
    # their mask rows are shorter than the full-attention ones.
    tree_swa_window: Optional[int] = None
    tree_mask_buf_swa: Optional[torch.Tensor] = None
    custom_mask_swa: Optional[torch.Tensor] = None
    retrieve_index: Optional[torch.Tensor] = None
    retrieve_next_token: Optional[torch.Tensor] = None
    retrieve_next_sibling: Optional[torch.Tensor] = None
    # Committed/live lengths before the verify caller temporarily expands
    # batch.seq_lens_cpu to the target-attention KV lengths.
    live_seq_lens_cpu: Optional[torch.Tensor] = None
    # Conservative request-lifetime bound for candidate graph dispatch.
    candidate_max_seq_len_upper_bound: Optional[int] = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)
        if self.num_tokens_per_req == -1:
            self.num_tokens_per_req = int(self.draft_token_num)
        self.num_tokens_for_logprob_per_req = int(self.draft_token_num)

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        target_worker: TpModelWorker,
    ) -> tuple[ForwardBatch, bool]:
        """Prepare a DFLASH verify forward batch for overlap scheduling.

        The caller computes and stores `batch.out_cache_loc` before this
        method is called. GPU keeps the original pre-planning path. NPU leaves
        attention/graph metadata initialization to ModelRunner because DP/EP
        padding can still change the compressor's runtime shapes.
        """
        from sglang.srt.speculative.spec_utils import prepare_mamba_track_for_verify

        if self.tree_parent_list is not None and not batch.forward_mode.is_idle():
            self._build_tree(batch)
        batch.input_ids = self.draft_token
        batch.spec_info = self
        if _is_npu and not batch.forward_mode.is_idle():
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                maybe_build_dsv4_verify_bundle,
            )

            batch.out_cache_loc_dsv4 = maybe_build_dsv4_verify_bundle(
                batch,
                self.draft_token_num,
                live_seq_lens_cpu=self.live_seq_lens_cpu,
            )

        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        if not batch.forward_mode.is_idle():
            # Rebuild mamba track indices (lazy: gather the positions planned
            # by mamba_lazy_spec_prepare) and clear the stale extend-time mask
            # before init_new snapshots them into the verify ForwardBatch.
            # Same hook eagle/ngram/dspark run before TARGET_VERIFY.
            prepare_mamba_track_for_verify(batch)
        verify_forward_batch = ForwardBatch.init_new(
            batch,
            target_worker.model_runner,
            capture_hidden_mode=self.capture_hidden_mode,
            return_hidden_states_before_norm=False,
        )

        can_run_cuda_graph = bool(
            target_worker.model_runner.decode_cuda_graph_runner
            and target_worker.model_runner.decode_cuda_graph_runner.can_run_graph(
                verify_forward_batch
            )
        )
        if _is_npu:
            # Do not pre-plan target verify on NPU. DP/EP padding can change
            # the compressor's logical batch without changing ForwardBatch's
            # stale-plan shape fields. Let ModelRunner select graph/eager and
            # initialize metadata after final batch preparation.
            return verify_forward_batch, can_run_cuda_graph
        elif can_run_cuda_graph:
            target_worker.model_runner.decode_cuda_graph_runner.load_batch(
                verify_forward_batch
            )
        elif not batch.forward_mode.is_idle():
            target_worker.model_runner.attn_backend.init_forward_metadata(
                verify_forward_batch
            )

        return verify_forward_batch, can_run_cuda_graph

    def _build_tree(self, batch: ScheduleBatch) -> None:
        from sglang.srt.speculative.eagle_utils import (
            TreeMaskMode,
            build_tree_kernel_efficient,
        )

        n = int(self.draft_token_num)
        tokens = self.draft_token.view(-1, n)
        (
            self.custom_mask,
            self.positions,
            self.retrieve_index,
            self.retrieve_next_token,
            self.retrieve_next_sibling,
            self.draft_token,
        ) = build_tree_kernel_efficient(
            bonus_tokens=tokens[:, 0],
            parent_list=self.tree_parent_list,
            top_scores_index=self.tree_selected_index,
            draft_tokens=tokens[:, 1:],
            seq_lens=batch.seq_lens,
            # Unused with a preallocated buffer; the kernel writes the used front.
            seq_lens_sum=0,
            topk=int(self.topk),
            spec_steps=int(self.tree_depth),
            num_verify_tokens=n,
            tree_mask_mode=TreeMaskMode.FULL_MASK,
            tree_mask_buf=self.tree_mask_buf,
        )
        if self.tree_swa_window is not None:
            self.custom_mask_swa = build_tree_kernel_efficient(
                bonus_tokens=tokens[:, 0],
                parent_list=self.tree_parent_list,
                top_scores_index=self.tree_selected_index,
                draft_tokens=tokens[:, 1:],
                seq_lens=torch.clamp(batch.seq_lens, max=self.tree_swa_window),
                seq_lens_sum=0,
                topk=int(self.topk),
                spec_steps=int(self.tree_depth),
                num_verify_tokens=n,
                tree_mask_mode=TreeMaskMode.FULL_MASK,
                tree_mask_buf=self.tree_mask_buf_swa,
            )[0]

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
        kv_start_idx: Optional[torch.Tensor] = None,
        kv_indices_buf: Optional[torch.Tensor] = None,
        sliding_window: bool = False,
    ):
        device = req_pool_indices.device
        bs = len(req_pool_indices)

        layout = self.ragged_verify_layout
        if layout is not None and layout.bs != bs:
            # Graph replay pads the batch to the captured slots; match it.
            layout = layout.padded_to_bucket(padded_bs=bs)

        if layout is None:
            qo_indptr = torch.arange(
                0,
                (bs + 1) * self.draft_token_num,
                step=self.draft_token_num,
                dtype=torch.int32,
                device=device,
            )
            verify_lens = self.draft_token_num
            kv_indices_extra = self.draft_token_num * bs
        else:
            qo_indptr = layout.qo_indptr_device
            verify_lens = layout.verify_lens
            kv_indices_extra = layout.total_verify_tokens

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + verify_lens
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        if kv_indices_buf is not None:
            # Sync-free fast-plan path: write straight into the attention
            # backend's cuda-graph kv_indices buffer (the captured kernels read
            # it), skipping both the fresh allocation and the wrapper plan()'s
            # device-to-device refresh copy.
            kv_indices = kv_indices_buf
        else:
            kv_indices = torch.empty(
                paged_kernel_lens_sum + kv_indices_extra,
                dtype=torch.int32,
                device=device,
            )
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            kv_start_idx,
            kv_indices,
            req_to_token.size(1),
        )
        mask = self.custom_mask
        if sliding_window and self.custom_mask_swa is not None:
            mask = self.custom_mask_swa
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
            elif mask.numel() > mask_numel:
                # The tree-mask buffer is sized for the longest context; hand the
                # backend only the used front so its mask packing stays small.
                mask = mask[:mask_numel]
        return kv_indices, cum_kv_seq_len, qo_indptr, mask
