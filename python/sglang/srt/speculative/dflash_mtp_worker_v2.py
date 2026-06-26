# DFLASH + native Qwen3.5-MTP multi-round refiner (own-KV) speculative worker (spec-v2).
#
# Subclass of DFlashWorkerV2 (so is_dflash() stays True everywhere). Selected by create_worker
# when SGLANG_DFLASH_MTP_REFINE_ROUNDS>0 and --speculative-algorithm DFLASH.
#
# Architecture (worker self-builds the MTP forwards; the MTP sits right after the DFlash stage):
#   - The MTP refiner is a directly-held module (qwen3_5_mtp). The worker maintains the MTP's
#     prefix-KV (nextn-states K/V) itself, using ForwardBatches it constructs (mirroring the DFlash
#     draft forward), so it fully controls positions / cache_loc / attention metadata.
#   - The committed prefix-KV is ALWAYS built from the TARGET final hidden (matches HF `_dense_mtp2`
#     training, which uses target_final_hidden -- not the draft hidden):
#       * SEED (prefill): one MTP extend over the prompt -> seed the prefix-KV from prompt hidden.
#       * SUPPLEMENT (each decode, post-verify): MTP extend over just the accept_len newly-committed
#         tokens (target hidden from the verify norm hook) -> append to the prefix-KV. Small/cheap.
#   - REFINE (each decode, pre-verify): MTP forward over the block (queries the committed prefix-KV
#     + scratch self), block-state[j] = fc(cat(norm_e(embed(T[j])), norm_h(H[j]))); decode -> refine
#     T; multi-round Jacobi (K rounds). The block's own K/V are scratch (cropped), never committed.
#   - Two cuda-graph types: (1) DFlash+lm_head+MTP+lm_head, (2) MTP+lm_head (extra rounds).
#
# The MTP is OWN-KV (matches what our mtp2 DFlash backbone was trained against), weights ship inside
# the Qwen3.5-4B checkpoint (draft-arch rewrite -> Qwen3_5ForCausalLMMTP).
#
# BUILD STATUS: raw MTP module + final-hidden capture working (phases 1/2a proven). The three
# worker-built forwards (seed / refine / supplement) are being wired (hooks below); they no-op until
# wired, so the worker currently serves identically to plain DFLASH.
from __future__ import annotations

import copy
import logging
import os
from copy import deepcopy
from typing import Optional

import torch

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import (
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

logger = logging.getLogger(__name__)


# NOTE: the refine uses the NATIVE causal MTP attention (DRAFT_EXTEND_V2 / causal extend) -- no custom
# mask. Our mtp2 backbone is (re)trained with causal-within-block refine (create_dflash_refine_mask),
# matching the MTP's native causal mode, so the sglang side needs no self-only custom mask.


class DFlashMtpWorkerV2(DFlashWorkerV2):
    """DFlash draft + native Qwen3.5 MTP multi-round refiner (own-KV, worker-managed forwards)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # builds the DFlash draft + everything
        self.mtp_refine_rounds = int(
            os.environ.get("SGLANG_DFLASH_MTP_REFINE_ROUNDS", "4")
        )
        # a+2..a+block use the refine; a+1 (slot 0) = the seam when use_seam, else fall back to keeping
        # the DFlash draft's own a+1 (refine_start, read once here -- never os.environ in the hot path).
        self._mtp_refine_start = int(os.environ.get("SGLANG_DFLASH_MTP_REFINE_START", "2"))
        # CUDA-GRAPH refine: the 2-round MTP forward is the only un-graphed piece of the decode chain
        # (~1.5ms eager = pure per-forward launch overhead). Opt in with SGLANG_DFLASH_REFINE_GRAPH=1
        # (default OFF; eager refine is the simple fallback and gives the same accept).
        self._refine_use_graph = (
            not self.server_args.disable_cuda_graph
            and os.environ.get("SGLANG_DFLASH_REFINE_GRAPH") == "1"
        )
        # Capture one graph per bs bucket (the decode buckets <= REFINE_MAX_BS, plus 1) so concurrency>1
        # is graphed too; replay pads the real bs up to the nearest bucket. kv_indices is sized
        # bs*block*max_context, so memory grows with bs -- hence the cap; bs above it -> eager.
        _decode_bs = list(self.server_args.cuda_graph_config.decode.bs)
        _max_refine_bs = int(os.environ.get("SGLANG_DFLASH_REFINE_MAX_BS", "16"))
        self._refine_capture_bs = sorted({1, *[b for b in _decode_bs if b <= _max_refine_bs]})
        self._refine_max_bs = max(self._refine_capture_bs)
        self._refine_graphs: dict = {}   # bs_bucket -> CUDAGraph
        self._refine_fbs: dict = {}      # bs_bucket -> ForwardBatch (views of the shared static buffers)
        self._refine_fwd_metas: dict = {}  # bs_bucket -> our draft-extend forward_metadata
        self._refine_static = None       # shared static buffers, sized for _refine_max_bs * block
        self._refine_graph_failed = False
        self._mtp_worker: Optional[TpModelWorker] = None
        self._mtp_runner = None
        self._mtp_model = None
        self._captured_final_hidden: Optional[torch.Tensor] = None
        # SEAM: the anchor (last-committed pos) post-norm TARGET final hidden, per request [bs, D];
        # block slot 0 (a+1) conditions on it (= HF dec_seam). Set at seed (prompt) + post-verify.
        self._seam_hidden: Optional[torch.Tensor] = None
        # GRAPH-SAFE final-hidden capture buffer. Under cuda-graph the python hook body does not re-run
        # on replay, but an in-place copy_ IS recorded as a kernel in the target verify graph, so the
        # buffer is refreshed every replay (a bare `= o0` rebind is NOT recorded -> froze the old path).
        # Allocated exactly like sglang's graph runners allocate their static I/O buffers (see
        # EAGLEDraftCudaGraphRunner "Static buffers": plain device tensor under `with torch.device`,
        # sized to the largest captured bucket = bs*block_size, held for the worker's lifetime, made
        # BEFORE the scheduler captures the target graph so the hook can target it during capture).
        # It lives on the worker rather than a runner because the graph that records the copy_ is the
        # TARGET model_runner's own CudaGraphRunner, which is MTP-agnostic. zeros (not empty) so a
        # missed/out-of-range copy reads 0 rather than another tensor's stale bytes.
        sa = self.server_args
        # Cover every captured decode-graph bucket (the bs list can exceed max_running_requests).
        max_bs = max(int(sa.max_running_requests or 256), int(sa.cuda_graph_config.decode.max_bs))
        tmr = self.target_worker.model_runner
        _hidden = int(tmr.model_config.hidden_size)
        with torch.device(self.device):
            self._norm_buf = torch.zeros(int(max_bs) * int(self.block_size), _hidden, dtype=tmr.dtype)
            # PER-REQUEST seam store, keyed by req_pool_index (NOT batch position). The seam follows a
            # request across batch-composition changes (continuous batching reorders / new prefills),
            # so concurrency>1 each request always refines on ITS OWN anchor hidden. Indexed by
            # req_pool_index (< max_running <= max_bs); _seam_valid gates requests that have one yet.
            self._seam_store = torch.zeros(int(max_bs), _hidden, dtype=tmr.dtype)
            self._seam_valid = torch.zeros(int(max_bs), dtype=torch.bool)
        self._init_mtp_refiner()

    # ----------------------------------------------------------------- instantiate (raw module)
    def _init_mtp_refiner(self):
        """Load the native Qwen3.5 MTP as a directly-held draft module (its own KV pool)."""
        sa = deepcopy(self.server_args)
        sa.skip_tokenizer_init = True
        sa.speculative_draft_model_path = self.server_args.model_path
        sa.speculative_draft_model_revision = "main"
        # Single full-attention layer (head_dim=256, mrope+partial rotary) -> fast full-attn backend
        # (the target itself resolves to triton because it is a linear+full hybrid).
        mtp_backend = os.environ.get("SGLANG_DFLASH_MTP_ATTN_BACKEND", "flashinfer")
        sa.speculative_draft_attention_backend = None
        sa.prefill_attention_backend = None
        sa.decode_attention_backend = None
        sa.attention_backend = mtp_backend
        sa.context_length = self.target_worker.model_runner.model_config.context_len

        saved = get_global_server_args()
        try:
            self._mtp_worker = TpModelWorker(
                server_args=sa,
                gpu_id=self.gpu_id,
                tp_rank=self.tp_rank,
                moe_ep_rank=self.moe_ep_rank,
                pp_rank=0,
                attn_cp_rank=self.attn_cp_rank,
                moe_dp_rank=self.moe_dp_rank,
                dp_rank=self.dp_rank,
                nccl_port=self.nccl_port,
                is_draft_worker=True,
            )
        finally:
            set_global_server_args_for_scheduler(saved)

        self._mtp_runner = self._mtp_worker.model_runner
        self._mtp_worker.draft_runner = self._mtp_runner
        self._mtp_model = self._mtp_runner.model
        # The MTP refiner shares the target embedding + lm_head.
        tgt = self.target_worker.model_runner.model
        if hasattr(self._mtp_model, "set_embed_and_head") and hasattr(
            tgt, "get_embed_and_head"
        ):
            embed, head = tgt.get_embed_and_head()
            self._mtp_model.set_embed_and_head(embed, head)
        if self.tp_rank == 0:
            logger.info(
                "Initialized DFLASH-MTP refiner module. model=%s, attn=%s, refine_rounds=%s",
                self._mtp_model.__class__.__name__,
                mtp_backend,
                self.mtp_refine_rounds,
            )
        self._register_final_hidden_hook()

    def _register_final_hidden_hook(self):
        """Hook the target language-model final norm to capture the POST-norm last hidden -- the MTP
        refiner's conditioning input. The checkpoint is post-norm trained (A/B: post-norm 4.498 vs
        pre-norm 4.309), so we feed out[0] (== inner.norm(last)) as-is; the MTP applies its own
        pre_fc_norm on top."""
        tgt = self.target_worker.model_runner.model
        inner = None
        for m in tgt.modules():
            if hasattr(m, "set_dflash_layers_to_capture") and hasattr(m, "norm"):
                inner = m
                break
        if inner is None:
            logger.warning("DFLASH-MTP: could not locate target final norm; refiner disabled.")
            return

        def _hook(_module, _inp, out):
            # out[0] = the single post-norm hidden (lm_head input) = the MTP's conditioning hidden.
            o0 = out[0] if isinstance(out, tuple) else out
            self._captured_final_hidden = o0.clone()
            # GRAPH-SAFE seam source: record an in-place copy of the post-norm hidden into the
            # persistent buffer. This copy_ kernel is captured in the target verify graph, so it
            # re-runs on every replay -> _live_norm_out (the buffer) always holds the CURRENT step
            # (a bare `= o0` rebind is a python op the graph never records -> the old path froze).
            # Eager prefills larger than the buffer fall back to o0 (prefill seeds from
            # _captured_final_hidden, not _live_norm_out, so that path is unused).
            n = o0.shape[0]
            buf = self._norm_buf
            if buf is not None and n <= buf.shape[0] and o0.shape[-1] == buf.shape[-1]:
                buf[:n].copy_(o0)
                self._live_norm_out = buf
            else:
                self._live_norm_out = o0

        inner.norm.register_forward_hook(_hook)

    # ----------------------------------------------------------------- pool / backend / cudagraph
    def alloc_memory_pool(
        self, memory_pool_config=None, req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        super().alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        self._mtp_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def init_attention_backends(self):
        super().init_attention_backends()
        self._mtp_worker.init_attention_backends()

    def init_cuda_graphs(self):
        super().init_cuda_graphs()
        # Initialize the MTP runner's eager_runner (needed by .forward) but skip its own decode/extend
        # graph capture -- the MTP forward runs eager; only the 2-round refine is graphed (below).
        self._mtp_worker.init_cuda_graphs(capture_decode_cuda_graph=False)
        # Capture the refine graphs HERE (startup capture phase, dummy inputs) rather than lazily on the
        # first real refine: lazy mid-serving capture aliased other graphs'/persistent KV/req_to_token
        # buffers and corrupted the target. All native cuda-graph runners capture at startup for this.
        if self._refine_use_graph and not self._refine_graph_failed:
            try:
                self._capture_refine_graph_startup()
                if self.tp_rank == 0:
                    logger.info("DFLASH-MTP refine graph captured at STARTUP")
            except Exception as e:  # noqa: BLE001
                import traceback
                logger.warning(
                    "DFLASH-MTP refine STARTUP cuda-graph capture failed (-> eager): %r\n%s",
                    e, traceback.format_exc(),
                )
                self._refine_graph_failed = True
                self._refine_graphs = {}

    # ----------------------------------------------------------------- worker-built MTP forwards
    @torch.inference_mode()
    def _maybe_extend_mtp_kv_prefill(self, model_worker_batch, next_token_ids):
        """SEED: run one MTP extend over the prompt -> seed the prefix-KV from the prompt's TARGET
        final hidden (norm hook). Mirrors _draft_extend_for_prefill but worker-managed (own runner +
        init_new -> correct attention plan), with save/restore so the DFlash prefill batch is intact.
        """
        if self._captured_final_hidden is None:
            return
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
        )
        from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
        from sglang.srt.speculative.eagle_utils import _eagle_prefill_tail_tokens

        batch = model_worker_batch
        if batch.forward_mode.is_idle():
            return
        saved_ids = batch.input_ids.clone()
        saved_spec = batch.spec_info
        saved_cap = batch.capture_hidden_mode
        saved_mode = batch.forward_mode
        # Capture extend_lens NOW: ForwardBatch.init_new (below) consumes batch.extend_lens -> None.
        saved_extend_lens = (
            batch.extend_lens.clone() if torch.is_tensor(getattr(batch, "extend_lens", None)) else None
        )
        try:
            # Shift each request's prompt by 1 (nextn input at pos p = token_{p+1}).
            tail = _eagle_prefill_tail_tokens(batch, next_token_ids)
            pt = 0
            for i, extend_len in enumerate(batch.extend_lens):
                ids = batch.input_ids[pt : pt + extend_len]
                batch.input_ids[pt : pt + extend_len] = torch.cat(
                    (ids[1:], tail[i].reshape(1))
                )
                pt += extend_len
            batch.spec_info = EagleDraftExtendInput(
                hidden_states=self._captured_final_hidden,
                num_tokens_per_req=1,
                num_tokens_for_logprob_per_req=1,
            )
            batch.capture_hidden_mode = CaptureHiddenMode.NULL
            fb = ForwardBatch.init_new(batch, self._mtp_runner)
            fb.return_logprob = False
            self._mtp_runner.forward(fb)  # writes the prompt nextn-states K/V into the MTP pool
            # SEAM (block 1): the eager prefill prunes the final-norm capture to the last token per req,
            # so _captured_final_hidden is [bs, D] (one anchor = last prompt token per req). Store each
            # request's anchor in the per-request seam store keyed by its req_pool_index.
            cfh0 = self._captured_final_hidden
            if cfh0 is not None and torch.is_tensor(cfh0) and cfh0.dim() == 2:
                rp = model_worker_batch.req_pool_indices[: cfh0.shape[0]].to(
                    self._seam_store.device, torch.long
                )
                self._seam_store[rp] = cfh0.detach().to(self._seam_store.dtype)
                self._seam_valid[rp] = True
        except Exception as e:  # noqa: BLE001
            if not getattr(self, "_warned_mtp_prefill", False):
                import traceback
                logger.warning(
                    "DFLASH-MTP seed (prefill) failed: %r | final_hidden=%s input_ids=%s "
                    "extend_lens=%s\n%s",
                    e,
                    tuple(self._captured_final_hidden.shape),
                    tuple(batch.input_ids.shape),
                    list(batch.extend_lens) if not torch.is_tensor(batch.extend_lens)
                    else batch.extend_lens.tolist(),
                    traceback.format_exc(),
                )
                self._warned_mtp_prefill = True
        finally:
            batch.input_ids = saved_ids
            batch.spec_info = saved_spec
            batch.capture_hidden_mode = saved_cap
            batch.forward_mode = saved_mode

    @torch.inference_mode()
    def _maybe_extend_mtp_kv_decode(self, commit_lens, bs, req_pool_indices):
        """CROSS-BLOCK SEAM capture (post-verify): the new anchor = the last committed token (row
        commit_len-1 of the verify block); its post-norm TARGET final hidden conditions the NEXT
        block's a+1 (= HF dec_seam). Stored PER req_pool_index so it follows the request across
        batch reorders/composition changes (concurrency>1). Reads the LIVE norm-out ref (graph-safe)."""
        lno = getattr(self, "_live_norm_out", None)
        bs_ = int(bs)
        block_size = int(self.block_size)
        if self._mtp_model is None or not torch.is_tensor(lno) or lno.shape[0] < bs_ * block_size:
            return
        segb = lno[: bs_ * block_size].view(bs_, block_size, -1)
        cl = commit_lens.to(torch.int64).clamp(min=1, max=block_size)
        seams = segb[torch.arange(bs_, device=segb.device), cl - 1]
        rp = req_pool_indices[:bs_].to(self._seam_store.device, torch.long)
        self._seam_store[rp] = seams.to(self._seam_store.dtype)
        self._seam_valid[rp] = True

    @torch.inference_mode()
    def _maybe_refine_block(
        self, draft_hidden, draft_tokens, model_worker_batch,
        block_positions_2d=None, block_cache_loc_2d=None, bs=None,
    ):
        """REFINE: run the MTP (K rounds) over the block (query the committed prefix-KV), decode ->
        refine draft_tokens in place. bs<=_refine_max_bs + seam -> cuda-graph replay; else eager.

        Block-state[j] = fc(cat(norm_e(embed(T[j])), norm_h(H[j]))); MTP slot j predicts token a+j+1
        -> writes draft_tokens[:, j+1]. Guarded so failure never breaks the DFlash baseline.
        """
        if block_cache_loc_2d is None or block_positions_2d is None or self._mtp_model is None:
            return
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

        block_size = int(self.block_size)
        bs_ = int(bs)
        # SEAM (a+1): the MTP's a+1 = decode at the anchor (last committed pos) on the TARGET final
        # hidden -- NOT the DFlash block-H slot 0 (untrained). Gather each request's anchor from the
        # PER-REQUEST seam store (keyed by req_pool_index, so it follows the request across batch
        # reorders / new prefills at concurrency>1). seam_h[i] = the seam for batch row i's request;
        # _seam_hidden mirrors it for the bs=1 cuda-graph path (which reads self._seam_hidden[0]).
        rp = model_worker_batch.req_pool_indices[:bs_].to(self._seam_store.device, torch.long)
        use_seam = bool(self._seam_valid[rp].all())
        seam_h = self._seam_store[rp] if use_seam else None
        self._seam_hidden = seam_h
        # CUDA-GRAPH fast path (bs=1 + seam): replay the captured 2-round refine (~0.3ms vs ~1.5ms
        # eager). Returns True when handled; False on the warmup steps / not-yet-captured -> fall
        # through to the eager path below (which also serves as the capture warmup).
        if (
            self._refine_use_graph and bs_ <= self._refine_max_bs and use_seam
            and not self._refine_graph_failed
        ):
            if self._refine_via_graph(
                draft_hidden, draft_tokens, model_worker_batch,
                block_positions_2d, block_cache_loc_2d, bs_,
            ):
                return
        # ---- eager path (bs>1 / no-seam fallback, and the warmup for graph capture) ----
        # Mutate a SHALLOW COPY (rebinding fields on the copy doesn't touch model_worker_batch), so we
        # don't need the old save-fields / finally-restore dance. The copy shares tensor refs; init_new
        # only reads them. The verify after refine still sees the original model_worker_batch intact.
        batch = copy.copy(model_worker_batch)
        device = self.device
        H = draft_hidden.reshape(-1, draft_hidden.shape[-1]).clone()  # [bs*block, hidden]
        if use_seam:
            H.view(bs_, block_size, -1)[:, 0] = seam_h[:bs_].to(H.dtype)  # slot-0 a+1 = anchor seam
        try:
            # Replicate prepare_for_draft_extend's field-setting (the recipe that builds a correct
            # multi-token extend attention plan: qo_indptr == cumsum(extend_lens)).
            # RoPE positions must match HF `_dense_mtp2`: refine_pos = draft_pos - 1 (the nextn 2-back
            # convention -- state at position p with next-token T[j] predicts p+2 = a+j+1). Without
            # the -1 the whole block is off by one in RoPE -> garbage refine. init_new uses
            # spec_info.positions when set (and does NOT overwrite it for draft-extend).
            refine_positions = (
                (block_positions_2d - 1).clamp(min=0).reshape(-1).to(torch.int64)
            )
            # Worker-managed ForwardBatch (the native prepare_for_draft_extend builds an accept-length
            # variable extend with qo=N+1, incompatible with our fixed-size block -> qo mismatch).
            batch.spec_info = EagleDraftExtendInput(
                hidden_states=H,
                num_tokens_per_req=block_size,
                num_tokens_for_logprob_per_req=block_size,
                positions=refine_positions,
            )
            batch.input_ids = draft_tokens.reshape(-1)
            batch.prefix_lens = batch.seq_lens.to(torch.int32)
            batch.extend_lens = torch.full(
                (bs_,), block_size, dtype=torch.int32, device=device
            )
            batch.extend_num_tokens = block_size * bs_
            batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
            batch.capture_hidden_mode = CaptureHiddenMode.NULL
            batch.out_cache_loc = block_cache_loc_2d.reshape(-1)
            fb = ForwardBatch.init_new(batch, self._mtp_runner)
            fb.seq_lens = fb.seq_lens + block_size
            self._mtp_runner.attn_backend.init_forward_metadata(fb)
            fb.mark_forward_metadata_ready()
            fb.return_logprob = False
            out = self._mtp_runner.forward(fb).logits_output
            logits = out.next_token_logits.view(bs_, block_size, -1)
            pred = logits.argmax(dim=-1)  # [bs, block]; MTP slot j -> token a+j+1
            new_tail = pred[:, : block_size - 1]
            # a+1 (slot 0) is the SEAM. If we injected the anchor target hidden into block slot 0
            # (use_seam), pred[0] IS the real seam -> write all of draft_tokens[1:] = pred[0:block-1].
            # Otherwise fall back to keeping the DFlash draft's own a+1 (refine_start=2), since the
            # untrained block-H slot 0 (refine_start=1) is garbage.
            if use_seam:
                refine_start = 1
            else:
                refine_start = self._mtp_refine_start
            if refine_start <= 1:
                draft_tokens[:, 1:].copy_(new_tail)
            else:
                # keep draft_tokens[:, 1] (DFlash a+1); refine a+2..a+block = pred[1..block-1]
                draft_tokens[:, refine_start:].copy_(new_tail[:, refine_start - 1:])
            # ROUND 2+ (design cap = 2; batch always has stragglers so one extra round suffices). H +
            # prefix-KV + positions/cache_loc are unchanged across rounds -> only the block TOKENS (T)
            # change, so just update fb.input_ids and re-forward (reuse fb; the MTP re-embeds T and
            # re-attends). The seam (slot 0) is stable (conditions on the anchor target hidden + anchor
            # token, both fixed). FIXED round count = cuda-graph-static control flow (no data-dependent
            # break) -> captures cleanly.
            for _r in range(1, max(1, int(self.mtp_refine_rounds))):
                fb.input_ids = draft_tokens.reshape(-1)
                out = self._mtp_runner.forward(fb).logits_output
                new_tail = out.next_token_logits.view(bs_, block_size, -1).argmax(dim=-1)[:, : block_size - 1]
                if refine_start <= 1:
                    draft_tokens[:, 1:].copy_(new_tail)
                else:
                    draft_tokens[:, refine_start:].copy_(new_tail[:, refine_start - 1:])
        except Exception as e:  # noqa: BLE001
            if not getattr(self, "_warned_mtp_refine", False):
                import traceback
                logger.warning("DFLASH-MTP refine failed: %r\n%s", e, traceback.format_exc())
                self._warned_mtp_refine = True

    # ----------------------------------------------------------------- refine cuda-graph (bs buckets)
    @torch.inference_mode()
    def _refine_via_graph(
        self, draft_hidden, draft_tokens, mwb, block_positions_2d, block_cache_loc_2d, bs
    ) -> bool:
        """Replay the captured refine for the smallest bs bucket >= bs (the extra rows are padded with
        the dummy seq_len=1 layout the bucket was captured with). Returns False if nothing is captured
        or bs exceeds the max bucket (caller runs eager). Only GPU work is captured; the flashinfer
        plan (variable prefix-KV page table) is refreshed per replay via init_forward_metadata_out_graph."""
        if not self._refine_graphs:
            return False
        bs_ = int(bs)
        bs_c = next((b for b in self._refine_capture_bs if b >= bs_), None)
        if bs_c is None:
            return False
        block = int(self.block_size)
        D = draft_hidden.shape[-1]
        s = self._refine_static
        fb = self._refine_fbs[bs_c]
        n, nc = bs_ * block, bs_c * block
        # live rows [:n]
        s.hidden[:n].copy_(draft_hidden.reshape(-1, D))
        s.hidden[:nc].view(bs_c, block, D)[:bs_, 0].copy_(self._seam_hidden[:bs_].to(s.hidden.dtype))
        s.block_tokens[:n].copy_(draft_tokens.reshape(-1))
        s.positions[:n].copy_((block_positions_2d - 1).clamp(min=0).reshape(-1).to(torch.int64))
        s.out_cache_loc[:n].copy_(block_cache_loc_2d.reshape(-1).to(s.out_cache_loc.dtype))
        s.seq_lens[:bs_].copy_(mwb.seq_lens[:bs_].to(torch.int64) + block)
        s.req_pool_indices[:bs_].copy_(mwb.req_pool_indices[:bs_])
        # padding rows [bs_:bs_c] -> the dummy (seq_len=1, req=0, slot 0) the bucket was captured with.
        if bs_c > bs_:
            s.seq_lens[bs_:bs_c].fill_(1)
            s.req_pool_indices[bs_:bs_c].zero_()
            s.out_cache_loc[n:nc].zero_()
            s.positions[n:nc].zero_()
            s.block_tokens[n:nc].zero_()
            s.hidden[n:nc].zero_()
        # CPU seq_lens mirror from the GPU source of truth: generate_attn_arg_prefill sizes kv_indices
        # from the CPU seq_lens_sum but the triton kernel WRITES sum(GPU seq_lens); CPU<GPU (overlap
        # scheduler lag) would overflow kv_indices -> illegal memory access.
        s.seq_lens_cpu[:bs_c].copy_(s.seq_lens[:bs_c].to("cpu"))
        fb.seq_lens_sum = int(s.seq_lens_cpu[:bs_c].sum())
        self._refine_attn_backend.init_forward_metadata_out_graph(fb, in_capture=False)
        self._refine_attn_backend.forward_metadata = self._refine_fwd_metas[bs_c]
        self._refine_graphs[bs_c].replay()
        draft_tokens.reshape(-1).copy_(s.block_tokens[:n])
        return True

    def _capture_refine_graph_startup(self):
        """Capture the refine graphs at STARTUP (cuda-graph capture phase) for every bs bucket with
        DUMMY inputs (seq_len=1, req=0). Real inputs are copied into the shared static buffers on every
        replay. Lazy mid-serving capture aliased other graphs'/persistent buffers and corrupted them."""
        if self._seam_hidden is None:  # serving overwrites this every step before the replay
            D = int(self.target_worker.model_runner.model_config.hidden_size)
            dt = self.target_worker.model_runner.dtype
            self._seam_hidden = torch.zeros(
                self._refine_max_bs, D, dtype=dt, device=self.device
            )
        self._capture_refine_graph()

    def _capture_refine_graph(self):
        """Capture the refine forward as a cuda graph per bs bucket in self._refine_capture_bs. The
        graphs share ONE set of max-bs static buffers + ONE dedicated flashinfer backend (its own
        workspace + cuda-graph state sized for the max bucket); each bucket plans/reads its [:bs] slice.
        Mirrors EAGLEDraftExtendCudaGraphRunner's per-bucket capture + pad-to-bucket replay."""
        from types import SimpleNamespace

        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
        from sglang.srt.layers.dp_attention import (
            DpPaddingMode,
            set_dp_buffer_len,
            set_is_extend_in_batch,
        )
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.model_executor.runner import model_capture_mode
        from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

        block = int(self.block_size)
        dev = self.device
        D = int(self.target_worker.model_runner.model_config.hidden_size)
        dt = self.target_worker.model_runner.dtype
        runner = self._mtp_runner
        be = runner.attn_backend
        be_full = getattr(be, "full_attn_backend", be)
        self._refine_is_hybrid = hasattr(be, "full_attn_backend")
        max_bs = int(self._refine_max_bs)
        max_n = max_bs * block
        rounds = max(1, int(self.mtp_refine_rounds))

        # DEDICATED draft-extend backend: its OWN flashinfer workspace (init_new_workspace=True) so the
        # per-step eager MTP KV-extend never clobbers the captured plan; cuda-graph state sized for the
        # max bucket and shared across bucket graphs (each plans its [:bs] slice).
        self._refine_be = FlashInferAttnBackend(
            runner, skip_prefill=False, init_new_workspace=True
        )
        self._refine_attn_backend = self._refine_be

        # Shared static buffers (max bucket). Each bucket graph reads/writes its [:bs*block] slice; the
        # replay fills [:real_bs] live and [real_bs:bs] with the dummy (seq_len=1) padding.
        _vocab = int(runner.model_config.vocab_size)
        s = SimpleNamespace()
        s.block_tokens = torch.zeros(max_n, dtype=torch.int64, device=dev)
        s.hidden = torch.zeros(max_n, D, dtype=dt, device=dev)
        s.positions = torch.zeros(max_n, dtype=torch.int64, device=dev)
        s.out_cache_loc = torch.zeros(max_n, dtype=torch.int64, device=dev)
        s.seq_lens = torch.zeros(max_bs, dtype=torch.int64, device=dev)
        s.seq_lens_cpu = torch.zeros(max_bs, dtype=torch.int64, device="cpu")
        s.req_pool_indices = torch.zeros(max_bs, dtype=torch.int64, device=dev)
        s.extend_seq_lens = torch.full((max_bs,), block, dtype=torch.int32, device=dev)
        s.num_correct_drafts = torch.full((max_bs,), block, dtype=torch.int32, device=dev)
        s.num_accept_tokens = torch.full((max_bs,), block, dtype=torch.int32, device=dev)
        s.next_token_logits_buffer = torch.zeros(max_n, _vocab, dtype=torch.float, device=dev)
        self._refine_static = s

        # argmax slices to the true embedding rows (lm_head is padded wider; an OOB index would be
        # re-embedded out of bounds in round 2 / the verify).
        _emb = self.target_worker.model_runner.model.get_embed_and_head()[0]
        vlim = int((_emb.weight if hasattr(_emb, "weight") else _emb).shape[0])
        # Share the TARGET verify/decode graph's mempool (its own full_cuda_graph_backend._pool) so
        # torch deconflicts the refine graphs' activations against it.
        _pool = self.target_worker.model_runner.decode_cuda_graph_runner.backend._pool

        def _capture_bucket(bs_c):
            n = bs_c * block
            s.seq_lens[:bs_c].fill_(1)
            s.seq_lens_cpu[:bs_c].fill_(1)
            s.req_pool_indices[:bs_c].zero_()
            s.out_cache_loc[:n].zero_()
            s.positions[:n].zero_()
            s.block_tokens[:n].zero_()
            spec = EagleDraftExtendInput(
                hidden_states=s.hidden[:n],
                num_correct_drafts=s.num_correct_drafts[:bs_c],
                num_accept_tokens=s.num_accept_tokens[:bs_c],
                num_tokens_per_req=block,
                num_tokens_for_logprob_per_req=block,
                positions=s.positions[:n],
            )
            spec.extend_seq_lens_cpu = [block] * bs_c
            spec.extend_seq_lens_tensor = s.extend_seq_lens[:bs_c]
            fb = ForwardBatch(
                forward_mode=ForwardMode.DRAFT_EXTEND_V2,
                batch_size=bs_c,
                input_ids=s.block_tokens[:n],
                req_pool_indices=s.req_pool_indices[:bs_c],
                seq_lens=s.seq_lens[:bs_c],
                seq_lens_cpu=s.seq_lens_cpu[:bs_c],
                seq_lens_sum=int(s.seq_lens_cpu[:bs_c].sum()),
                out_cache_loc=s.out_cache_loc[:n],
                positions=s.positions[:n],
                extend_seq_lens=s.extend_seq_lens[:bs_c],
                next_token_logits_buffer=s.next_token_logits_buffer[:n],
                spec_algorithm=runner.spec_algorithm,
                spec_info=spec,
                capture_hidden_mode=CaptureHiddenMode.LAST,
                return_logprob=False,
                dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            )

            def _run():
                fb.dp_local_start_pos = fb.dp_local_num_tokens = None
                set_dp_buffer_len(None, n, fb.dp_padding_mode.is_max_len(), None)
                set_is_extend_in_batch(False)
                for _r in range(rounds):
                    _ocl = fb.out_cache_loc
                    _hs = fb.spec_info.hidden_states
                    out = runner.model.forward(fb.input_ids, fb.positions, fb)
                    fb.out_cache_loc = _ocl
                    fb.spec_info.hidden_states = _hs
                    nt = out.next_token_logits[:, :vlim].view(bs_c, block, -1)
                    s.block_tokens[:n].view(bs_c, block)[:, 1:].copy_(
                        nt.argmax(dim=-1)[:, : block - 1]
                    )

            with forward_context(ForwardContext(attn_backend=be)):
                self._refine_be.init_forward_metadata_out_graph(fb, in_capture=True)
                self._refine_fwd_metas[bs_c] = self._refine_be.forward_metadata
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        _run()
                torch.cuda.current_stream().wait_stream(stream)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=_pool):
                    _run()
            self._refine_graphs[bs_c] = g
            self._refine_fbs[bs_c] = fb

        if self._refine_is_hybrid:
            be.full_attn_backend = self._refine_be
        try:
            with model_capture_mode():
                self._refine_be.init_cuda_graph_state(max_bs, max_n)
                for bs_c in self._refine_capture_bs:
                    _capture_bucket(bs_c)
        finally:
            if self._refine_is_hybrid:
                be.full_attn_backend = be_full
        if self.tp_rank == 0:
            logger.info(
                "DFLASH-MTP refine cuda-graph captured: bs buckets=%s block=%d rounds=%d",
                self._refine_capture_bs, block, rounds,
            )
