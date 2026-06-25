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
#   - Two cuda-graph types (phase 4): (1) DFlash+lm_head+MTP+lm_head, (2) MTP+lm_head (extra rounds).
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
        # CUDA-GRAPH refine (bs=1): the 2-round MTP forward is the only un-graphed piece of the
        # decode chain (~1.5ms eager = pure per-forward launch overhead). Capture it into a single
        # graph -> ~0.9ms. ponytail: default OFF (eager refine is the validated path, accept 4.298) --
        # the graph captures but a long-sequence KV corruption remains open; opt in with
        # SGLANG_DFLASH_REFINE_GRAPH=1.
        self._refine_use_graph = (
            not self.server_args.disable_cuda_graph
            and os.environ.get("SGLANG_DFLASH_REFINE_GRAPH") == "1"
        )
        self._refine_graph = None
        self._refine_static = None
        self._refine_graph_failed = False
        self._refine_warmup = 0
        self._mtp_worker: Optional[TpModelWorker] = None
        self._mtp_runner = None
        self._mtp_model = None
        self._captured_final_hidden: Optional[torch.Tensor] = None
        # SEAM: the anchor (last-committed pos) double-normed TARGET final hidden, per request [bs, D];
        # block slot 0 (a+1) conditions on it (= HF dec_seam). Set at seed (prompt) + post-verify.
        self._seam_hidden: Optional[torch.Tensor] = None
        # GRAPH-SAFE final-hidden capture: a PERSISTENT buffer the norm hook copies into in-place.
        # Under cuda-graph the python hook body does not re-run on replay, but an in-place copy_ IS
        # recorded as a kernel in the target verify graph, so the buffer is refreshed every replay
        # (a bare `= o0` rebind is NOT recorded -> froze the old path). Allocated here (before the
        # scheduler captures the target graph) so the hook can target it during capture. Sized for the
        # largest captured decode batch (bs*block_size); larger eager prefills fall back to a fresh ref.
        sa = self.server_args
        max_bs = int(getattr(sa, "max_running_requests", None) or 256)
        try:  # cover every captured decode-graph bucket (the list can exceed max_running_requests)
            max_bs = max(max_bs, int(sa.cuda_graph_config.decode.max_bs))
        except Exception:  # noqa: BLE001
            pass
        tmr = self.target_worker.model_runner
        self._norm_buf = torch.empty(
            int(max_bs) * int(self.block_size),
            int(tmr.model_config.hidden_size),
            dtype=tmr.dtype,
            device=self.device,
        )
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
        """Hook the target language-model final norm to capture the post-norm hidden (eager only;
        cuda-graph-safe capture comes in phase 4)."""
        tgt = self.target_worker.model_runner.model
        inner = None
        for m in tgt.modules():
            if hasattr(m, "set_dflash_layers_to_capture") and hasattr(m, "norm"):
                inner = m
                break
        if inner is None:
            logger.warning("DFLASH-MTP: could not locate target final norm; refiner disabled.")
            return

        # mtp2 was TRAINED on a DOUBLE-normed final hidden: HF `inner.norm(outputs.hidden_states[-1])`
        # where hidden_states[-1] was ALREADY post-norm in the training transformers version, so the
        # MTP conditions on norm(norm(x))*(1+w), NOT the single post-norm sglang produces. Confirmed
        # quantitatively: measured HF/sgl per-dim ratio == (1+w)/RMS (the signature of a 2nd RMSNorm).
        # Reproduce it here (manual, to avoid re-entrant hook recursion). Toggle off for the native path.
        self._mtp_double_norm = os.environ.get("SGLANG_DFLASH_MTP_DOUBLE_NORM", "1") == "1"
        _eps = float(getattr(inner.norm, "variance_epsilon", 1e-6))

        def _hook(_module, _inp, out):
            # out[0] = the gemma single post-norm hidden (lm_head input) = sglang's standard MTP hidden.
            o0 = out[0] if isinstance(out, tuple) else out
            if self._mtp_double_norm:
                x2 = o0.float()
                rms2 = x2 * torch.rsqrt(x2.pow(2).mean(-1, keepdim=True) + _eps)
                h = (rms2 * (1.0 + _module.weight.detach().float())).to(o0.dtype)
            else:
                h = o0.clone()
            self._captured_final_hidden = h
            # GRAPH-SAFE seam source: record an in-place copy of the post-norm hidden into the
            # persistent buffer. This copy_ kernel is captured in the target verify graph, so it
            # re-runs on every replay -> _live_norm_out (the buffer) always holds the CURRENT step
            # (a bare `= o0` rebind is a python op the graph never records -> the old path froze).
            # The decode seam double-norms it. Eager prefills larger than the buffer fall back to o0
            # (prefill seeds from _captured_final_hidden, not _live_norm_out, so that path is unused).
            n = o0.shape[0]
            buf = self._norm_buf
            if buf is not None and n <= buf.shape[0] and o0.shape[-1] == buf.shape[-1]:
                buf[:n].copy_(o0)
                self._live_norm_out = buf
            else:
                self._live_norm_out = o0
            self._seam_norm_w = _module.weight
            self._seam_norm_eps = _eps

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
        # Capture the 2-round refine graph HERE (startup capture phase, dummy inputs, shared graph
        # pool) rather than lazily on the first real refine. Lazy mid-serving capture let the refine
        # graph's memory alias other graphs'/persistent KV/req_to_token buffers -> target-side
        # corruption (bug 3). All native cuda-graph runners capture at startup for this reason.
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
                self._refine_graph = None

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
            # so _captured_final_hidden is already [bs, D]; its LAST row = the anchor (last prompt token)
            # that the block-1 a+1 seam conditions on (works for [1,D] pruned or [S,D] full; bs=1).
            cfh0 = self._captured_final_hidden
            if cfh0 is not None and torch.is_tensor(cfh0) and cfh0.dim() == 2:
                self._seam_hidden = cfh0[-1:].detach().clone()  # [1, D]
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
    def _maybe_extend_mtp_kv_decode(self, commit_lens, bs):
        """CROSS-BLOCK SEAM capture (post-verify): the new anchor = the last committed token (row
        commit_len-1 of the verify block); its double-normed TARGET final hidden conditions the NEXT
        block's a+1 (= HF dec_seam). Reads the LIVE norm-out ref (graph-safe across cuda-graph replays).
        (We deleted the old prefix-KV supplement: it overwrote the refine's RoPE(pos-1) prefix with
        RoPE(pos) and POLLUTED it -- the refine's own KV writes are the correct prefix.)"""
        lno = getattr(self, "_live_norm_out", None)
        bs_ = int(bs)
        block_size = int(self.block_size)
        if self._mtp_model is None or not torch.is_tensor(lno) or lno.shape[0] < bs_ * block_size:
            return
        seg = lno[: bs_ * block_size]
        if self._mtp_double_norm:
            x2 = seg.float()
            rms2 = x2 * torch.rsqrt(x2.pow(2).mean(-1, keepdim=True) + self._seam_norm_eps)
            seg = (rms2 * (1.0 + self._seam_norm_w.detach().float())).to(lno.dtype)
        segb = seg.view(bs_, block_size, -1)
        cl = commit_lens.to(torch.int64).clamp(min=1, max=block_size)
        self._seam_hidden = segb[torch.arange(bs_, device=segb.device), cl - 1].clone()

    @torch.inference_mode()
    def _maybe_refine_block(
        self, draft_hidden, draft_tokens, model_worker_batch,
        block_positions_2d=None, block_cache_loc_2d=None, bs=None,
    ):
        """REFINE: run the MTP over the block (query the committed prefix-KV), decode -> refine
        draft_tokens in place. First version: single-round extend over the block (reuses the seed's
        init_new pattern). TODO: self-only mask (no block<->block) + multi-round + supplements.

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
        seam_h = getattr(self, "_seam_hidden", None)
        # SEAM (a+1): in Track-2 the MTP's a+1 = decode at the anchor (last committed pos) on the
        # TARGET final hidden -- NOT the DFlash block-H slot 0 (which is untrained). Inject the anchor
        # target hidden into block slot 0 so refine pred[0] becomes the real seam (== HF dec_seam).
        use_seam = (
            seam_h is not None and torch.is_tensor(seam_h) and seam_h.shape[0] >= bs_
        )
        # CUDA-GRAPH fast path (bs=1 + seam): replay the captured 2-round refine (~0.3ms vs ~1.5ms
        # eager). Returns True when handled; False on the warmup steps / not-yet-captured -> fall
        # through to the eager path below (which also serves as the capture warmup).
        if self._refine_use_graph and bs_ == 1 and use_seam and not self._refine_graph_failed:
            if self._refine_via_graph(
                draft_hidden, draft_tokens, model_worker_batch,
                block_positions_2d, block_cache_loc_2d,
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
            # break) -> captures cleanly in phase 4.
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

    # ----------------------------------------------------------------- refine cuda-graph (bs=1)
    @torch.inference_mode()
    def _refine_via_graph(
        self, draft_hidden, draft_tokens, mwb, block_positions_2d, block_cache_loc_2d
    ) -> bool:
        """Replay the captured 2-round refine. Returns True when handled; False on warmup steps or
        capture failure (caller then runs the eager path). Only the GPU work is captured -- the
        flashinfer plan (variable prefix-KV page table) is refreshed eagerly per step via
        init_forward_metadata_out_graph, exactly like the native draft-extend cuda-graph runner."""
        block = int(self.block_size)
        D = draft_hidden.shape[-1]
        if self._refine_graph is None:
            # A couple eager warmups first so cuBLAS/flashinfer lazy-init before the capture.
            self._refine_warmup += 1
            if self._refine_warmup <= 2:
                return False
            try:
                self._capture_refine_graph(
                    draft_hidden, draft_tokens, mwb, block_positions_2d, block_cache_loc_2d
                )
            except Exception as e:  # noqa: BLE001
                import traceback
                logger.warning(
                    "DFLASH-MTP refine cuda-graph capture failed (-> eager): %r\n%s",
                    e, traceback.format_exc(),
                )
                self._refine_graph_failed = True
                self._refine_graph = None
                return False

        s, fb = self._refine_static, self._refine_fb
        # replay-prep: copy this step's inputs into the static buffers (one-ended; small DMA copies).
        s.hidden.copy_(draft_hidden.reshape(-1, D))
        s.hidden[0].copy_(self._seam_hidden[0].to(s.hidden.dtype))  # slot-0 a+1 seam
        s.block_tokens.copy_(draft_tokens.reshape(-1))
        s.positions.copy_((block_positions_2d - 1).clamp(min=0).reshape(-1).to(torch.int64))
        s.out_cache_loc.copy_(block_cache_loc_2d.reshape(-1).to(s.out_cache_loc.dtype))
        s.seq_lens.copy_(mwb.seq_lens[:1].to(torch.int64) + block)
        if mwb.seq_lens_cpu is not None:
            s.seq_lens_cpu.copy_(mwb.seq_lens_cpu[:1].to(torch.int64) + block)
        else:
            s.seq_lens_cpu.copy_(s.seq_lens.to("cpu"))
        s.req_pool_indices.copy_(mwb.req_pool_indices[:1])
        fb.seq_lens_sum = int(s.seq_lens_cpu.item())  # cpu .item() -> no GPU sync
        # refresh the draft-extend page table from the new prefix, then replay the baked graph.
        self._refine_attn_backend.init_forward_metadata_out_graph(fb, in_capture=False)
        # Re-assign OUR draft-extend metadata (a seed/decode on this backend may have clobbered it).
        self._refine_attn_backend.forward_metadata = self._refine_fwd_meta
        self._refine_graph.replay()
        draft_tokens.reshape(-1).copy_(s.block_tokens)
        return True

    def _capture_refine_graph_startup(self):
        """Capture the refine graph at STARTUP with DUMMY inputs (seq_lens=1, req=0, slots 0..block-1),
        in the cuda-graph capture phase -- exactly like the native runners. Lazy mid-serving capture
        let the refine graph's memory alias other graphs'/persistent buffers, corrupting the target.
        Values are irrelevant (the real inputs are copied into the static buffers on every replay)."""
        from types import SimpleNamespace

        block = int(self.block_size)
        dev = self.device
        D = int(self.target_worker.model_runner.model_config.hidden_size)
        dt = self.target_worker.model_runner.dtype
        draft_hidden = torch.zeros(block, D, dtype=dt, device=dev)
        draft_tokens = torch.zeros(block, dtype=torch.int64, device=dev)
        block_positions_2d = (torch.arange(block, device=dev, dtype=torch.int64) + 1).view(1, block)
        block_cache_loc_2d = torch.arange(block, dtype=torch.int64, device=dev).view(1, block)
        mwb = SimpleNamespace(
            seq_lens=torch.ones(1, dtype=torch.int64, device=dev),
            seq_lens_cpu=torch.ones(1, dtype=torch.int64, device="cpu"),
            req_pool_indices=torch.zeros(1, dtype=torch.int64, device=dev),
        )
        if self._seam_hidden is None:  # serving overwrites this every step before the replay
            self._seam_hidden = torch.zeros(block, D, dtype=dt, device=dev)
        self._capture_refine_graph(
            draft_hidden, draft_tokens, mwb, block_positions_2d, block_cache_loc_2d
        )

    def _capture_refine_graph(
        self, draft_hidden, draft_tokens, mwb, block_positions_2d, block_cache_loc_2d
    ):
        from types import SimpleNamespace

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
        D = draft_hidden.shape[-1]
        dev = self.device
        runner = self._mtp_runner
        be = runner.attn_backend
        # Qwen3.5 is a linear+full hybrid -> runner.attn_backend is HybridLinearAttnBackend. Its
        # eager init_forward_metadata has a draft_extend_v2 shortcut that drives ONLY the full-attn
        # sub-backend (the MTP is full-attention only: mtp full_attention_interval=1, no mamba), but
        # init_forward_metadata_out_graph loops ALL sub-backends and the linear one rejects
        # draft_extend ("Invalid forward mode"). Mirror the eager shortcut: plan the full-attn
        # sub-backend's draft-extend cuda-graph directly. The model still runs through the hybrid
        # wrapper (forward_context below), whose forward_extend routes the full-attn layer to it.
        be_full = getattr(be, "full_attn_backend", be)
        self._refine_is_hybrid = hasattr(be, "full_attn_backend")
        # DEDICATED draft-extend backend for refine. The per-step eager MTP KV-extend (writing
        # accepted tokens into the MTP pool, runner.forward) and my captured refine graph would
        # otherwise SHARE be_full's flashinfer prefill wrapper: the eager extend re-plans that
        # wrapper for the GROWING sequence every decode step, clobbering the captured graph's baked
        # wrapper/workspace state -> length-dependent illegal-memory-access mid-request (bug 2).
        # Give refine its own FlashInferAttnBackend (own wrappers + workspace), exactly like the
        # native EAGLEDraftExtendCudaGraphRunner's dedicated draft_extend_attn_backend.
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        self._refine_be = FlashInferAttnBackend(self._mtp_runner, skip_prefill=False)
        self._refine_attn_backend = self._refine_be
        rounds = max(1, int(self.mtp_refine_rounds))

        s = SimpleNamespace()
        s.block_tokens = torch.zeros(block, dtype=torch.int64, device=dev)
        s.hidden = torch.zeros(block, D, dtype=draft_hidden.dtype, device=dev)
        s.positions = torch.zeros(block, dtype=torch.int64, device=dev)
        s.out_cache_loc = torch.zeros(block, dtype=block_cache_loc_2d.dtype, device=dev)
        s.seq_lens = torch.zeros(1, dtype=torch.int64, device=dev)
        s.seq_lens_cpu = torch.zeros(1, dtype=torch.int64, device="cpu")
        s.req_pool_indices = torch.zeros(1, dtype=mwb.req_pool_indices.dtype, device=dev)
        s.extend_seq_lens = torch.full((1,), block, dtype=torch.int32, device=dev)
        # The draft-extend cuda-graph qo path (generate_attn_arg_prefill) reads these for bs/layout;
        # the eager (non-graph) qo path does not, so the eager refine never set them. Fixed block ->
        # constant num_tokens_per_req=block layout, so the VALUES only need to be a [bs=1] tensor.
        s.num_correct_drafts = torch.full((1,), block, dtype=torch.int32, device=dev)
        s.num_accept_tokens = torch.full((1,), block, dtype=torch.int32, device=dev)
        # Pre-allocated logits output buffer, exactly like the native draft-extend runner. Without it
        # the LogitsProcessor allocates a fresh [block, vocab] (~8MB) tensor INSIDE the graph pool on
        # every forward; that un-managed allocation is a candidate for the graph-pool aliasing that
        # corrupts the target. Providing a fixed buffer makes the logits matmul write to stable memory.
        _vocab = int(runner.model_config.vocab_size)
        s.next_token_logits_buffer = torch.zeros(block, _vocab, dtype=torch.float, device=dev)

        # Seed the static buffers from THIS (real) call so capture runs on valid data.
        s.hidden.copy_(draft_hidden.reshape(-1, D))
        s.hidden[0].copy_(self._seam_hidden[0].to(s.hidden.dtype))
        s.block_tokens.copy_(draft_tokens.reshape(-1))
        s.positions.copy_((block_positions_2d - 1).clamp(min=0).reshape(-1).to(torch.int64))
        s.out_cache_loc.copy_(block_cache_loc_2d.reshape(-1).to(s.out_cache_loc.dtype))
        s.seq_lens.copy_(mwb.seq_lens[:1].to(torch.int64) + block)
        if mwb.seq_lens_cpu is not None:
            s.seq_lens_cpu.copy_(mwb.seq_lens_cpu[:1].to(torch.int64) + block)
        else:
            s.seq_lens_cpu.copy_(s.seq_lens.to("cpu"))
        s.req_pool_indices.copy_(mwb.req_pool_indices[:1])

        # CAPTURE against a DUMMY page table (seq_lens=1, req_pool_indices=0 -> the reserved all-zero
        # req_to_token row = kv slot 0), exactly like the native EAGLEDraftExtendCudaGraphRunner
        # (it captures with seq_len_fill_value=1, req=0). Capturing on the REAL mid-generation page
        # table is what triggered the rare illegal-memory-access -- the flashinfer kernel must be
        # recorded against the canonical seq_len=1 dummy, then have its kv_indptr/kv_indices refreshed
        # per replay (init_forward_metadata_out_graph below). Real values are written in the replay path.
        s.seq_lens.fill_(1)
        s.seq_lens_cpu.fill_(1)
        s.req_pool_indices.zero_()

        spec = EagleDraftExtendInput(
            hidden_states=s.hidden,
            num_correct_drafts=s.num_correct_drafts,
            num_accept_tokens=s.num_accept_tokens,
            num_tokens_per_req=block,
            num_tokens_for_logprob_per_req=block,
            positions=s.positions,
        )
        spec.extend_seq_lens_cpu = [block]
        spec.extend_seq_lens_tensor = s.extend_seq_lens
        fb = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            batch_size=1,
            input_ids=s.block_tokens,
            req_pool_indices=s.req_pool_indices,
            seq_lens=s.seq_lens,
            seq_lens_cpu=s.seq_lens_cpu,
            seq_lens_sum=int(s.seq_lens_cpu.item()),
            out_cache_loc=s.out_cache_loc,
            positions=s.positions,
            extend_seq_lens=s.extend_seq_lens,
            next_token_logits_buffer=s.next_token_logits_buffer,
            spec_algorithm=runner.spec_algorithm,
            spec_info=spec,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            return_logprob=False,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
        )

        # The lm_head/logits are padded WIDER than the embedding table (ParallelLMHead alignment);
        # argmax over the padded width can pick an index >= embedding rows -> round-2 (or the verify)
        # re-embeds that token and the gather goes out of bounds. The eager path goes through
        # runner.forward whose logits processor handles this; calling model.forward directly does not.
        # Slice the logits to the actual embedding row count (the true gather bound) before argmax.
        try:
            _emb = self.target_worker.model_runner.model.get_embed_and_head()[0]
            _emb_w = _emb.weight if hasattr(_emb, "weight") else _emb
            vlim = int(_emb_w.shape[0])
        except Exception:  # noqa: BLE001
            vlim = int(runner.model_config.vocab_size)
        if self.tp_rank == 0:
            logger.info("DFLASH-MTP refine logits slice: embed_rows=%d vocab_cfg=%d",
                        vlim, int(runner.model_config.vocab_size))

        def _run():
            # Mirror model_runner._forward_raw -> _prepare_eager_forward_batch / the native
            # run_once preamble: set the DP/MLP-sync buffer length and dp-local fields. Qwen3.5 is
            # MoE -> its expert dispatch sizes a gathered buffer from global_dp_buffer_len; calling
            # model.forward DIRECTLY (bypassing runner.forward) leaves this at the previous TARGET
            # forward's (larger decode) value, so the MoE gather/scatter is baked oversized at capture
            # -> writes past the buffer at replay -> corrupts memory the target reads later (bug 3).
            fb.dp_local_start_pos = fb.dp_local_num_tokens = None
            set_dp_buffer_len(None, block, fb.dp_padding_mode.is_max_len(), None)
            set_is_extend_in_batch(False)
            for _r in range(rounds):  # round 2+ reuses the same fb (input_ids updated in place)
                # model.forward MUTATES fb.out_cache_loc and fb.spec_info.hidden_states (the draft-extend
                # repoints them mid-forward). Back up and restore around EVERY forward exactly like the
                # native EAGLEDraftExtendCudaGraphRunner.run_once, otherwise the static out_cache_loc
                # buffer gets repointed to a transient tensor after round 1 -> later rounds/replays (and
                # the verify) write KV to garbage slots -> progressive KV-pool corruption (bug 3).
                _ocl_bak = fb.out_cache_loc
                _hs_bak = fb.spec_info.hidden_states
                out = runner.model.forward(fb.input_ids, fb.positions, fb)
                fb.out_cache_loc = _ocl_bak
                fb.spec_info.hidden_states = _hs_bak
                nt = out.next_token_logits[:, :vlim].view(1, block, -1)
                # use_seam is always true on this path -> refine_start=1: write draft_tokens[1:].
                s.block_tokens.view(1, block)[:, 1:].copy_(nt.argmax(dim=-1)[:, : block - 1])

        # Route the hybrid wrapper's full sub-backend to the DEDICATED refine backend for the whole
        # capture so the model forward (through the hybrid) uses the dedicated wrappers, then restore
        # so the per-step eager MTP KV-extend keeps using the original shared backend.
        if self._refine_is_hybrid:
            be.full_attn_backend = self._refine_be
        try:
            with model_capture_mode():
                self._refine_be.init_cuda_graph_state(1, block)
                with forward_context(ForwardContext(attn_backend=be)):
                    self._refine_be.init_forward_metadata_out_graph(fb, in_capture=True)
                    # Stash our draft-extend metadata; re-assigned before every replay as belt-and-suspenders
                    # (the dedicated backend is not shared, so nothing should clobber it).
                    self._refine_fwd_meta = self._refine_be.forward_metadata
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(2):
                            _run()
                    torch.cuda.current_stream().wait_stream(stream)
                    # Capture into the SHARED global graph memory pool, exactly like the native
                    # cuda-graph runners (capture_one -> graph_pool_handle()). A private pool (the
                    # default torch.cuda.graph(g) with no pool) is NOT deconflicted against the target's
                    # decode/draft graphs, so its captured activations can overlap their memory ->
                    # corruption that surfaces only after enough steps (the shifting illegal-access /
                    # store-assert symptom). Sharing the pool lets torch's graph allocator deconflict.
                    from sglang.srt.model_executor.runner import (
                        get_global_graph_memory_pool,
                        set_global_graph_memory_pool,
                    )

                    _pool = get_global_graph_memory_pool()
                    if _pool is None:
                        _pool = torch.cuda.graph_pool_handle()
                        set_global_graph_memory_pool(_pool)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=_pool):
                        _run()
        finally:
            if self._refine_is_hybrid:
                be.full_attn_backend = be_full

        self._refine_static = s
        self._refine_fb = fb
        self._refine_graph = g
        if self.tp_rank == 0:
            logger.info(
                "DFLASH-MTP refine cuda-graph captured (bs=1, block=%d, rounds=%d)",
                block, rounds,
            )
