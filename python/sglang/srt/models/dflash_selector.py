# DFlash candidate-selector draft model (P160 family: direct-edge parallel-scan
# selector, r=256, K=16). Shares the DFlash Qwen3 backbone; the only extra weights
# are three projections that turn the draft's per-position hidden states + the
# target lm_head top-K candidates into a K x K transition lattice, from which a
# block of draft tokens is decoded (greedy) or sampled (T=1, lossless).
#
# Ported verbatim from the training/inference reference (only the P160-active
# configuration: parallel_scan=True, direct_edge=True, memory_order=1,
# select_position_zero=False, global_chain=False, parallel_marginal=False).

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.models.dspark import gather_and_crop_vocab

logger = logging.getLogger(__name__)


def _rms_normalize(values: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(values, (values.shape[-1],), eps=eps)


class CandidateSelector(nn.Module):
    """Direct-edge parallel-scan candidate selector (P160).

    Holds exactly three learned projections (all bias-free):
      - token_projection : hidden_size -> state_rank
      - hidden_projection: hidden_size -> state_rank
      - state_query      : state_rank  -> state_rank
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        state_rank: int,
        top_k: int,
        block_size: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.state_rank = int(state_rank)
        self.top_k = int(top_k)
        self.block_size = int(block_size)
        self.rms_norm_eps = float(rms_norm_eps)

        self.token_projection = nn.Linear(hidden_size, state_rank, bias=False)
        self.hidden_projection = nn.Linear(hidden_size, state_rank, bias=False)
        self.state_query = nn.Linear(state_rank, state_rank, bias=False)

    # -- lattice construction ------------------------------------------------

    def _candidate_unaries(
        self, base_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k candidate logits/ids; slot 0 is the highest-logit (greedy top-1) token.

        `topk` returns descending, so slot 0 == the DFlash top-1. Training additionally
        forced `base_logits.argmax` into slot 0 to (a) pass a startup equivalence assert
        and (b) stay deterministic under bf16 ties; inference needs neither — the target
        verify makes greedy/T=1 lossless regardless of which tied token slot 0 holds.
        # ponytail: dropped the argmax swap/replace; topk slot 0 suffices at inference.
        """
        unary_logits, candidate_ids = torch.topk(base_logits, self.top_k, dim=-1)
        return unary_logits.float(), candidate_ids

    def prepare_lattice(
        self,
        *,
        base_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        embedding_weight: torch.Tensor,
    ):
        """Return (candidate_ids, unary_logits, candidate_factors, hidden_factors).

        Shapes: candidate_ids/unary_logits [B, L, K]; candidate_factors [B, L, K, r];
        hidden_factors [B, L, r].
        """
        unary_logits, candidate_ids = self._candidate_unaries(base_logits)
        candidate_embeddings = F.embedding(candidate_ids, embedding_weight)
        candidate_factors = self.token_projection(
            _rms_normalize(candidate_embeddings, self.rms_norm_eps)
        )
        hidden_factors = self.hidden_projection(
            _rms_normalize(hidden_states, self.rms_norm_eps)
        )
        return candidate_ids, unary_logits, candidate_factors, hidden_factors

    def transition_scores(
        self,
        *,
        unary_logits: torch.Tensor,
        candidate_factors: torch.Tensor,
        hidden_factors: torch.Tensor,
    ) -> torch.Tensor:
        """Local K x K transition scores [B, L-1, K, K].

        transition[b, e, p, c] = unary[b, e+1, c]
            + <state_query(silu(cand_factor[e, p] + hidden_factor[e+1])),
               cand_factor[e+1, c]> / sqrt(r)
        """
        previous_factors = candidate_factors[:, :-1]
        edge_inputs = F.silu(previous_factors + hidden_factors[:, 1:].unsqueeze(2))
        queries = self.state_query(edge_inputs)
        corrections = torch.einsum(
            "blpr,blcr->blpc",
            queries.float(),
            candidate_factors[:, 1:].float(),
        ) / math.sqrt(self.state_rank)
        return unary_logits[:, 1:].unsqueeze(2) + corrections

    # -- finite-state map composition ---------------------------------------

    @staticmethod
    def _inclusive_candidate_function_scan(maps: torch.Tensor) -> torch.Tensor:
        """Compose per-edge K->K functions with log-depth (Hillis-Steele)."""
        prefix = maps
        offset = 1
        while offset < maps.shape[1]:
            previous = prefix
            prefix = previous.clone()
            prefix[:, offset:] = previous[:, offset:].gather(-1, previous[:, :-offset])
            offset *= 2
        return prefix

    def _candidate_indices_from_maps(
        self, maps: torch.Tensor, initial_indices: torch.Tensor
    ) -> torch.Tensor:
        prefix_maps = self._inclusive_candidate_function_scan(maps)
        suffix_indices = prefix_maps.gather(
            -1, initial_indices.view(-1, 1, 1).expand(-1, maps.shape[1], -1)
        )[:, :, 0]
        return torch.cat((initial_indices.unsqueeze(1), suffix_indices), dim=1)

    # -- decoders ------------------------------------------------------------

    def decode_local(
        self, *, candidate_ids: torch.Tensor, transition_scores: torch.Tensor
    ) -> torch.Tensor:
        """Greedy per-edge argmax + prefix-scan compose (selector.decode)."""
        local_maps = transition_scores.argmax(dim=-1)
        initial_indices = torch.zeros(
            candidate_ids.shape[0], dtype=torch.long, device=candidate_ids.device
        )
        path_indices = self._candidate_indices_from_maps(local_maps, initial_indices)
        return candidate_ids.gather(-1, path_indices.unsqueeze(-1))[:, :, 0]

    def decode_normalized_map(
        self,
        *,
        candidate_ids: torch.Tensor,
        transition_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Global Viterbi MAP over the locally normalized Markov chain.

        Position 0 pinned one-hot to slot 0 (DFlash top-1).
        """
        batch_size, edges, top_k, _ = transition_scores.shape
        initial = transition_scores.new_full((batch_size, top_k), -torch.inf)
        initial[:, 0] = 0.0
        initial = torch.log_softmax(initial.float(), dim=-1)
        transitions = torch.log_softmax(transition_scores.float(), dim=-1)

        value = initial
        backpointers = []
        for edge in range(edges):
            candidates = value.unsqueeze(-1) + transitions[:, edge]  # [B, K(prev), K]
            value, predecessor = candidates.max(dim=1)
            backpointers.append(predecessor)
        current = value.argmax(dim=-1)
        reversed_states = [current]
        for predecessor in reversed(backpointers):
            current = predecessor.gather(-1, current.unsqueeze(-1))[:, 0]
            reversed_states.append(current)
        path_indices = torch.stack(list(reversed(reversed_states)), dim=1)
        return candidate_ids.gather(-1, path_indices.unsqueeze(-1))[:, :, 0]

    # -- T=1 lossless sampling ----------------------------------------------

    @torch.no_grad()
    def sample_temperature_one(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        transition_scores: torch.Tensor,
        uniforms: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ancestral sample one path.

        Returns (tokens [B, L], log_q [B, L], q_rows [B, L, K]) where q_rows[b, i]
        is the exact categorical distribution over the L-th position's K candidates
        along the sampled path (position 0 = unary softmax; i>0 = the predecessor-
        selected transition row). q_rows is needed for lossless residual sampling.

        Position 0 ~ softmax(unary[:, 0]); position i ~ softmax over the
        predecessor-selected row of the local K x K transition. One independent
        uniform per block position (inverse-CDF).
        """
        top_k = self.top_k
        initial_probs = torch.softmax(unary_logits[:, 0].float(), dim=-1)
        initial_cdf = initial_probs.cumsum(dim=-1)
        initial_indices = (
            uniforms[:, :1].ge(initial_cdf).sum(dim=-1).clamp_max(top_k - 1)
        )

        transition_probs = torch.softmax(transition_scores.float(), dim=-1)
        transition_cdf = transition_probs.cumsum(dim=-1)
        local_maps = (
            uniforms[:, 1:, None, None].ge(transition_cdf).sum(dim=-1).clamp_max(top_k - 1)
        )
        path_indices = self._candidate_indices_from_maps(local_maps, initial_indices)
        tokens = candidate_ids.gather(-1, path_indices.unsqueeze(-1))[:, :, 0]

        initial_selected = initial_probs.gather(-1, path_indices[:, :1])
        predecessor_indices = path_indices[:, :-1]
        successor_indices = path_indices[:, 1:]
        realized_rows = transition_probs.gather(
            2,
            predecessor_indices[:, :, None, None].expand(-1, -1, 1, top_k),
        )[:, :, 0]
        suffix_selected = realized_rows.gather(-1, successor_indices.unsqueeze(-1))[
            :, :, 0
        ]
        selected_probs = torch.cat((initial_selected, suffix_selected), dim=1)
        log_q = selected_probs.clamp_min(torch.finfo(torch.float32).tiny).log()
        # Per-position categorical q over the K candidates along the path.
        q_rows = torch.cat((initial_probs.unsqueeze(1), realized_rows), dim=1)
        return tokens, log_q, q_rows


def _parse_selector_config(config) -> dict:
    dflash_config = getattr(config, "dflash_config", None) or {}
    if not isinstance(dflash_config, dict):
        # HF may wrap nested config dicts; fall back to attribute access.
        dflash_config = dict(getattr(dflash_config, "__dict__", {}))
    rank = int(dflash_config.get("candidate_selector_rank", 0))
    top_k = int(dflash_config.get("candidate_selector_top_k", 0))
    if rank <= 0 or top_k <= 0:
        raise ValueError(
            "DFlash selector draft requires candidate_selector_rank>0 and "
            f"candidate_selector_top_k>0 in dflash_config; got rank={rank}, top_k={top_k}."
        )
    if not bool(dflash_config.get("candidate_selector_parallel_scan", False)):
        raise ValueError(
            "This selector port only supports candidate_selector_parallel_scan=True."
        )
    if not bool(dflash_config.get("candidate_selector_direct_edge", False)):
        raise ValueError(
            "This selector port only supports candidate_selector_direct_edge=True."
        )
    return {"state_rank": rank, "top_k": top_k}


class Qwen3DFlashSelectorModel(DFlashDraftModel):
    """DFlash backbone + candidate selector. Reuses the DFLASH speculative worker."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        selector_cfg = _parse_selector_config(config)
        self.candidate_selector = CandidateSelector(
            hidden_size=int(config.hidden_size),
            state_rank=selector_cfg["state_rank"],
            top_k=selector_cfg["top_k"],
            block_size=self.block_size,
            rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )
        # The target lm_head is attached at load time (embeddings are passed per call).
        self.lm_head: Optional[nn.Module] = None

    def attach_shared_modules(self, *, lm_head: nn.Module) -> None:
        self.lm_head = lm_head

    def compute_base_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Full (org-vocab-cropped) base logits from draft hidden via target lm_head."""
        if self.lm_head is None:
            raise ValueError(
                "DFlash selector requires the target lm_head "
                "(call attach_shared_modules first)."
            )
        weight = self.lm_head.weight
        if hidden.dtype != weight.dtype:
            hidden = hidden.to(weight.dtype)
        local_logits = torch.matmul(hidden, weight.T)
        return gather_and_crop_vocab(local_logits, self.lm_head)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        selector_weights = []
        backbone_weights = []
        for name, loaded_weight in weights:
            if name.startswith("embed_tokens.") or name.startswith("lm_head."):
                continue
            if name.startswith("candidate_selector."):
                selector_weights.append((name, loaded_weight))
            else:
                backbone_weights.append((name, loaded_weight))

        super().load_weights(backbone_weights)

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in selector_weights:
            if name not in params_dict:
                raise ValueError(
                    f"DFlash selector unexpected weight {name!r} not in model params."
                )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None:
                weight_loader(param, loaded_weight)
            else:
                param.data.copy_(loaded_weight)


EntryClass = [Qwen3DFlashSelectorModel]
