from __future__ import annotations

import logging
from numbers import Integral
from typing import Any, List, Optional, Tuple

import heapq
import torch

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"
logger = logging.getLogger(__name__)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    """Select target layer indices used to build DFlash context features.

    Mirrors the upstream DFlash helper in `docs/dflash/model/utils.py`, but keeps the
    logic local to SGLang.

    Args:
        num_target_layers: Number of transformer layers in the runtime target model.
        num_draft_layers: Number of layers in the DFlash draft model.

    Returns:
        A list of 0-based target layer indices of length `num_draft_layers`.

    Notes:
        - DFlash uses hidden states after each selected target layer (HF-style).
        - SGLang captures "before layer i", so the model hook will typically add +1
          when mapping to capture points.
    """
    if num_target_layers <= 0:
        raise ValueError(
            f"num_target_layers must be positive, got {num_target_layers}."
        )
    if num_draft_layers <= 0:
        raise ValueError(f"num_draft_layers must be positive, got {num_draft_layers}.")

    if num_draft_layers == 1:
        return [num_target_layers // 2]

    start = 1
    end = num_target_layers - 3
    if end < start:
        raise ValueError(
            "DFlash layer selection requires num_target_layers >= 4. "
            f"Got num_target_layers={num_target_layers}."
        )

    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def get_dflash_config(config: Any) -> dict:
    if isinstance(config, dict):
        cfg = config.get("dflash_config", None)
    else:
        cfg = getattr(config, "dflash_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg

    try:
        return dict(cfg)
    except Exception:
        return {}


def resolve_dflash_block_size(
    *,
    draft_hf_config: Any,
    default: Optional[int] = None,
) -> Optional[int]:
    """Resolve DFLASH block size from draft config.

    Precedence:
      1) `dflash_config.block_size`
      2) top-level `block_size`
      3) `default`
    """
    dflash_cfg = get_dflash_config(draft_hf_config)
    dflash_block_size = dflash_cfg.get("block_size", None)
    if isinstance(draft_hf_config, dict):
        top_level_block_size = draft_hf_config.get("block_size", None)
    else:
        top_level_block_size = getattr(draft_hf_config, "block_size", None)

    parsed_dflash_block_size = None
    if dflash_block_size is not None:
        try:
            parsed_dflash_block_size = int(dflash_block_size)
        except Exception as e:
            raise ValueError(
                f"Invalid DFLASH dflash_config.block_size={dflash_block_size!r}."
            ) from e

    parsed_top_level_block_size = None
    if top_level_block_size is not None:
        try:
            parsed_top_level_block_size = int(top_level_block_size)
        except Exception as e:
            raise ValueError(
                f"Invalid DFLASH block_size={top_level_block_size!r}."
            ) from e

    if (
        parsed_dflash_block_size is not None
        and parsed_top_level_block_size is not None
        and parsed_dflash_block_size != parsed_top_level_block_size
    ):
        logger.warning(
            "DFLASH draft config has both block_size=%s and dflash_config.block_size=%s; using dflash_config.block_size.",
            top_level_block_size,
            dflash_block_size,
        )

    block_size = (
        parsed_dflash_block_size
        if parsed_dflash_block_size is not None
        else parsed_top_level_block_size
    )
    if block_size is None:
        return default

    if block_size <= 0:
        raise ValueError(f"DFLASH block_size must be positive, got {block_size}.")
    return block_size


def resolve_dflash_target_layer_ids(
    *,
    draft_hf_config: Any,
    target_num_layers: int,
    draft_num_layers: int,
) -> List[int]:
    """Resolve target layer ids used to build DFlash context features.

    Precedence:
      1) `draft_hf_config.dflash_config.target_layer_ids`
      2) `draft_hf_config.target_layer_ids` (fallback to base config)
      3) default `build_target_layer_ids(target_num_layers, draft_num_layers)`

    Notes:
        The number of draft transformer layers is *not* fundamentally tied to the number
        of target-layer features (K) used as DFlash context. We treat
        `len(target_layer_ids)` as K when explicitly provided. For backward compatibility
        (and for current released checkpoints), the default still uses K == draft_num_layers.
    """
    cfg = get_dflash_config(draft_hf_config)
    layer_ids = cfg.get("target_layer_ids", None)
    if layer_ids is None:
        layer_ids = getattr(draft_hf_config, "target_layer_ids", None)
    if layer_ids is None:
        return build_target_layer_ids(target_num_layers, draft_num_layers)

    if not isinstance(layer_ids, (list, tuple)):
        raise ValueError(
            "DFLASH dflash_config.target_layer_ids must be a list of ints, "
            f"got type={type(layer_ids).__name__}."
        )

    resolved: List[int] = [int(x) for x in layer_ids]
    if len(resolved) <= 0:
        raise ValueError(
            "DFLASH dflash_config.target_layer_ids must be non-empty. "
            f"Got len(target_layer_ids)={len(resolved)}."
        )

    for idx, val in enumerate(resolved):
        if val < 0 or val >= int(target_num_layers):
            raise ValueError(
                "DFLASH target_layer_ids contains an out-of-range layer id. "
                f"target_layer_ids[{idx}]={val}, target_num_layers={int(target_num_layers)}."
            )
    return resolved


def resolve_dflash_mask_token(*, draft_hf_config: Any) -> str:
    cfg = get_dflash_config(draft_hf_config)
    mask_token = cfg.get("mask_token", None)
    if mask_token is None:
        return DEFAULT_DFLASH_MASK_TOKEN
    if not isinstance(mask_token, str) or not mask_token:
        raise ValueError(
            "DFLASH dflash_config.mask_token must be a non-empty string, "
            f"got {mask_token!r}."
        )
    return mask_token


def resolve_dflash_mask_token_id(*, draft_hf_config: Any) -> Optional[int]:
    cfg = get_dflash_config(draft_hf_config)
    mask_token_id = cfg.get("mask_token_id", None)
    if mask_token_id is None:
        return None
    if not isinstance(mask_token_id, Integral) or isinstance(mask_token_id, bool):
        raise ValueError(
            "DFLASH dflash_config.mask_token_id must be an integer, "
            f"got {mask_token_id!r} (type={type(mask_token_id).__name__})."
        )
    mask_token_id = int(mask_token_id)
    if mask_token_id < 0:
        raise ValueError(
            "DFLASH dflash_config.mask_token_id must be non-negative, "
            f"got {mask_token_id}."
        )
    return mask_token_id


def can_dflash_slice_qkv_weight(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether DFlash can slice KV weights from a fused QKV linear layer."""
    quant_method = getattr(qkv_proj, "quant_method", None)
    if not isinstance(quant_method, UnquantizedLinearMethod):
        return (
            False,
            "quantized qkv_proj is not supported for this path "
            f"(quant_method={type(quant_method).__name__})",
        )
    if not hasattr(qkv_proj, "weight"):
        return False, "qkv weight tensor is missing"
    return True, ""


def can_dflash_use_fused_qkv_proj(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether a QKV layer is eligible for DFlash fused KV materialization."""
    eligible, reason = can_dflash_slice_qkv_weight(qkv_proj)
    if not eligible:
        return False, reason
    if getattr(qkv_proj, "bias", None) is not None:
        return False, "qkv bias is not supported for fused KV path"
    return True, ""


def compute_dflash_accept_len_and_bonus(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens (greedy verify rule).

    Args:
        candidates: Token ids proposed by the DFlash draft, including the current token.
            Shape: [bs, block_size]. candidates[:, 0] is the current token.
        target_predict: Token ids predicted by the target model for each position in the block.
            Shape: [bs, block_size]. target_predict[:, t] corresponds to argmax at position t.

    Returns:
        accept_len: int32 tensor [bs], number of accepted *draft* tokens (excluding current token and bonus token).
        bonus: int64 tensor [bs], the target-predicted token at index accept_len (the "bonus" token to append).

    Notes:
        Matches the reference implementation rule:
          accept while candidates[:, 1:] == target_predict[:, :-1] consecutively.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )

    bs, block_size = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    matches = candidates[:, 1:] == target_predict[:, :-1]
    accept_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    bonus = target_predict[torch.arange(bs, device=target_predict.device), accept_len]
    return accept_len, bonus.to(torch.int64)

def compute_dflash_tree_accept_len_and_bonus(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    parent_list: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens (tree verify rule).

    Args:
        candidates: Token ids proposed by the DFlash draft, including the current token.
            Shape: [bs, block_size]. candidates[:, 0] is the current token.
        target_predict: Token ids predicted by the target model for each position in the block.
            Shape: [bs, block_size]. target_predict[:, t] corresponds to argmax at position t.
        parent_list: Parent indices for each token, shape [bs, block_size]. parent_list[b, i] is the parent of the i-th token in request b.

    Returns:
        accept_len: int32 tensor [bs], number of accepted *draft* tokens (excluding current token and bonus token).
        bonus: int64 tensor [bs], the target-predicted token for the last accepted node (the "bonus" token to append).
        path_tokens: int64 tensor [bs, block_size - 1], tokens along the longest accepted path (excluding root),
            padded with -1 beyond accept_len.
        path_indices: int64 tensor [bs, block_size], node indices along the longest accepted path (including root),
            padded with -1 beyond accept_len.

    Notes:
        For each request, we walk all paths from the root and accept the longest path
        where each node's candidate matches the target prediction of its parent.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )
    if parent_list.shape != candidates.shape:
        raise ValueError(
            "parent_list must have the same shape as candidates. "
            f"parent_list.shape={tuple(parent_list.shape)}"
        )

    bs, block_size = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    device = candidates.device
    accepted = torch.zeros((bs, block_size), dtype=torch.bool, device=device)
    accepted[:, 0] = True

    safe_parents = parent_list.clamp(min=0)
    parent_targets = target_predict.gather(1, safe_parents)
    accepted[:, 1:] = candidates[:, 1:] == parent_targets[:, 1:]

    depth_accept = torch.full((bs, block_size), -1, dtype=torch.int32, device=device)
    depth_accept[:, 0] = 0

    for idx in range(1, block_size):
        parent = parent_list[:, idx]
        safe_parent = parent.clamp(min=0)
        parent_depth = depth_accept.gather(1, safe_parent.unsqueeze(1)).squeeze(1)
        depth_accept[:, idx] = torch.where(
            accepted[:, idx] & (parent >= 0) & (parent_depth >= 0),
            parent_depth + 1,
            torch.tensor(-1, dtype=torch.int32, device=device),
        )

    accept_len, best_idx = depth_accept.max(dim=1)
    bonus = target_predict[torch.arange(bs, device=device), best_idx]

    path_indices = torch.full((bs, block_size), -1, dtype=torch.long, device=device)
    current = best_idx.to(torch.long)
    accept_len_long = accept_len.to(torch.long)

    for step in range(block_size):
        position = accept_len_long - step
        active = position >= 0
        if not active.any():
            break
        batch_idx = torch.arange(bs, device=device)
        path_indices[batch_idx[active], position[active]] = current[active]
        current_safe = current.clamp(min=0)
        next_parent = parent_list.gather(1, current_safe.unsqueeze(1)).squeeze(1)
        current = torch.where(active, next_parent, current)

    selected_indices = path_indices[:, 1:].clamp(min=0)
    gathered_tokens = candidates.gather(1, selected_indices)
    positions = torch.arange(1, block_size, device=device).unsqueeze(0)
    path_mask = positions <= accept_len_long.unsqueeze(1)
    path_tokens = torch.where(path_mask, gathered_tokens, torch.full_like(gathered_tokens, -1))

    return (
        accept_len,
        bonus.to(torch.int64),
        path_tokens.to(torch.int64),
        path_indices.to(torch.int64),
    )


def build_tree_verify_tokens(
    *,
    verified_id: torch.Tensor,
    draft_logits: torch.Tensor,
    topk: int,
    num_draft_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build tree verify tokens for DFlash eagle-style tree verify (TARGET_VERIFY mode).

    This function constructs a pruned tree of draft tokens and returns:
      - The pruned tokens (including the root verified_id)
      - A *full* topk-tree parent list compatible with the EAGLE kernel
      - A selected_index list that encodes the pruned tree inside the full tree

    Args:
        verified_id: Current token per request, shape [bs]
        draft_logits: Draft model logits, shape [bs, num_steps, vocab_size]
        topk: Number of top candidates to select at each node
        num_draft_tokens: Total number of tokens in the pruned tree (including root)

    Returns:
        draft_tokens: Flattened pruned tokens, shape [bs * num_draft_tokens]
        parent_list: Full topk-tree parent list, shape [bs, topk * (depth - 1) + 1]
        selected_index: Selected indices encoding the pruned tree, shape [bs, num_draft_tokens - 1]
        tree_mask: Optional tree mask buffer (placeholder for compatibility)
    """
    bs = draft_logits.shape[0]
    device = draft_logits.device

    # Compute softmax probabilities
    draft_probs = torch.softmax(draft_logits, dim=-1)  # [bs, num_steps, vocab]

    # Select top-k tokens at each position
    topk_probs, topk_ids = torch.topk(draft_probs, k=topk, dim=-1)  # [bs, num_steps, topk]

    num_pos = topk_probs.shape[1]  # Number of positions (num_steps)
    depth = num_draft_tokens - 1  # diffusion: single step, use full verify length

    full_tree_nodes = topk * (depth - 1) + 1

    if topk == 1:
        # Linear chain: only one candidate per level in the full tree.
        tokens = torch.cat([verified_id[:, None], topk_ids[:, :, 0]], dim=1)
        if num_draft_tokens < tokens.shape[1]:
            tokens = tokens[:, :num_draft_tokens]
        elif num_draft_tokens > tokens.shape[1]:
            pad_count = num_draft_tokens - tokens.shape[1]
            pad_tokens = tokens[:, -1:].expand(bs, pad_count)
            tokens = torch.cat([tokens, pad_tokens], dim=1)

        # Full parent list is just a chain of length depth (offset by 1 for root).
        parent = (
            torch.arange(full_tree_nodes, device=device)
            .unsqueeze(0)
            .expand(bs, -1)
            - 1
        )
        selected_index = torch.arange(0, num_draft_tokens, device=device).unsqueeze(0).expand(bs, -1)
        draft_tokens = tokens[:, :num_draft_tokens].flatten()
        return draft_tokens, parent, selected_index, topk_probs

    # Build the full topk tree parent list (shared across batch).
    parent_list_full = torch.full(
        (bs, full_tree_nodes), -1, dtype=torch.long, device=device
    )
    for layer in range(1, depth):
        layer_base = 1 + (layer - 1) * topk
        if layer == 1:
            parent_list_full[:, layer_base : layer_base + topk] = 0
        else:
            parent_list_full[:, layer_base : layer_base + topk] = 1 + (layer - 2) * topk

    # Build tree using heap-based approach (similar to dflash_transformers.py)
    draft_tokens_list = []
    selected_index_list = []

    for b in range(bs):
        # Heap: (-joint_prob, path, prob, parent_idx, node_idx)
        heap: list[tuple[float, list[int], float, int, int]] = []

        # Initialize with position 0's top-k candidates
        for r in range(topk):
            p = float(topk_probs[b, 0, r].item())
            token_id = int(topk_ids[b, 0, r].item())
            node_idx = 1 + r
            heapq.heappush(heap, (-p, [token_id], p, -1, node_idx))

        selected_nodes: list[tuple[int, int]] = []  # (node_idx, parent_selected_idx)
        selected_tokens: list[int] = []
        i = 0

        # Only limit by total number of tokens (matching dflash_transformers.py logic)
        while heap and i < num_draft_tokens - 1:
            _neg_prob, path, prob, parent_idx, node_idx = heapq.heappop(heap)
            i += 1
            selected_nodes.append((node_idx, parent_idx))
            selected_tokens.append(path[-1])

            # Expand to next position
            pos = len(path) - 1
            next_pos = pos + 1

            # Only expand if there are more positions available
            if next_pos < num_pos:
                layer_base = 1 + next_pos * topk
                for r in range(topk):
                    next_p = float(topk_probs[b, next_pos, r].item())
                    next_token = int(topk_ids[b, next_pos, r].item())
                    joint_prob = prob * next_p
                    child_node_idx = layer_base + r
                    heapq.heappush(
                        heap,
                        (
                            -joint_prob,
                            path + [next_token],
                            joint_prob,
                            len(selected_nodes) - 1,
                            child_node_idx,
                        ),
                    )

        # Build pruned tokens and selected_index in original tree index space
        tokens = [int(verified_id[b].item())]
        selected_index = []

        for (node_idx, parent_idx), token in zip(selected_nodes, selected_tokens, strict=True):
            tokens.append(int(token))
            selected_index.append(int(node_idx))

        # Pad pruned tokens / selected_index if needed
        while len(tokens) < num_draft_tokens:
            tokens.append(tokens[-1])
        while len(selected_index) < num_draft_tokens - 1:
            selected_index.append(selected_index[-1] if selected_index else 1)

        # Ensure selected_index is within full_tree_nodes bounds
        if any(idx <= 0 or idx >= full_tree_nodes for idx in selected_index):
            raise RuntimeError(
                f"Invalid selected_index for full tree: min={min(selected_index)}, max={max(selected_index)}, "
                f"full_tree_nodes={full_tree_nodes}."
            )

        draft_tokens_list.append(torch.tensor(tokens, dtype=torch.long, device=device))
        selected_index_list.append(torch.tensor(selected_index, dtype=torch.long, device=device))

    draft_tokens = torch.stack(draft_tokens_list, dim=0).flatten()
    selected_index = torch.stack(selected_index_list, dim=0)

    return draft_tokens, parent_list_full, selected_index, topk_probs
