# Adapted from the DFlash reference implementation (HF) but implemented with
# SGLang primitives (RadixAttention + SGLang KV cache). This model intentionally
# does not include token embeddings or an LM head; DFlash uses the target model's
# embedding/lm_head.

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from sglang.srt.configs.laguna import normalize_gating
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_utils import (
    can_dflash_slice_qkv_weight,
    get_dflash_attention_sliding_window_size,
    get_dflash_layer_types,
    parse_dflash_draft_config,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_npu
from sglang.srt.utils.common import get_compiler_backend
from sglang.srt.utils.hf_transformers_utils import get_rope_config

_is_npu = is_npu()
if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope
logger = logging.getLogger(__name__)

try:  # pragma: no cover - the fallback when neither this nor flashinfer applies
    from flashinfer import top_k as _flashinfer_top_k
except Exception:
    _flashinfer_top_k = None


# Threshold-pruned, int32-packed-key deterministic top-k. One bandwidth pass packs
# each tile's BF16 values into order-preserving int32 keys and reduces to a tile
# maximum; the 16th largest tile maximum is a rigorous lower bound on the global
# top-k, so the ~133 tiles of 149 below it are dropped without a second read. Ties
# resolve to the lowest index, deterministically.
#
# On H200 over a 151936 vocab this is 1.3-2.5x flashinfer's radix kernel, and the
# gap widens with batch: flashinfer's is structure-bound rather than bandwidth-bound
# (its achieved bandwidth is the same on H200 and B200), so it does not scale.


_TOPK_K = 16  # the kernel's staging is sized for it
_TOPK_TILE = 1024
_TOPK_MAXES_PAD = 256  # power-of-two >= ceil(vocab / TILE)
_TOPK_CAND_PAD = 4096  # power-of-two >= ceil(vocab / TILE) * TOP_K
_TOPK_SENTINEL = tl.constexpr(-(2**31))


@triton.jit
def _pack_keys(values, column_rank):
    bits = values.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    monotonic = tl.where(bits >= 0x8000, (~bits) & 0xFFFF, bits | 0x8000)
    return ((monotonic - 32768) << 16) | column_rank


@triton.jit
def _tile_max_kernel(
    logits_ptr,
    tile_max_ptr,
    vocab,
    num_tiles,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = (pid // num_tiles).to(tl.int64)
    tile = pid % num_tiles
    offsets = tl.arange(0, TILE_SIZE)
    column = tile * TILE_SIZE + offsets
    values = tl.load(
        logits_ptr + row * vocab + column,
        mask=column < vocab,
        other=float("-inf"),
    )
    keys = _pack_keys(values, (TILE_SIZE - 1) - offsets)
    keys = tl.where(column < vocab, keys, _TOPK_SENTINEL)
    tl.store(tile_max_ptr + row * num_tiles + tile, tl.max(keys, axis=0))


@triton.jit
def _threshold_kernel(
    tile_max_ptr,
    threshold_ptr,
    num_tiles,
    PAD: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, PAD)
    keys = tl.load(
        tile_max_ptr + row * num_tiles + offsets,
        mask=offsets < num_tiles,
        other=_TOPK_SENTINEL,
    )
    threshold = tl.max(keys, axis=0)
    for _ in tl.static_range(K - 1):
        keys = tl.where(keys == threshold, _TOPK_SENTINEL, keys)
        threshold = tl.max(keys, axis=0)
    tl.store(threshold_ptr + row, threshold)


@triton.jit
def _live_tile_topk_kernel(
    logits_ptr,
    tile_max_ptr,
    threshold_ptr,
    cand_ptr,
    vocab,
    num_tiles,
    TILE_SIZE: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(0)
    row = (pid // num_tiles).to(tl.int64)
    tile = pid % num_tiles
    staging = cand_ptr + row * (num_tiles * K) + tile * K
    slot = tl.arange(0, K)
    tile_max = tl.load(tile_max_ptr + row * num_tiles + tile)
    threshold = tl.load(threshold_ptr + row)
    if tile_max < threshold:
        tl.store(staging + slot, tl.full([K], _TOPK_SENTINEL, tl.int32))
        return
    offsets = tl.arange(0, TILE_SIZE)
    column = tile * TILE_SIZE + offsets
    values = tl.load(
        logits_ptr + row * vocab + column,
        mask=column < vocab,
        other=float("-inf"),
    )
    keys = _pack_keys(values, (TILE_SIZE - 1) - offsets)
    keys = tl.where(column < vocab, keys, _TOPK_SENTINEL)
    for j in tl.static_range(K):
        best = tl.max(keys, axis=0)
        tl.store(staging + j, best)
        keys = tl.where(keys == best, _TOPK_SENTINEL, keys)


@triton.jit
def _final_topk_kernel(
    logits_ptr,
    cand_ptr,
    out_values_ptr,
    out_indices_ptr,
    vocab,
    num_candidates,
    TILE_SIZE: tl.constexpr,
    PAD: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, PAD)
    keys = tl.load(
        cand_ptr + row * num_candidates + offsets,
        mask=offsets < num_candidates,
        other=_TOPK_SENTINEL,
    )
    for j in tl.static_range(K):
        best = tl.max(keys, axis=0)
        best_slot = tl.argmax(keys, axis=0)
        tile = best_slot // K
        rank = best & (TILE_SIZE - 1)
        index = tile.to(tl.int64) * TILE_SIZE + ((TILE_SIZE - 1) - rank)
        value = tl.load(logits_ptr + row * vocab + index)
        tl.store(out_values_ptr + row * K + j, value)
        tl.store(out_indices_ptr + row * K + j, index)
        keys = tl.where(offsets == best_slot, _TOPK_SENTINEL, keys)


def _triton_topk(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact top-16 (descending values, int64 indices) of a 2D CUDA BF16 tensor."""

    if logits.dim() != 2 or not logits.is_contiguous() or not logits.is_cuda:
        raise ValueError("triton_topk16 requires a contiguous 2D CUDA tensor")
    if logits.dtype != torch.bfloat16:
        raise ValueError("triton_topk16 packs BF16 bit patterns")
    rows, vocab = logits.shape
    num_tiles = (vocab + _TOPK_TILE - 1) // _TOPK_TILE
    num_candidates = num_tiles * top_k
    if num_tiles > _TOPK_MAXES_PAD or num_candidates > _TOPK_CAND_PAD:
        raise ValueError(f"vocab {vocab} exceeds staged capacity")
    device = logits.device
    tile_max = torch.empty((rows, num_tiles), dtype=torch.int32, device=device)
    threshold = torch.empty((rows,), dtype=torch.int32, device=device)
    candidates = torch.empty((rows, num_candidates), dtype=torch.int32, device=device)
    out_values = torch.empty((rows, top_k), dtype=logits.dtype, device=device)
    out_indices = torch.empty((rows, top_k), dtype=torch.int64, device=device)
    _tile_max_kernel[(rows * num_tiles,)](
        logits, tile_max, vocab, num_tiles, TILE_SIZE=_TOPK_TILE, num_warps=4
    )
    _threshold_kernel[(rows,)](
        tile_max, threshold, num_tiles, PAD=_TOPK_MAXES_PAD, K=top_k, num_warps=1
    )
    _live_tile_topk_kernel[(rows * num_tiles,)](
        logits,
        tile_max,
        threshold,
        candidates,
        vocab,
        num_tiles,
        TILE_SIZE=_TOPK_TILE,
        K=top_k,
        num_warps=4,
    )
    _final_topk_kernel[(rows,)](
        logits,
        candidates,
        out_values,
        out_indices,
        vocab,
        num_candidates,
        TILE_SIZE=_TOPK_TILE,
        PAD=_TOPK_CAND_PAD,
        K=top_k,
        num_warps=4,
    )
    return out_values, out_indices


def _radix_topk(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k over the last dim of a 2D [N, vocab] tensor, sorted descending so slot 0
    is the top-1. The packed-key Triton kernel where its preconditions hold, else
    flashinfer's radix kernel, else torch.topk."""
    if (
        envs.SGLANG_DFLASH_PACKED_TOPK.get()
        and k == _TOPK_K
        and scores.dtype == torch.bfloat16
        and scores.dim() == 2
        and scores.is_contiguous()
        and scores.shape[1] <= _TOPK_MAXES_PAD * _TOPK_TILE
    ):
        return _triton_topk(scores, k)
    if _flashinfer_top_k is not None:
        return _flashinfer_top_k(scores, k, sorted=True, deterministic=True)
    return torch.topk(scores, k, dim=-1)


def _get_dflash_layer_attention_params(
    config, layer_id: int
) -> Tuple[int, AttentionType]:
    layer_types = get_dflash_layer_types(config)
    if layer_types is None:
        return -1, AttentionType.ENCODER_ONLY
    if layer_id >= len(layer_types):
        raise ValueError(
            "DFLASH config.layer_types must contain one entry per draft layer. "
            f"Got {len(layer_types)} entries, layer_id={layer_id}."
        )

    layer_type = layer_types[layer_id]
    if layer_type == "full_attention":
        text_config = getattr(config, "text_config", None) or config
        attention_type = (
            AttentionType.DECODER
            if getattr(text_config, "is_causal", False)
            else AttentionType.ENCODER_ONLY
        )
        return -1, attention_type
    if layer_type == "sliding_attention":
        sliding_window_size = get_dflash_attention_sliding_window_size(config)
        assert sliding_window_size is not None
        return sliding_window_size, AttentionType.DECODER
    raise ValueError(
        "Unsupported DFLASH draft layer type. "
        f"layer_types[{layer_id}]={layer_type!r}."
    )


class DFlashAttention(nn.Module):
    def __init__(self, config, layer_id: int, quant_config=None) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        tp_size = int(get_parallel().tp_size)
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(
            getattr(config, "num_key_value_heads", total_num_heads)
        )
        head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))

        self.hidden_size = hidden_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        assert self.total_num_heads % tp_size == 0, (
            f"DFlashAttention requires total_num_heads divisible by tp_size. "
            f"total_num_heads={self.total_num_heads}, tp_size={tp_size}."
        )
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0, (
                f"DFlashAttention requires total_num_kv_heads divisible by tp_size when >= tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        else:
            assert tp_size % self.total_num_kv_heads == 0, (
                f"DFlashAttention requires tp_size divisible by total_num_kv_heads when total_num_kv_heads < tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix="qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix="o_proj",
        )

        # Per-head Q/K RMSNorm, matching HF Qwen3.
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        rope_theta, rope_scaling = get_rope_config(config)
        rope_is_neox_style = bool(
            getattr(
                config, "rope_is_neox_style", getattr(config, "is_neox_style", True)
            )
        )
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )

        self.scaling = head_dim**-0.5
        self.sliding_window_size, self.attn_type = _get_dflash_layer_attention_params(
            config, layer_id
        )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window_size,
            attn_type=self.attn_type,
        )

    def forward_prepare_npu(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn.layer_id == 0:
            self.rotary_emb.get_cos_sin_with_position(positions)
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            eps=self.q_norm.variance_epsilon,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            q_bias=getattr(self.q_norm, "bias", None),
            k_bias=getattr(self.k_norm, "bias", None),
        )
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if _is_npu:
            q, k, v = self.forward_prepare_npu(positions, hidden_states)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        attn_output = self.apply_attention_output(attn_output, hidden_states)
        output, _ = self.o_proj(attn_output)
        return output

    def apply_attention_output(
        self, attn_output: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return attn_output

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project hidden_states to K/V only (skip Q).

        This is used by DFlash to materialize ctx tokens into the draft KV cache:
        we only need K/V for the cached tokens; Q is never consumed.
        """
        # Fast path for unquantized weights: slice the fused QKV weight and run one GEMM.
        can_slice_qkv_weight, _ = can_dflash_slice_qkv_weight(self.qkv_proj)
        if can_slice_qkv_weight:
            kv_slice = slice(self.q_size, self.q_size + 2 * self.kv_size)
            weight = self.qkv_proj.weight[kv_slice]
            bias = (
                self.qkv_proj.bias[kv_slice] if self.qkv_proj.bias is not None else None
            )
            kv = F.linear(hidden_states, weight, bias)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            return k, v

        # Fallback: compute full QKV and discard Q (keeps compatibility with quantized weights).
        qkv, _ = self.qkv_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return k_by_head.view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Match K shape so RoPE kernel head-count check passes on all backends.
        dummy_q = k.new_empty(k.shape)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class DFlashMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(getattr(config, "intermediate_size", 0))
        if intermediate_size <= 0:
            raise ValueError(
                f"Invalid intermediate_size={intermediate_size} for DFlash MLP."
            )

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix="gate_up_proj" if not prefix else f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix="down_proj" if not prefix else f"{prefix}.down_proj",
        )
        hidden_act = getattr(config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported DFlash activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# Channel tile. 512 divides the 2560 and 4096 hidden sizes DFlash drafts use, and is
# a multiple of every plausible conv_group_size, so a tile never straddles a group.
_CONV_BLOCK_H = 512


@triton.jit
def _grouped_conv_kernel(
    x_ptr,
    delta_ptr,
    base_ptr,
    out_ptr,
    hidden_size,
    x_row_stride,
    delta_row_stride,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    TAPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """out[r, c] = sum_t (base[t, c] + delta[r, t, c // GROUP_SIZE]) * x[r - t, c]

    One program per (row, channel tile). The taps read backwards inside the row's
    DFlash block and are zero across its boundary, so a row never sees a position the
    draft has not proposed. Everything between the load of x and the store of out
    stays in registers -- the point of the kernel is that the coefficient sum, the
    shifted operand and the per-tap products never reach HBM.
    """
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < hidden_size
    position = row % BLOCK_ROWS  # position inside this row's block
    groups = offs // GROUP_SIZE

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for tap in tl.static_range(TAPS):
        # position >= tap keeps the tap inside the block; the clamp only keeps the
        # masked-off address in range.
        source = tl.maximum(row - tap, 0)
        x = tl.load(
            x_ptr + source * x_row_stride + offs,
            mask=mask & (position >= tap),
            other=0.0,
        ).to(tl.float32)
        base = tl.load(base_ptr + tap * hidden_size + offs, mask=mask, other=0.0)
        delta = tl.load(
            delta_ptr + row * delta_row_stride + tap * NUM_GROUPS + groups,
            mask=mask,
            other=0.0,
        )
        acc += (base.to(tl.float32) + delta.to(tl.float32)) * x
    tl.store(
        out_ptr + row * hidden_size + offs,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


class DFlashGroupedConv(nn.Module):
    """Grouped dynamic depthwise K-tap convolution across one DFlash block.

        row_i <- sum_t (base[t] + delta_i[t]) * row_{i-t}

    `base` is a static per-channel kernel and `delta` is predicted for the row by one
    projection, shared by every channel of a group: a hidden of H with group size g
    carries H/g coefficients per tap instead of H. The same projection produces the
    kernel for the sublayer's input and the one for its output, which is why `prepare`
    hands the second half to `finish` rather than projecting twice.

    Taps read backwards within the block and are zero across its boundary, so a row
    never sees a position the draft has not proposed yet.
    """

    def __init__(
        self, hidden_size: int, block_size: int, taps: int, group_size: int
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"DFLASH conv_group_size={group_size} must divide "
                f"hidden_size={hidden_size}."
            )
        self.hidden_size = int(hidden_size)
        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = self.hidden_size // self.group_size
        # [input/output, tap, channel], the layout training exports.
        self.base_kernel = nn.Parameter(torch.empty(2, self.taps, self.hidden_size))
        self.kernel_projection = nn.Linear(
            self.hidden_size, 2 * self.taps * self.num_groups, bias=False
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        # The kernel indexes both operands by hand; a layout it does not expect would
        # read the wrong elements and still return a plausible tensor.
        assert hidden_states.stride(-1) == 1 and hidden_states.is_contiguous()
        assert delta.stride(-1) == 1 and delta.stride(-2) == self.num_groups
        rows = hidden_states.numel() // self.hidden_size
        out = torch.empty_like(hidden_states)
        _grouped_conv_kernel[(rows, triton.cdiv(self.hidden_size, _CONV_BLOCK_H))](
            hidden_states,
            delta,
            self.base_kernel[side],
            out,
            self.hidden_size,
            hidden_states.stride(-2),
            delta.stride(-3),
            GROUP_SIZE=self.group_size,
            NUM_GROUPS=self.num_groups,
            BLOCK_ROWS=self.block_size,
            TAPS=self.taps,
            BLOCK_H=_CONV_BLOCK_H,
        )
        return out

    def prepare(self, hidden_states: torch.Tensor):
        """Convolve a sublayer's input; return it with the kernel for its output."""
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1], 2, self.taps, self.num_groups
        )
        return (
            self._convolve(hidden_states, coefficients[..., 0, :, :], side=0),
            coefficients[..., 1, :, :],
        )

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, side=1)


class DFlashDecoderLayer(nn.Module):
    attention_cls = DFlashAttention

    def __init__(self, config, layer_id: int, quant_config=None) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = self.attention_cls(
            config=config, layer_id=layer_id, quant_config=quant_config
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = DFlashMLP(config=config, quant_config=quant_config)

        # DFlash2 wraps each sublayer in a grouped convolution along the block. The
        # module names match what training exports, so no weight remapping is needed,
        # and a DFlash checkpoint leaves both None and takes the path it always took.
        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        self.attention_conv = None
        self.mlp_conv = None
        if draft_config.conv_type == "grouped_dynamic_depthwise":
            block_size = draft_config.resolve_block_size(default=16)

            def grouped_conv():
                return DFlashGroupedConv(
                    hidden_size,
                    block_size,
                    draft_config.conv_kernel_size,
                    draft_config.conv_group_size,
                )

            self.attention_conv = grouped_conv()
            self.mlp_conv = grouped_conv()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            # Keep return types consistent for upstream callers.
            if residual is None:
                residual = hidden_states
            return hidden_states, residual

        # Pre-norm attention with fused residual+norm when possible (Qwen3-style).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attention_kernel = None
        if self.attention_conv is not None:
            hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)

        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        if attention_kernel is not None:
            attn_out = self.attention_conv.finish(attn_out, attention_kernel)

        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)

        mlp_kernel = None
        if self.mlp_conv is not None:
            hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if mlp_kernel is not None:
            hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        return hidden_states, residual


class DFlashDraftModel(nn.Module):
    """SGLang DFlash draft model (no embedding / lm_head weights).

    The checkpoint provides:
      - transformer weights for `layers.*`
      - `fc.weight`, `hidden_norm.weight` for projecting target context features
      - `norm.weight` for final normalization
    """

    decoder_layer_cls = DFlashDecoderLayer
    supports_fused_context_kv = True

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config

        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.layers = nn.ModuleList(
            [
                self.decoder_layer_cls(
                    config=config, layer_id=i, quant_config=quant_config
                )
                for i in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Project per-token target context features:
        # concat(K * hidden_size) -> hidden_size, where K is the number of target-layer
        # feature tensors concatenated per token (not necessarily equal to num_layers).
        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        target_num_layers = (
            int(draft_config.num_target_layers)
            if draft_config.num_target_layers is not None
            else num_layers
        )
        target_layer_ids = draft_config.resolve_target_layer_ids(
            target_num_layers=target_num_layers, draft_num_layers=num_layers
        )
        num_context_features = len(target_layer_ids)

        self.num_context_features = int(num_context_features)
        self.fc = nn.Linear(
            self.num_context_features * hidden_size, hidden_size, bias=False
        )
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.block_size = draft_config.resolve_block_size(default=16)

    def get_attention_sliding_window_size(self) -> Optional[int]:
        return get_dflash_attention_sliding_window_size(self.config)

    def prepare_context_hidden_for_kv(
        self, layer: DFlashDecoderLayer, ctx_hidden: torch.Tensor
    ) -> torch.Tensor:
        return ctx_hidden

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target-layer hidden states into draft hidden_size."""
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                f"but got shape={tuple(target_hidden.shape)}. "
                "This usually means the target model is capturing a different number of layer features than "
                "the draft checkpoint/config expects."
            )
        return self.hidden_norm(self.fc(target_hidden))

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            if hasattr(self, "forward_embed"):
                input_embeds = self.forward_embed(input_ids)
            else:
                raise ValueError(
                    "DFlashDraftModel requires `input_embeds` (use the target "
                    "embedding)."
                )
        hidden_states = input_embeds
        residual: Optional[torch.Tensor] = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if hidden_states.numel() != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped_name = name[len("model.") :]
                if stripped_name in params_dict:
                    return stripped_name
            else:
                prefixed_name = f"model.{name}"
                if prefixed_name in params_dict:
                    return prefixed_name
            return None

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                resolved_name = resolve_param_name(mapped_name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                resolved_name = resolve_param_name(name)
                if resolved_name is None:
                    # Ignore unexpected weights (e.g., HF rotary caches).
                    continue
                param = params_dict[resolved_name]
                if resolved_name.endswith("fc.weight") and tuple(
                    loaded_weight.shape
                ) != tuple(param.shape):
                    raise ValueError(
                        "DFLASH fc.weight shape mismatch. This usually means the draft checkpoint's "
                        "number of context features (K) does not match this config. "
                        f"Expected fc.weight.shape={tuple(param.shape)} "
                        f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                        f"but got {tuple(loaded_weight.shape)} for weight '{name}'."
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


class DFlashLagunaAttention(DFlashAttention):
    """Laguna DFlash attention with the trained Laguna softplus gate."""

    def __init__(self, config, layer_id: int, quant_config=None) -> None:
        super().__init__(config=config, layer_id=layer_id, quant_config=quant_config)
        hidden_size = int(config.hidden_size)
        total_num_heads = self.total_num_heads
        gating = normalize_gating(getattr(config, "gating", True))
        self.gating = gating
        self.gate_per_head = gating == "per-head"
        if self.gating == "disabled":
            self.g_proj = None
        else:
            g_out = (
                total_num_heads
                if self.gate_per_head
                else total_num_heads * self.head_dim
            )
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                g_out,
                bias=False,
                quant_config=quant_config,
                prefix="g_proj",
            )

    def apply_attention_output(
        self, attn_output: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.g_proj is None:
            return attn_output

        gate, _ = self.g_proj(hidden_states)
        gate = F.softplus(gate.float()).to(attn_output.dtype)
        if self.gate_per_head:
            attn_shape = attn_output.shape
            return (
                attn_output.view(*attn_shape[:-1], self.num_heads, self.head_dim)
                * gate.unsqueeze(-1)
            ).view(attn_shape)
        else:
            return attn_output * gate


class DFlashLagunaDecoderLayer(DFlashDecoderLayer):
    attention_cls = DFlashLagunaAttention


class DFlashLagunaForCausalLM(DFlashDraftModel):
    """Laguna DFlash draft model matching the exported Speculators checkpoint."""

    decoder_layer_cls = DFlashLagunaDecoderLayer
    supports_fused_context_kv = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        hidden_size = int(config.hidden_size)
        self.aux_hidden_norms = nn.ModuleList(
            [
                RMSNorm(hidden_size, eps=rms_norm_eps)
                for _ in range(self.num_context_features)
            ]
        )

    def prepare_context_hidden_for_kv(
        self, layer: DFlashLagunaDecoderLayer, ctx_hidden: torch.Tensor
    ) -> torch.Tensor:
        return layer.input_layernorm(ctx_hidden)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "Laguna DFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                f"but got shape={tuple(target_hidden.shape)}."
            )

        num_slices = int(self.num_context_features)
        slice_size = int(target_hidden.shape[-1]) // num_slices
        slices = target_hidden.view(target_hidden.shape[0], num_slices, slice_size)
        compute_dtype = self.fc.weight.dtype
        if slices.dtype != compute_dtype:
            slices = slices.to(compute_dtype)
        normed = torch.empty_like(slices)
        for i, norm in enumerate(self.aux_hidden_norms):
            normed[:, i, :] = norm(slices[:, i, :])
        fused = normed.reshape(target_hidden.shape[0], -1)
        return self.hidden_norm(self.fc(fused))


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def _score_edges(
    *,
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Two codebook gathers, a scale by the projected hidden, and the K x K contraction.

    Compiled because it is a dozen kernels on tensors small enough that the cost is
    the kernel count rather than the bytes: at batch 1 the whole edge-scoring step is
    ~0.02 ms of dispatch. dynamic=True keeps it to one compilation for every batch
    size, the way the rest of the tree compiles small elementwise chains.
    """
    keys = successor_table[candidate_ids]
    candidates = predecessor_table[candidate_ids]
    anchor = predecessor_table[anchor_token_ids]
    predecessors = torch.cat(
        [anchor[:, None, None].expand(-1, 1, top_k, -1), candidates[:, :-1]], dim=1
    )
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], keys
    )


class CandidateSelector(nn.Module):
    """Direct-edge parallel-scan candidate selector: two bias-free projections plus a
    [vocab, r] token table turn draft hidden + target-lm-head top-K candidates into a
    K x K transition lattice. Training ships the table folded, so this side only
    gathers rows; it is replicated, not vocab-sharded, since candidate_ids are global.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        state_rank: int,
        top_k: int,
        block_size: int,
    ) -> None:
        super().__init__()
        self.state_rank = int(state_rank)
        self.top_k = int(top_k)
        self.block_size = int(block_size)
        # An edge is scored directly in the two token directions:
        #
        #   edge(p -> c) = <A[p] * project(h), B[c]>
        #
        # A and B are separate [vocab, r] tables, so the predecessor and the successor
        # are untied; training folds the 1/sqrt(r) scale into B and ships both tables
        # materialized, so this side only gathers rows. They are replicated rather than
        # vocab-sharded because candidate ids are global.
        self.predecessor_token_table = nn.Parameter(
            torch.empty(int(vocab_size), self.state_rank), requires_grad=False
        )
        self.successor_token_table = nn.Parameter(
            torch.empty(int(vocab_size), self.state_rank), requires_grad=False
        )
        self.hidden_projection = nn.Linear(hidden_size, state_rank, bias=False)
        # Filled before cuda-graph capture (see the setup method below).
        self._scan_a: Optional[torch.Tensor] = None
        self._scan_b: Optional[torch.Tensor] = None

    def alloc_decode_buffers(self, max_bs: int, num_edges: int, device) -> None:
        """Grow the prefix-scan ping-pong buffers (no-op if already big enough) so the
        scan never allocates in-graph. Pre-called before capture; also a lazy grow.
        """
        edges = max(int(num_edges), 1)
        cur = self._scan_a
        if cur is not None and cur.shape[0] >= int(max_bs) and cur.shape[1] >= edges:
            return
        bs_cap = max(int(max_bs), cur.shape[0] if cur is not None else 0)
        edge_cap = max(edges, cur.shape[1] if cur is not None else 0)
        self._scan_a = torch.empty(
            (bs_cap, edge_cap, self.top_k), dtype=torch.long, device=device
        )
        self._scan_b = torch.empty_like(self._scan_a)

    def build_lattice(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """candidate_ids/unary_logits: [B, L, K] -> [B, L, previous_K, current_K]:

            score[b,e,p,c] = unary[b,e,c] + <A[pred[b,e,p]] * project(h[b,e]), B[c]>

        pred is cand[b,e-1]; slot 0's is the verified anchor, broadcast over p so it
        needs no code path of its own. The 1/sqrt(r) scale is folded into B by the
        training export, and the hidden is projected without a further rms_norm.
        """
        return _score_edges(
            predecessor_table=self.predecessor_token_table,
            successor_table=self.successor_token_table,
            candidate_ids=candidate_ids,
            unary_logits=unary_logits,
            hidden=self.hidden_projection(hidden_states),
            anchor_token_ids=anchor_token_ids,
            top_k=self.top_k,
        )

    def _candidate_indices_from_maps(self, maps, initial_indices) -> torch.Tensor:
        # Compose the per-edge K->K maps into prefixes with a log-depth (Hillis-Steele)
        # scan over two static ping-pong buffers, then read the path from initial_indices.
        bs, edges = int(maps.shape[0]), int(maps.shape[1])
        self.alloc_decode_buffers(bs, edges, maps.device)  # no-op if big enough
        src, dst = self._scan_a[:bs, :edges], self._scan_b[:bs, :edges]
        src.copy_(maps)
        offset = 1
        while offset < edges:
            dst.copy_(src)
            torch.gather(src[:, offset:], -1, src[:, :-offset], out=dst[:, offset:])
            src, dst = dst, src
            offset *= 2
        suffix_indices = src.gather(
            -1, initial_indices.view(-1, 1, 1).expand(-1, edges, -1)
        )[:, :, 0]
        return torch.cat((initial_indices.unsqueeze(1), suffix_indices), dim=1)

    def decode_local(
        self, *, candidate_ids: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """Greedy per-edge argmax + prefix-scan compose, walked from the slot the
        anchor edge scores highest."""
        path_indices = self._candidate_indices_from_maps(
            scores[:, 1:].argmax(dim=-1), scores[:, 0, 0].argmax(dim=-1)
        )
        return candidate_ids.gather(-1, path_indices.unsqueeze(-1))[:, :, 0]

    def sample_path(
        self,
        *,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        uniforms: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ancestral sample one path (inverse-CDF, one uniform per position; softmax scaled
        by the per-request temperature so q matches the target). Returns (tokens, q_rows),
        q_rows the per-position categorical over the K candidates for the verify.

        greedy_mask rows take the argmax instead, bit-identical to decode_local: the
        captured graph is shared by greedy and sampling batches, so the choice has to be
        a tensor select rather than a Python branch."""
        top_k = self.top_k
        temps = temperatures.view(-1, 1)
        initial_probs = torch.softmax(scores[:, 0, 0].float() / temps, dim=-1)
        initial_indices = (
            uniforms[:, :1]
            .ge(initial_probs.cumsum(dim=-1))
            .sum(dim=-1)
            .clamp_max(top_k - 1)
        )
        transition_probs = torch.softmax(
            scores[:, 1:].float() / temps[:, :, None, None], dim=-1
        )
        local_maps = (
            uniforms[:, 1:, None, None]
            .ge(transition_probs.cumsum(dim=-1))
            .sum(dim=-1)
            .clamp_max(top_k - 1)
        )
        if greedy_mask is not None:
            initial_indices = torch.where(
                greedy_mask, scores[:, 0, 0].argmax(dim=-1), initial_indices
            )
            local_maps = torch.where(
                greedy_mask[:, None, None], scores[:, 1:].argmax(dim=-1), local_maps
            )
        path_indices = self._candidate_indices_from_maps(local_maps, initial_indices)
        tokens = candidate_ids.gather(-1, path_indices.unsqueeze(-1))[:, :, 0]
        realized_rows = transition_probs.gather(
            2, path_indices[:, :-1, None, None].expand(-1, -1, 1, top_k)
        )[:, :, 0]
        q_rows = torch.cat((initial_probs.unsqueeze(1), realized_rows), dim=1)
        return tokens, q_rows


class Qwen3DFlashSelectorModel(DFlashDraftModel):
    """DFlash backbone + candidate selector. Reuses the DFLASH speculative worker."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        dflash_config = getattr(config, "dflash_config", None) or {}
        selector_config = dflash_config.get("dflashv2_selector") or {}
        rank = int(selector_config.get("rank", 0))
        top_k = int(selector_config.get("top_k", 0))
        parameterization = str(selector_config.get("parameterization", ""))
        if parameterization != "direct_ab":
            raise ValueError(
                "DFlash selector draft requires dflashv2_selector.parameterization "
                f"'direct_ab'; got {parameterization!r}."
            )
        if rank <= 0 or top_k <= 0:
            raise ValueError(
                "DFlash selector draft requires dflash_config.dflashv2_selector with "
                f"rank>0 and top_k>0; got rank={rank}, top_k={top_k}."
            )
        # The selector spans the *proposal* slots, not the block rows: the anchor holds
        # row 0 as context and proposes nothing. A checkpoint that disagrees is caught
        # by the slot-count check in the worker's `_propose_selector_block`.
        self.candidate_selector = CandidateSelector(
            hidden_size=int(config.hidden_size),
            vocab_size=int(config.vocab_size),
            state_rank=rank,
            top_k=top_k,
            block_size=int(
                dflash_config.get("proposal_block_size", self.block_size - 1)
            ),
        )
        # The target lm_head is attached at load time (embeddings passed per call).
        self.lm_head: Optional[nn.Module] = None

    def compute_candidates(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k base candidates via the target lm_head: hidden [N, H] -> global
        candidate_ids / unary_logits [N, K]. Under TP (vocab-sharded lm_head): local top-k
        per shard, all-gather K logits/ids (not the full vocab), then a global top-k --
        identical candidates at O(tp*K) instead of O(vocab) gather bandwidth."""
        if self.lm_head is None:
            raise ValueError(
                "DFlash selector requires the target lm_head to be set on the draft "
                "model before capture (draft_model.lm_head = target lm_head)."
            )
        k = self.candidate_selector.top_k
        weight = self.lm_head.weight
        hidden = hidden.to(weight.dtype)
        if get_tensor_model_parallel_world_size() == 1:
            org = int(self.lm_head.org_vocab_size)
            vals, ids = _radix_topk(torch.matmul(hidden, weight[:org].T), k)
            return ids.long(), vals.float()
        shard = self.lm_head.shard_indices
        vals, ids = _radix_topk(
            torch.matmul(hidden, weight[: int(shard.num_org_elements)].T), k
        )
        global_ids = ids.long() + int(shard.org_vocab_start_index)
        gathered_vals = tensor_model_parallel_all_gather(vals.float(), dim=-1)
        gathered_ids = tensor_model_parallel_all_gather(global_ids, dim=-1)
        top_vals, sel = torch.topk(gathered_vals, k, dim=-1)
        return torch.gather(gathered_ids, -1, sel).long(), top_vals.float()


EntryClass = [DFlashDraftModel, DFlashLagunaForCausalLM, Qwen3DFlashSelectorModel]
