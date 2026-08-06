# Adapted from the DFlash reference implementation (HF) but implemented with
# SGLang primitives (RadixAttention + SGLang KV cache). This model intentionally
# does not include token embeddings or an LM head; DFlash uses the target model's
# embedding/lm_head.

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
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
from sglang.srt.utils import is_npu
from sglang.srt.utils.hf_transformers_utils import get_rope_config

_is_npu = is_npu()
if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope
logger = logging.getLogger(__name__)

try:  # radix top-k: ~2x torch.topk on a 150k vocab, one kernel, graph-capturable
    from flashinfer import top_k as _flashinfer_top_k
except Exception:  # pragma: no cover - falls back to torch.topk
    _flashinfer_top_k = None


def _radix_topk(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k over the last dim of a 2D [N, vocab] tensor, sorted descending so slot 0
    is the top-1. flashinfer radix kernel when available, else torch.topk."""
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
        blocks = hidden_states.view(
            -1, self.block_size, self.num_groups, self.group_size
        )
        delta = delta.reshape(-1, self.block_size, self.taps, self.num_groups, 1)
        base = self.base_kernel[side].view(
            1, 1, self.taps, self.num_groups, self.group_size
        )
        coefficients = base + delta
        out = coefficients[:, :, 0] * blocks
        for tap in range(1, self.taps):
            shifted = F.pad(blocks[:, :-tap], (0, 0, 0, 0, tap, 0))
            out = out + coefficients[:, :, tap] * shifted
        return out.view_as(hidden_states)

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
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.state_rank = int(state_rank)
        self.top_k = int(top_k)
        self.block_size = int(block_size)
        self.rms_norm_eps = float(rms_norm_eps)
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
        """Grow the prefix-scan ping-pong buffers (cap-doubling, no-op if big enough) so
        the scan never allocates in-graph. Pre-called before capture; also a lazy grow.
        """
        edges = max(int(num_edges), 1)
        cur = self._scan_a
        if cur is not None and cur.shape[0] >= int(max_bs) and cur.shape[1] >= edges:
            return
        bs_cap = max(int(max_bs), cur.shape[0] * 2 if cur is not None else 0)
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
        predecessor_table = self.predecessor_token_table
        keys = self.successor_token_table[candidate_ids]
        hidden = self.hidden_projection(hidden_states)
        candidates = predecessor_table[candidate_ids]
        anchor = predecessor_table[anchor_token_ids]
        predecessors = torch.cat(
            [anchor[:, None, None].expand(-1, 1, self.top_k, -1), candidates[:, :-1]],
            dim=1,
        )
        return unary_logits[:, :, None] + torch.einsum(
            "blpr,blcr->blpc",
            predecessors * hidden[:, :, None],
            keys,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ancestral sample one path (inverse-CDF, one uniform per position; softmax scaled
        by the per-request temperature so q matches the target). Returns (tokens, q_rows),
        q_rows the per-position categorical over the K candidates for the verify."""
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
            rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
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
