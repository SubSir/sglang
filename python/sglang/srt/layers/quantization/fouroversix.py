from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped


class FouroversixFp4Config(QuantizationConfig):
    """Config class for fouroversix NVFP4 quantization (dense-only).

    This integration quantizes activations to NVFP4 and uses fouroversix fp4_matmul
    for GEMM. We currently treat weights as BF16/FP16 loaded from checkpoint.

    Notes:
    - This implementation focuses on dense Linear layers only (no MoE).
    - Weight-only NVFP4 checkpoint formats are not supported here.
    """

    def __init__(
        self,
        ignored_layers: Optional[List[str]] = None,
        fp4_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.ignored_layers = ignored_layers or []
        self.fp4_config = fp4_config or {"scale_rule": "static_6"}

    @classmethod
    def get_name(cls) -> str:
        return "fouroversix"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # NVFP4 requires recent NVIDIA GPUs; keep conservative.
        return 100

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FouroversixFp4Config":
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert", "ignore"], None
        )
        fp4_cfg = cls.get_from_keys_or(config, ["fp4_config", "fouroversix"], None)
        return cls(ignored_layers=ignored_layers or [], fp4_config=fp4_cfg)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            return FouroversixFp4LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class FouroversixFp4LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: FouroversixFp4Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # Keep default (unquantized) weight storage; compute path will quantize activations.
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        from sglang.srt.layers.parameter import ModelWeightParameter

        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=params_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        try:
            from fouroversix import QuantizationConfig as F6QuantizationConfig
            from fouroversix import quantize_to_fp4
        except ImportError as err:
            raise ImportError(
                "The package `fouroversix` is required for --quantization fouroversix. "
                "Please install it with `pip install fouroversix`."
            ) from err

        cfg_kwargs = dict(self.quant_config.fp4_config or {})
        qcfg = F6QuantizationConfig(**cfg_kwargs)

        w_q = quantize_to_fp4(layer.weight.data, qcfg)

        layer.register_buffer("weight_q_values", w_q.values)
        layer.register_buffer("weight_q_scale_factors", w_q.scale_factors)
        layer.register_buffer("weight_q_amax", w_q.amax)

        layer._weight_q_dtype = w_q.dtype
        layer._weight_q_original_shape = w_q.original_shape
        layer._weight_q_scale_rule = w_q.scale_rule
        layer._weight_q_padded_shape = w_q.padded_shape

        del layer.weight

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        try:
            from fouroversix import QuantizationConfig as F6QuantizationConfig
            from fouroversix import fp4_matmul, quantize_to_fp4
            from fouroversix.quantize.quantized_tensor import QuantizedTensor
        except ImportError as err:
            raise ImportError(
                "The package `fouroversix` is required for --quantization fouroversix. "
                "Please install it with `pip install fouroversix`."
            ) from err

        cfg_kwargs = dict(self.quant_config.fp4_config or {})
        qcfg = F6QuantizationConfig(**cfg_kwargs)

        x_q = quantize_to_fp4(x, qcfg)

        w_q = QuantizedTensor(
            values=layer.weight_q_values,
            scale_factors=layer.weight_q_scale_factors,
            amax=layer.weight_q_amax,
            dtype=layer._weight_q_dtype,
            original_shape=layer._weight_q_original_shape,
            scale_rule=layer._weight_q_scale_rule,
            padded_shape=layer._weight_q_padded_shape,
        )

        out = fp4_matmul(x_q, w_q)

        if bias is not None:
            out = out + bias
        return out
