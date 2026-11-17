# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Restore BFL E4M3 weights with TorchAO's existing static FP8 tensor.

Separate converter views follow the pinned Dynamo NVFP4 helper. FP8 has scalar
weight/input scales, not NVFP4 block scales or packed nibbles. Diffusers handles
architecture mapping and TorchAO handles activation casting and scaled GEMM.
"""

import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def _restore_fp8_weight(qdata, scale, act_quant_scale):
    """Wrap saved tensors with the calibrated Klein FP8 execution policy."""
    from torchao.float8.inference import Float8MMConfig
    from torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor import (
        PrototypeFloat8Tensor,
    )
    from torchao.quantization.granularity import PerTensor
    from torchao.quantization.quantize_.common import KernelPreference
    from torchao.quantization.quantize_.workflows import QuantizeTensorToFloat8Kwargs

    mm_config = Float8MMConfig(emulate=False, use_fast_accum=False, pad_inner_dim=False)
    return PrototypeFloat8Tensor(
        qdata,
        scale,
        act_quant_scale=act_quant_scale,
        block_size=list(qdata.shape),
        mm_config=mm_config,
        act_quant_kwargs=QuantizeTensorToFloat8Kwargs(
            float8_dtype=torch.float8_e4m3fn,
            granularity=PerTensor(),
            mm_config=mm_config,
            kernel_preference=KernelPreference.TORCH,
        ),
        kernel_preference=KernelPreference.TORCH,
        dtype=torch.bfloat16,
    )


# Keep TorchAO's Python tensor construction outside Dynamo tracing. The graph
# contains only this operator and plain tensor arguments; TorchAO still owns
# activation quantization and the native GEMM. This boundary prevents Inductor
# from fusing inside the linear operation and requires registration at PT2 load.
@torch.library.custom_op(
    "dynamo_flux::fp8_linear", mutates_args=(), device_types="cuda"
)
def _fp8_linear(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    act_quant_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.nn.functional.linear(
        x, _restore_fp8_weight(qdata, scale, act_quant_scale)
    )


@_fp8_linear.register_fake
def _fp8_linear_fake(x, qdata, scale, act_quant_scale):
    return x.new_empty((x.shape[0], qdata.shape[0]))


class FP8Linear(torch.nn.Module):
    """Flatten inputs so TorchAO's saved per-tensor scale has rank two.

    The checkpoint activation scale is [1, 1]. TorchAO requires its rank to
    match the input during static casting. GEMM stays entirely in TorchAO.
    """

    def __init__(self, weight, in_features, out_features):
        super().__init__()
        if weight.act_quant_scale is None:
            raise ValueError("FP8Linear requires the saved activation scale")
        # Plain parameters survive source-module offload while an exported
        # program retains references. Never register the tensor subclass itself.
        self.qdata = torch.nn.Parameter(weight.qdata, requires_grad=False)
        self.scale = torch.nn.Parameter(weight.scale, requires_grad=False)
        self.act_quant_scale = torch.nn.Parameter(
            weight.act_quant_scale, requires_grad=False
        )
        restored_weight = self.weight
        if (
            type(weight) is not type(restored_weight)
            or weight.dtype != torch.bfloat16
            or weight.__tensor_flatten__() != restored_weight.__tensor_flatten__()
            or weight.shape != (out_features, in_features)
        ):
            raise ValueError(
                "FP8Linear requires the calibrated Klein FP8 tensor configuration"
            )
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        shape = x.shape
        output = _fp8_linear(
            x.reshape(-1, shape[-1]), self.qdata, self.scale, self.act_quant_scale
        )
        return output.reshape(*shape[:-1], self.out_features)

    @property
    def weight(self):
        """Reconstruct the native weight for inspection, outside compiled forward."""
        return _restore_fp8_weight(self.qdata, self.scale, self.act_quant_scale)


def load_fp8_state(checkpoint):
    """Map saved weights and scalar scales without dequantizing the weights."""
    from diffusers.loaders.single_file_utils import (
        convert_flux2_transformer_checkpoint_to_diffusers,
    )

    with safe_open(checkpoint, framework="pt") as stream:
        metadata = stream.metadata() or {}
    settings = json.loads(metadata.get("_quantization_metadata", "{}"))
    layers = settings.get("layers", {})
    if settings.get("format_version") != "1.0" or not layers:
        raise ValueError(
            "Klein FP8 requires quantization metadata format 1.0 with declared layers"
        )
    if any(layer.get("format") != "float8_e4m3fn" for layer in layers.values()):
        raise ValueError(
            "Klein FP8 requires exclusively float8_e4m3fn checkpoint layers"
        )
    raw = load_file(checkpoint)
    scale_views = {}
    for suffix in ("weight_scale", "input_scale"):
        sources = {
            key.removesuffix(f".{suffix}") for key in raw if key.endswith(f".{suffix}")
        }
        if sources != layers.keys():
            raise ValueError(
                f"FP8 metadata/{suffix} mismatch: missing={sorted(layers.keys() - sources)}, "
                f"unexpected={sorted(sources - layers.keys())}"
            )
        view = {}
        for source in sources:
            scale = raw.pop(f"{source}.{suffix}")
            if scale.dtype != torch.float32 or scale.numel() != 1:
                raise ValueError(f"{source}.{suffix}: expected a float32 scalar")
            if not torch.isfinite(scale).all().item() or scale.item() <= 0:
                raise ValueError(
                    f"{source}.{suffix}: expected a finite, positive scale"
                )
            # The converter chunks fused QKV along dim 0. Duplicate scalar
            # scales for those three outputs; all copies have the saved value.
            view[f"{source}.weight"] = scale.reshape(1).repeat(3)
        scale_views[suffix] = convert_flux2_transformer_checkpoint_to_diffusers(view)
    fp8_sources = {
        key.removesuffix(".weight")
        for key, value in raw.items()
        if value.dtype == torch.float8_e4m3fn
    }
    if fp8_sources != layers.keys():
        raise ValueError(
            "Declared Klein FP8 layers do not match the stored E4M3 weight tensors"
        )
    dense = convert_flux2_transformer_checkpoint_to_diffusers(raw)
    weight_scales = scale_views["weight_scale"]
    input_scales = scale_views["input_scale"]
    if not weight_scales or weight_scales.keys() != input_scales.keys():
        raise ValueError(
            "Converted FP8 weight and activation scale targets do not match"
        )
    quantized = {}
    for key, scale in weight_scales.items():
        weight = dense.pop(key)
        if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
            raise ValueError(f"{key}: expected a rank-two E4M3 weight")
        quantized[key.removesuffix(".weight")] = {
            "weight": weight,
            "weight_scale": scale.reshape(-1)[0].reshape(1, 1).clone(),
            "input_scale": input_scales[key].reshape(-1)[0].reshape(1, 1).clone(),
        }
    return dense, quantized


def build_fp8_transformer(
    checkpoint, config_repo, subfolder="transformer", device="cuda"
):
    """Wrap checkpoint tensors directly; never invoke TorchAO weight PTQ."""
    from diffusers import Flux2Transformer2DModel

    dense, quantized = load_fp8_state(checkpoint)
    config = Flux2Transformer2DModel.load_config(config_repo, subfolder=subfolder)
    with torch.device("meta"):
        model = Flux2Transformer2DModel.from_config(config)
    model = model.to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(dense, strict=False, assign=True)
    missing = sorted(set(missing) - {f"{path}.weight" for path in quantized})
    if missing or unexpected:
        raise ValueError(
            f"Klein FP8 checkpoint/config mismatch: missing={missing}, unexpected={unexpected}"
        )
    for path, parts in quantized.items():
        layer = model.get_submodule(path)
        qdata = parts["weight"].to(device).contiguous()
        if (
            not isinstance(layer, torch.nn.Linear)
            or layer.bias is not None
            or tuple(layer.weight.shape) != qdata.shape
        ):
            raise ValueError(
                f"{path}: checkpoint requires a matching bias-free linear layer"
            )
        # BFL and TorchAO use multiplicative dequantization scales: w = q * s.
        # Pass both saved scales unchanged. No .from_hp(), quantize_(), scale
        # recalculation or dense weight reconstruction is involved.
        weight = _restore_fp8_weight(
            qdata,
            parts["weight_scale"].to(device),
            parts["input_scale"].to(device),
        )
        model.set_submodule(
            path, FP8Linear(weight, layer.in_features, layer.out_features)
        )
    model.eval()
    return model
