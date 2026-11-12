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
"""Transformer precision setup, shared by eager and tuned FLUX serving."""

import logging

import torch
from torch.utils._pytree import tree_map

logger = logging.getLogger(__name__)
PRECISIONS = ("bf16", "fp16", "fp8", "fp4", "nvfp4")


def check_precision_support(precision: str) -> None:
    """Check FP8/NVFP4 support on the current CUDA device before loading weights.

    ``precision`` must be one of ``PRECISIONS`` (validated by the worker CLI).
    Raise ``ValueError`` for FP8 below compute capability 8.9 or NVFP4 below
    10.0. ``fp4`` and ``nvfp4`` are aliases. BF16 and FP16 need no additional
    quantization check here; the pipeline still requires BF16-capable hardware.
    """
    if precision in ("fp4", "nvfp4"):
        if torch.cuda.get_device_capability()[0] < 10:
            raise ValueError(
                "FP4/NVFP4 requires a Blackwell GPU (compute capability 10.0+)"
            )
    elif precision == "fp8" and torch.cuda.get_device_capability() < (8, 9):
        raise ValueError("FP8 requires an Ada or newer GPU (compute capability 8.9+)")


def _cast_floating(value, dtype):
    """Cast floating tensors in a nested value, preserving integer tensors."""
    return tree_map(
        lambda tensor: (
            tensor.to(dtype=dtype)
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
            else tensor
        ),
        value,
    )


def _fp16_inputs(_module, args, kwargs):
    return _cast_floating(args, torch.float16), _cast_floating(kwargs, torch.float16)


def _bf16_output(_module, _args, output):
    return _cast_floating(output, torch.bfloat16)


def _fp8_linear(module, name: str) -> bool:
    """Select large FLUX linear layers, keeping embeddings and output in BF16."""
    return (
        isinstance(module, torch.nn.Linear)
        and "embed" not in name
        and name not in ("norm_out.linear", "proj_out")
        and module.in_features >= 1024
        and module.out_features >= 1024
    )


def prepare_transformer(transformer, precision: str) -> None:
    """Configure the loaded FLUX transformer in place before tuning or serving.

    FP16 converts weights and installs hooks to keep the BF16 pipeline boundary.
    FP8 quantizes eligible linear layers with TorchAO, leaving the remaining
    layers in BF16. BF16 and NVFP4 need no conversion here: the worker loads
    their weights directly. The text encoder and VAE are not modified.

    Call once after moving the pipeline to CUDA and checking hardware support.
    Precision is independent of backend selection, so eager fallback preserves
    these changes. Raise ``ValueError`` if FP8 has no eligible linear layers.
    """
    if precision == "fp16":
        transformer.to(dtype=torch.float16)
        # The text encoder, scheduler, and VAE keep their BF16 interface. These
        # hooks also surround the tuned backend and the eager fallback.
        transformer.register_forward_pre_hook(_fp16_inputs, with_kwargs=True)
        transformer.register_forward_hook(_bf16_output)
    elif precision == "fp8":
        # TorchAO is optional for BF16/FP16 serving without AI Tune.
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            quantize_,
        )
        from torchao.quantization.granularity import PerTensor
        from torchao.utils import unwrap_tensor_subclass

        count = sum(
            _fp8_linear(module, name) for name, module in transformer.named_modules()
        )
        if not count:
            raise ValueError("No transformer linear layers are eligible for FP8")
        quantize_(
            transformer,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
            filter_fn=_fp8_linear,
        )
        # Store quantized weights as plain tensor components for AoT export and
        # CPU offload. The parametrization reconstructs FP8 weights for execution.
        unwrap_tensor_subclass(transformer)
        logger.info("Quantized %d transformer linear layers to FP8 E4M3", count)
    logger.info("Transformer precision: %s", precision)
