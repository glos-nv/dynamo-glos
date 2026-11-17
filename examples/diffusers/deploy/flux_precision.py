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


@torch.library.custom_op(
    "dynamo_flux::layer_norm", mutates_args=(), device_types="cuda"
)
def _layer_norm(
    x: torch.Tensor,
    shape: list[int],
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, shape, weight, bias, eps)


@_layer_norm.register_fake
def _layer_norm_fake(x, shape, weight, bias, eps):
    return x.new_empty(x.shape)


@torch.library.custom_op("dynamo_flux::rms_norm", mutates_args=(), device_types="cuda")
def _rms_norm(
    x: torch.Tensor, shape: list[int], weight: torch.Tensor | None, eps: float | None
) -> torch.Tensor:
    return torch.nn.functional.rms_norm(x, shape, weight, eps)


@_rms_norm.register_fake
def _rms_norm_fake(x, shape, weight, eps):
    return x.new_empty(x.shape)


class _NativeNorm(torch.nn.Module):
    """Keep native normalization arithmetic at the quantization boundary.

    Compiler reduction/decomposition differences can change FP8/FP4 rounding
    decisions. Preserve the original CUDA normalization, including its weights,
    while allowing compilation of the surrounding transformer graph.
    """

    def __init__(self, norm):
        super().__init__()
        self.weight = norm.weight
        self.is_rms = isinstance(norm, torch.nn.RMSNorm)
        self.bias = None if self.is_rms else norm.bias
        self.shape = list(norm.normalized_shape)
        self.eps = norm.eps

    def forward(self, x):
        if self.is_rms:
            return _rms_norm(x, self.shape, self.weight, self.eps)
        return _layer_norm(x, self.shape, self.weight, self.bias, self.eps)


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


def prepare_transformer(transformer, precision: str) -> None:
    """Configure the loaded FLUX transformer in place before tuning or serving.

    FP16 converts weights and installs hooks to keep the BF16 pipeline boundary.
    The worker loads BF16, FP8 and NVFP4 weights directly. Quantized transformers
    retain native normalization kernels to preserve their calibration inputs
    when compiled. The text encoder and VAE are not modified.

    Call once after moving the pipeline to CUDA and checking hardware support.
    Precision is independent of backend selection, so eager fallback preserves
    these changes.
    """
    if precision == "fp16":
        transformer.to(dtype=torch.float16)
        # The text encoder, scheduler, and VAE keep their BF16 interface. These
        # hooks also surround the tuned backend and the eager fallback.
        transformer.register_forward_pre_hook(_fp16_inputs, with_kwargs=True)
        transformer.register_forward_hook(_bf16_output)
    elif precision in ("fp8", "fp4", "nvfp4"):
        for name, module in list(transformer.named_modules()):
            if isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
                transformer.set_submodule(name, _NativeNorm(module))
    logger.info("Transformer precision: %s", precision)
