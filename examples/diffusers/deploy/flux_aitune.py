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
"""Select and validate a FLUX transformer backend during worker startup.

AI Tune optimizes inference execution; it does not train or fine-tune weights.
Import this module only when tuning is enabled, since AI Tune is optional.
"""

import logging

import aitune.torch as ait
import numpy as np
import torch
from aitune.torch.backend import (
    BuildMode,
    TorchEagerBackend,
    TorchInductorAotBackend,
    TorchInductorJitBackend,
    TorchTensorRTAotBackend,
    TorchTensorRTAotBackendConfig,
    TorchTensorRTConfig,
    TorchTensorRTJitBackend,
    TorchTensorRTJitBackendConfig,
)
from aitune.torch.module.wrapper_module import ModuleState
from aitune.torch.task.profiling import ProfilingConfig

logger = logging.getLogger(__name__)
MAX_PIXEL_RMSE = 0.05


def _contiguous_inputs(_module, args, kwargs):
    """Give exported graphs consistent tensor strides for generation and editing."""
    # Generation uses strided latents; editing concatenates contiguous latents.
    # AoT artifacts assume the strides seen during export. Normalize outside the
    # exported graph, both while recording and when calling the tuned module.
    args = tuple(
        value.contiguous() if torch.is_tensor(value) else value for value in args
    )
    kwargs = {
        name: value.contiguous() if torch.is_tensor(value) else value
        for name, value in kwargs.items()
    }
    return args, kwargs


def tune_transformer(
    pipe,
    *,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    max_image_side: int,
    size_multiple: int,
) -> None:
    """Replace ``pipe.transformer`` in place with a validated AI Tune module.

    The FLUX.2-klein pipeline must already be on CUDA with its selected precision
    applied. Record generation and reference-image editing, then compare
    TorchInductor and Torch-TensorRT in AoT and JIT modes at batch size one.
    Prefer the fastest successful compiled candidate, even if eager is faster.
    If every candidate fails, log an error and retain eager execution at the
    selected precision. Both compiled and eager selections undergo image checks.

    Args:
        pipe: Loaded ``Flux2KleinPipeline``; its other components remain unchanged.
        width: Startup workload image width in pixels.
        height: Startup workload image height in pixels.
        steps: Denoising steps used for recording and validation.
        guidance_scale: Guidance scale used for recording and validation.
        max_image_side: Worker's maximum supported image side in pixels.
        size_multiple: Pixels per packed latent token along each image axis
            (16 for FLUX.2-klein), used to derive dynamic token-count bounds.

    Raises:
        RuntimeError: No backend was selected, tuning did not complete, or the
            selected backend's images fail validation against the eager baseline.
            These failures stop startup before model registration.
    """
    logger.info("Optimizing the loaded transformer with AoT and JIT backends")

    def generate(prompt, *, image=None, seed=42):
        return pipe(
            prompt=prompt,
            image=image,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]

    generation = {"prompt": "A red fox in a sunny alpine meadow"}
    reference = generate(**generation)
    editing = {
        "prompt": "Turn this image into a watercolor painting",
        "image": reference,
    }
    held_out = {"prompt": "A blue ceramic teapot on a wooden table", "seed": 123}
    # Pair each request with its eager output before replacing the transformer.
    validation_cases = [
        (generation, reference),
        (editing, generate(**editing)),
        (held_out, generate(**held_out)),
    ]

    trt_config = TorchTensorRTConfig(
        require_full_compilation=True,
        # FLUX computes rotary frequencies in FP64; TRT needs FP32.
        truncate_double=True,
    )
    candidates = [
        TorchInductorAotBackend(),
        TorchTensorRTAotBackend(
            TorchTensorRTAotBackendConfig(compile_config=trt_config)
        ),
        TorchInductorJitBackend(),
        TorchTensorRTJitBackend(
            TorchTensorRTJitBackendConfig(compile_config=trt_config, fullgraph=True)
        ),
    ]
    # Diagnostic mode prefers a successful compiled candidate, even if slower, but
    # restores the eager baseline when every candidate fails.
    strategy = ait.MinLatencyStrategy(
        backends=candidates,
        profiling_config=ProfilingConfig(batch_sizes=[1], batching=False),
    ).enable_performance_validation("diagnostic")
    # One packed token per 16x16 pixels; editing concatenates image and reference.
    # Explicit bounds let AoT artifacts serve the worker's existing size range.
    image_tokens = ait.DynamicDim(
        "image_tokens",
        min=1,
        opt=(width // size_multiple) * (height // size_multiple),
        max=2 * (max_image_side // size_multiple) ** 2,
    )
    # Keep all denoising samples, including both generation and editing shapes.
    ait.config.max_num_samples_stored = 2 * steps
    pipe.transformer.register_forward_pre_hook(_contiguous_inputs, with_kwargs=True)
    config = pipe.transformer.config
    pipe.transformer = ait.Module(
        pipe.transformer,
        name="transformer",
        strategy=strategy,
        dynamic_shapes={
            "hidden_states": (1, image_tokens, config.in_channels),
            "img_ids": (1, image_tokens, len(config.axes_dims_rope)),
        },
    )

    def record_workload():
        generate(**generation)
        generate(**editing)

    ait.tune(record_workload, [{}], batch_sizes=[1], ignore_failing_modules=False)
    backends = _validate_backend_selection(pipe.transformer, strategy, candidates)
    for request, expected in validation_cases:
        _validate_image(generate(**request), expected)
    logger.info("Serving transformer: %s", [backend.describe() for backend in backends])


def _validate_backend_selection(transformer, strategy, candidates):
    """Return selected graph backends, logging candidate failures for eager graphs.

    An eager selection is a supported fallback. An unfinished tuning state,
    empty selection, or unexpected backend mode is a startup error.
    """
    if transformer.state != ModuleState.TUNED:
        raise RuntimeError("AI Tune did not produce a tuned transformer")
    backends = list(transformer.module.backends.values())
    if not backends:
        raise RuntimeError("AI Tune did not select a transformer backend")
    eager_count = 0
    for backend in backends:
        if isinstance(backend, TorchEagerBackend):
            eager_count += 1
        elif backend.build_mode not in (
            BuildMode.AHEAD_OF_TIME,
            BuildMode.JUST_IN_TIME,
        ):
            raise RuntimeError("AI Tune selected an unexpected backend")
    if eager_count:
        error_logs = [
            str(transformer.cache_dir / graph.name / candidate.key() / "build.log")
            for graph in transformer.graph_specs
            for candidate in candidates
        ]
        logger.error(
            "Tuning failed: no AoT or JIT candidate succeeded for %d transformer graph(s). "
            "Falling back to eager execution with the selected precision unchanged. "
            "Candidate results: %s. Error logs: %s",
            eager_count,
            strategy.backend_results,
            error_logs,
        )
    return backends


def _validate_image(actual, expected) -> None:
    """Compare an output image with the same-seed, same-precision eager image.

    Pixel RMSE is measured after scaling 8-bit pixel values to [0, 1]. Raise
    ``RuntimeError`` for a shape mismatch, nonfinite output, or RMSE above
    ``MAX_PIXEL_RMSE``. This is a startup regression check, not a quality score
    against the original BF16 model.
    """
    actual = np.asarray(actual, dtype=np.float32) / 255.0
    expected = np.asarray(expected, dtype=np.float32) / 255.0
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise RuntimeError("Invalid output from the selected transformer backend")
    rmse = float(np.sqrt(np.mean((actual - expected) ** 2)))
    if rmse > MAX_PIXEL_RMSE:
        raise RuntimeError(f"Selected backend differs from baseline: RMSE={rmse}")
    logger.info("Selected backend output validated: pixel RMSE=%.6f", rmse)
