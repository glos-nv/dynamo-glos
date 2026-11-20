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
"""Select and check a FLUX transformer backend in the offline tuning process.

AI Tune optimizes inference execution; it does not train or fine-tune weights.
Import this module only when tuning is enabled, since AI Tune is optional.
"""

import logging
import time
from unittest.mock import patch

import aitune.torch as ait
import numpy as np
import torch
from aitune.torch.backend import Backend, TorchEagerBackend
from aitune.torch.module.wrapper_module import ModuleState
from aitune.torch.task.profiling import ProfilingConfig
from skimage.metrics import structural_similarity

logger = logging.getLogger(__name__)


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
    report: dict,
    progress,
    output_dir,
    min_ssim: float,
    backend_id: str | None = None,
) -> None:
    """Tune an eager pipeline in place, recording build times and image agreement.

    The FLUX.2-klein pipeline must already be on CUDA with its selected precision
    applied. Record generation and reference-image editing, then evaluate
    AI Tune's default backend candidates at batch size one, without compiler
    overrides. Prefer the fastest successful candidate, even if eager is faster.
    Eager fallback is rejected: it cannot produce an optimized serving artifact.
    Image mismatches are recorded but do not prevent saving or benchmarking.
    ``backend_id`` optionally isolates one library-discovered default for the
    diagnostic matrix; normal tuning evaluates all defaults without overrides.
    ``report`` is updated in place and ``progress`` persists it after each stage.

    Args:
        pipe: Loaded ``Flux2KleinPipeline``; its other components remain unchanged.
        width: Tuning workload image width in pixels.
        height: Tuning workload image height in pixels.
        steps: Denoising steps used for recording and validation.
        guidance_scale: Guidance scale used for recording and validation.
        max_image_side: Worker's maximum supported image side in pixels.
        size_multiple: Pixels per packed latent token along each image axis
            (16 for FLUX.2-klein), used to derive dynamic token-count bounds.
        report: Mutable result record including build timings and image checks.
        progress: Callback that persists the current report.
        output_dir: Directory for eager and compiled validation images.
        min_ssim: Minimum RGB structural similarity required for image agreement.
        backend_id: Optional ID of an installed default candidate to isolate.

    Raises:
        RuntimeError: No optimized backend was selected or tuning did not complete.
    """
    logger.info("Optimizing the loaded transformer with AI Tune's default backends")
    report["image_validation"] = {
        "metric": "ssim",
        "threshold": min_ssim,
        "scores": {},
    }

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

    # Diagnostic mode prefers a successful candidate, even if slower, but
    # restores the eager baseline when every candidate fails.
    strategy = ait.MinLatencyStrategy(
        profiling_config=ProfilingConfig(batch_sizes=[1], batching=False),
    ).enable_performance_validation("diagnostic")
    if backend_id is not None:
        # AI Tune 0.6.0 exposes resolved defaults through this private property.
        candidates = strategy._backends
        index = int(backend_id) - 1
        if not 0 <= index < len(candidates):
            raise ValueError(f"Unknown default backend ID: {backend_id}")
        strategy._backends_override = [candidates[index]]
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

    original_build = Backend.build
    report["backend_builds"] = []

    def timed_build(backend, *args, **kwargs):
        if isinstance(backend, TorchEagerBackend):
            return original_build(backend, *args, **kwargs)
        entry = {"backend": backend.describe(), "success": False}
        started = time.perf_counter()
        try:
            result = original_build(backend, *args, **kwargs)
            entry["success"] = True
            return result
        finally:
            entry["seconds"] = time.perf_counter() - started
            report["backend_builds"].append(entry)
            report["compilation_seconds"] = sum(
                item["seconds"] for item in report["backend_builds"]
            )
            progress()

    with patch.object(Backend, "build", timed_build):
        ait.tune(record_workload, [{}], batch_sizes=[1], ignore_failing_modules=False)
    report["candidate_results"] = strategy.backend_results
    backends = _validate_backend_selection(pipe.transformer, strategy)
    report["compile"] = "PASS"
    clip_cases = []
    for label, (request, expected) in zip(
        ("generation", "editing", "held-out"), validation_cases
    ):
        report["phase"] = label if label != "held-out" else "validation"
        progress()
        actual = generate(**request)
        actual.save(output_dir / f"{label}.png")
        expected.save(output_dir / f"{label}-eager.png")
        clip_cases.append((label, request["prompt"], expected, actual))
        if label != "held-out":
            report[label] = "PASS"
        try:
            score = _image_ssim(actual, expected)
            report["image_validation"]["scores"][label] = score
            logger.info(
                "%s image agreement: SSIM=%.6f (minimum %.6f)", label, score, min_ssim
            )
            if score < min_ssim:
                raise RuntimeError(
                    f"Selected backend differs from baseline: SSIM={score:.6f} < {min_ssim:.6f}"
                )
        except RuntimeError as error:
            report.setdefault("validation_errors", []).append(
                {"case": label, "error": str(error)}
            )
            logger.warning(
                "%s image validation failed: %s; retaining the tuned artifact",
                label,
                error,
            )
        report["validation"] = "FAIL" if report.get("validation_errors") else "PASS"
        progress()
    # Run the auxiliary model only after generation and backend profiling finish.
    from flux_clip import score_image_pairs

    report["clip_score"] = score_image_pairs(clip_cases)
    progress()
    logger.info("Tuned transformer: %s", [backend.describe() for backend in backends])


def _validate_backend_selection(transformer, strategy):
    """Require an optimized backend for every graph before saving an artifact."""
    if transformer.state != ModuleState.TUNED:
        raise RuntimeError("AI Tune did not produce a tuned transformer")
    backends = list(transformer.module.backends.values())
    if not backends:
        raise RuntimeError("AI Tune did not select a transformer backend")
    eager_count = sum(isinstance(backend, TorchEagerBackend) for backend in backends)
    if eager_count:
        error_logs = sorted(
            str(path)
            for graph in transformer.graph_specs
            for path in (transformer.cache_dir / graph.name).glob("*/build.log")
        )
        logger.error(
            "Tuning failed: no optimization candidate succeeded for %d transformer graph(s). "
            "No optimized artifact will be published. "
            "Candidate results: %s. Error logs: %s",
            eager_count,
            strategy.backend_results,
            error_logs,
        )
        raise RuntimeError("No optimized artifact: AI Tune selected eager fallback")
    return backends


def _image_ssim(actual, expected) -> float:
    """Return mean RGB SSIM against the same-seed, same-precision eager image.

    Use an 11-pixel Gaussian window (sigma 1.5), population covariance, and
    explicit range 1 for normalized pixels. Average the three channel scores.
    This measures image agreement, not prompt adherence or overall quality.
    Invalid images raise RuntimeError before scoring.
    """
    actual = np.asarray(actual, dtype=np.float32) / 255.0
    expected = np.asarray(expected, dtype=np.float32) / 255.0
    if (
        actual.shape != expected.shape
        or actual.ndim != 3
        or actual.shape[-1] != 3
        or min(actual.shape[:2]) < 11
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
    ):
        raise RuntimeError("SSIM requires matching, finite RGB images at least 11x11")
    score = float(
        structural_similarity(
            expected,
            actual,
            data_range=1.0,
            channel_axis=-1,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )
    if not np.isfinite(score):
        raise RuntimeError("Nonfinite SSIM from the selected transformer backend")
    return score
