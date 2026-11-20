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
"""Diagnostic prompt alignment for eager/tuned image pairs, outside benchmarks."""

import logging
import time

logger = logging.getLogger(__name__)
MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def score_image_pairs(cases) -> dict:
    """Score (label, prompt, eager_image, tuned_image) tuples with CLIP on CPU.

    CLIPScore is 100 times nonnegative image/text embedding cosine similarity,
    not the learned logit scale or a softmax probability. Report each pair and
    its tuned-minus-eager delta; no score determines validation pass/fail.
    The checkpoint is downloaded once into the usual Hugging Face cache.
    Diagnostic failures are recorded without interrupting artifact publication.
    """
    started = time.perf_counter()
    result = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "formula": "100 * max(cosine(image, text), 0)",
        "device": "cpu",
        "dtype": "float32",
        "status": "ERROR",
        "cases": {},
    }
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            backend="pil",
        )
        model = (
            CLIPModel.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                use_safetensors=False,
            )
            .to(device="cpu", dtype=torch.float32)
            .eval()
        )
        with torch.inference_mode():
            for label, prompt, eager_image, tuned_image in cases:
                inputs = processor(
                    text=[prompt],
                    images=[eager_image.convert("RGB"), tuned_image.convert("RGB")],
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                outputs = model(**inputs)
                image_features = torch.nn.functional.normalize(
                    outputs.image_embeds, dim=-1
                )
                text_features = torch.nn.functional.normalize(
                    outputs.text_embeds, dim=-1
                )
                cosine = (image_features * text_features).sum(dim=-1)
                if not torch.isfinite(cosine).all():
                    raise ValueError(f"Nonfinite CLIPScore for {label}")
                eager, tuned = (100 * cosine.clamp(0, 1)).tolist()
                result["cases"][label] = {
                    "prompt": prompt,
                    "eager": eager,
                    "tuned": tuned,
                    "delta": tuned - eager,
                }
                logger.info(
                    "%s CLIPScore: eager=%.4f, tuned=%.4f, delta=%+.4f",
                    label,
                    eager,
                    tuned,
                    tuned - eager,
                )
        result["status"] = "PASS"
    except Exception as error:  # noqa: BLE001 - optional diagnostic must not block serving artifacts
        result["error"] = f"{type(error).__name__}: {error}"
        logger.warning(
            "CLIPScore unavailable; continuing without complete scores: %s", error
        )
    result["seconds"] = time.perf_counter() - started
    return result
