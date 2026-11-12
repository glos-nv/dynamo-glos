#!/usr/bin/env python3
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
"""FLUX.2-klein diffusers example worker for Dynamo's /v1/images endpoints.

Serves text-to-image and image-to-image (reference-image editing) from a single
``Flux2KleinPipeline``, with a BF16, FP16, FP8, or NVFP4 transformer.
``--precision fp4`` is an alias for ``nvfp4``. The text encoder and VAE stay BF16.
It registers as ``ModelType.Images`` so the stock Dynamo frontend exposes:

* ``POST /v1/images/generations`` -- text-to-image, or image-to-image when the
  request also carries ``input_reference``
* ``POST /v1/images/edits`` -- image-to-image; ``input_reference`` is required

``input_reference`` is an ``https://`` URL or a ``data:`` URL; it is fetched and
decoded through Dynamo's SSRF-safe ``ImageLoader``, so the usual ``DYN_MM_*``
policy variables apply. The reference is resized to the requested output size so
that ``torch.compile`` sees one shape per mode.

One request is processed at a time because the diffusers pipeline is not
re-entrant. Excess requests are rejected before reference images are loaded.
"""

import argparse
import asyncio
import base64
import inspect
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import torch
import uvloop
from dynamo.common.multimodal import ImageLoader
from dynamo.common.protocols.image_protocol import (
    ImageAccounting,
    ImageData,
    ImagesResponseNvExt,
    NvCreateImageRequest,
    NvImagesResponse,
)
from dynamo.llm import (  # type: ignore[attr-defined]
    ModelInput,
    ModelType,
    WorkerType,
    register_model,
)
from dynamo.llm.exceptions import InvalidArgument, WorkerOverloaded
from dynamo.runtime import DistributedRuntime, dynamo_endpoint
from flux_precision import PRECISIONS, check_precision_support, prepare_transformer
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-4B"
NVFP4_REPO = "black-forest-labs/FLUX.2-klein-4b-nvfp4"
NVFP4_FILE = "flux-2-klein-4b-nvfp4.safetensors"
DEFAULT_SIZE = "1024x1024"
DEFAULT_NUM_INFERENCE_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_MAX_IMAGE_SIDE = 2048
DEFAULT_MAX_NUM_INFERENCE_STEPS = 50
# Flux2 packs 2x2 latent patches on an 8x VAE, so sides must be multiples of 16.
SIZE_MULTIPLE = 16
SUPPORTED_OUTPUT_FORMATS = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}
TORCH_COMPILE_MODE_CHOICES = ("default", "reduce-overhead", "max-autotune")


def _get_worker_namespace() -> str:
    """Resolve the Dynamo namespace for endpoint registration.

    The Kubernetes operator injects DYN_NAMESPACE (and optionally a rollout
    suffix). Compose/local runs keep using the historical "dynamo" default.
    """
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace


class FluxImageBackend:
    """Serve one FLUX pipeline, serializing GPU work on a dedicated thread."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model_name: str = args.served_model_name or args.model
        self.pipe: Any = None
        self.supports_edit = False
        self._image_loader = ImageLoader()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._request_task: asyncio.Task[dict[str, Any]] | None = None

    async def initialize_model(self) -> None:
        """Load, optionally tune, and warm up the pipeline before registration."""
        loop = asyncio.get_running_loop()
        self.pipe = await loop.run_in_executor(self._executor, self._load)
        if self.args.warmup > 0:
            await loop.run_in_executor(self._executor, self._warmup)
        logger.info("Pipeline ready (%s)", self.model_name)

    async def close(self) -> None:
        """Drain an in-flight request before shutting down the inference thread."""
        if self._request_task is not None:
            await asyncio.gather(self._request_task, return_exceptions=True)
        await asyncio.to_thread(self._executor.shutdown, wait=True)

    def _request_finished(self, task: asyncio.Task[dict[str, Any]]) -> None:
        self._request_task = None
        if not task.cancelled():
            task.exception()

    def _load(self) -> Any:
        """Load the chosen precision and optionally select a validated backend."""
        from diffusers import Flux2KleinPipeline

        check_precision_support(self.args.precision)
        if self.args.precision in ("fp4", "nvfp4"):
            # NVFP4 transformer (Blackwell FP4 tensor cores); the text encoder
            # and VAE stay BF16 and come from the base model.
            from huggingface_hub import hf_hub_download
            from nvfp4_loader import build_nvfp4_transformer

            logger.info("Loading NVFP4 transformer from %s", NVFP4_REPO)
            checkpoint = hf_hub_download(NVFP4_REPO, NVFP4_FILE)
            transformer = build_nvfp4_transformer(checkpoint, self.args.model)
            pipe = Flux2KleinPipeline.from_pretrained(
                self.args.model, transformer=transformer, torch_dtype=torch.bfloat16
            )
        else:
            logger.info("Loading %s", self.args.model)
            pipe = Flux2KleinPipeline.from_pretrained(
                self.args.model, torch_dtype=torch.bfloat16
            )
        pipe = pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        prepare_transformer(pipe.transformer, self.args.precision)

        if self.args.use_aitune:
            # Keep AI Tune optional for the original worker launch path.
            from flux_aitune import tune_transformer

            width, height = self._parse_size(self.args.default_size)
            tune_transformer(
                pipe,
                width=width,
                height=height,
                steps=self.args.default_num_inference_steps,
                guidance_scale=self.args.default_guidance_scale,
                max_image_side=self.args.max_image_side,
                size_multiple=SIZE_MULTIPLE,
            )
        elif self.args.torch_compile:
            logger.info(
                "torch.compile(transformer.forward, mode=%s)",
                self.args.torch_compile_mode,
            )
            pipe.transformer.forward = torch.compile(
                pipe.transformer.forward,
                mode=self.args.torch_compile_mode,
                fullgraph=False,
                dynamic=False,
            )
            # t2i and i2i have different token counts: allow a graph per mode.
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit, 16
            )

        # Fail at startup, not on the first edit request, if this diffusers
        # build's pipeline does not accept a reference image.
        self.supports_edit = "image" in inspect.signature(pipe.__call__).parameters
        if not self.supports_edit:
            logger.warning(
                "%s.__call__ has no `image` parameter; serving text-to-image only",
                type(pipe).__name__,
            )
        return pipe

    def _warmup(self) -> None:
        """Warm generation and editing, including any deferred JIT compilation."""
        width, height = self._parse_size(self.args.default_size)
        blank = Image.new("RGB", (width, height), (127, 127, 127))
        modes = [("t2i", None)]
        if self.supports_edit:
            modes.append(("i2i", blank))
        for label, image in modes:
            started = time.perf_counter()
            for _ in range(self.args.warmup):
                self._run(
                    prompt="warmup",
                    image=image,
                    width=width,
                    height=height,
                    steps=self.args.default_num_inference_steps,
                    guidance_scale=self.args.default_guidance_scale,
                    seed=0,
                )
            logger.info(
                "Warmup[%s]: %d iters in %.1fs",
                label,
                self.args.warmup,
                time.perf_counter() - started,
            )

    def _parse_size(self, size: str | None) -> tuple[int, int]:
        raw = size or self.args.default_size
        try:
            width_str, height_str = raw.lower().split("x", 1)
            width, height = int(width_str), int(height_str)
        except ValueError as exc:
            raise ValueError(f"Invalid size '{raw}', expected WxH") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid size '{raw}', must be positive")
        if width % SIZE_MULTIPLE or height % SIZE_MULTIPLE:
            raise ValueError(
                f"Invalid size '{raw}', sides must be multiples of {SIZE_MULTIPLE}"
            )
        if max(width, height) > self.args.max_image_side:
            raise ValueError(
                f"Invalid size '{raw}', exceeds maximum side {self.args.max_image_side}"
            )
        return width, height

    def _run(
        self,
        prompt: str,
        image: Image.Image | None,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> Image.Image:
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "output_type": "pil",
        }
        if image is not None:
            kwargs["image"] = image
        if seed is not None:
            kwargs["generator"] = torch.Generator(device="cuda").manual_seed(seed)
        return self.pipe(**kwargs).images[0]

    @staticmethod
    def _encode(image: Image.Image, output_format: str) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format=SUPPORTED_OUTPUT_FORMATS[output_format])
        return buf.getvalue()

    @dynamo_endpoint(NvCreateImageRequest, NvImagesResponse)
    async def create_image(self, request: NvCreateImageRequest):
        """Generate one image (t2i, or i2i when ``input_reference`` is set)."""
        if self.pipe is None:
            raise RuntimeError("Pipeline is not initialized")
        if self._request_task is not None:
            raise WorkerOverloaded(
                "Image worker is busy; retry after the current request"
            )

        # Cancellation of the caller must not release admission while work continues.
        task = asyncio.create_task(self._create_image(request))
        self._request_task = task
        task.add_done_callback(self._request_finished)
        yield await asyncio.shield(task)

    async def _create_image(self, request: NvCreateImageRequest) -> dict[str, Any]:
        # Validation failures are the client's fault -> InvalidArgument (HTTP 400).
        nvext = request.nvext
        if nvext and nvext.negative_prompt is not None:
            raise InvalidArgument("negative_prompt is not supported by this example")
        for field in ("background", "quality", "style", "moderation"):
            if getattr(request, field) is not None:
                raise InvalidArgument(f"{field} is not supported by this example")
        try:
            width, height = self._parse_size(request.size)
        except ValueError as exc:
            raise InvalidArgument(str(exc)) from exc
        if request.n not in (None, 1):
            raise InvalidArgument("Only n=1 is supported")
        output_format = (request.output_format or "png").lower()
        if output_format not in SUPPORTED_OUTPUT_FORMATS:
            raise InvalidArgument(
                f"Unsupported output_format '{output_format}', "
                f"expected one of {sorted(SUPPORTED_OUTPUT_FORMATS)}"
            )
        response_format = request.response_format or "b64_json"
        if response_format != "b64_json":
            raise InvalidArgument(
                "Only response_format=b64_json is supported by this example"
            )
        steps = (
            nvext.num_inference_steps
            if nvext and nvext.num_inference_steps is not None
            else self.args.default_num_inference_steps
        )
        if steps <= 0 or steps > self.args.max_num_inference_steps:
            raise InvalidArgument(
                f"num_inference_steps must be in [1, {self.args.max_num_inference_steps}]"
            )
        guidance_scale = (
            nvext.guidance_scale
            if nvext and nvext.guidance_scale is not None
            else self.args.default_guidance_scale
        )
        seed = nvext.seed if nvext and nvext.seed is not None else None

        reference: Image.Image | None = None
        if request.input_reference is not None:
            if not self.supports_edit:
                raise InvalidArgument(
                    "This diffusers build does not support reference images"
                )
            try:
                reference = await self._image_loader.load_image(request.input_reference)
            except (ValueError, Image.DecompressionBombError) as exc:
                # Keep URLs and inline image data out of the client error.
                raise InvalidArgument("Failed to load input_reference") from exc
            reference = await asyncio.to_thread(
                reference.resize, (width, height), Image.LANCZOS
            )

        kind = "i2i" if reference is not None else "t2i"
        logger.info(
            "create_image(%s) size=%dx%d steps=%d prompt=%r",
            kind,
            width,
            height,
            steps,
            request.prompt[:60],
        )

        started = time.perf_counter()
        image = await asyncio.get_running_loop().run_in_executor(
            self._executor,
            partial(
                self._run,
                prompt=request.prompt,
                image=reference,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
            ),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info("create_image(%s) done in %.1f ms", kind, elapsed_ms)

        payload = await asyncio.to_thread(self._encode, image, output_format)
        return NvImagesResponse(
            created=int(time.time()),
            data=[ImageData(b64_json=base64.b64encode(payload).decode())],
            size=f"{width}x{height}",
            output_format=output_format,
            nvext=ImagesResponseNvExt(
                usage=ImageAccounting(
                    output_images=1,
                    input_images=1 if reference is not None else 0,
                    num_inference_steps=steps,
                )
            ),
        ).model_dump(exclude_none=True)


async def _register_model(endpoint, model_name: str) -> None:
    try:
        await register_model(
            ModelInput.Text,  # type: ignore[attr-defined]
            ModelType.Images,
            endpoint,
            model_name,
            model_name,
            worker_type=WorkerType.Aggregated,
        )
        logger.info("Successfully registered model: %s", model_name)
    except Exception as e:
        logger.exception("Failed to register model")
        raise RuntimeError("Model registration failed") from e


async def backend_worker(runtime: DistributedRuntime, args: argparse.Namespace) -> None:
    namespace_name = _get_worker_namespace()
    endpoint = runtime.endpoint(f"{namespace_name}.backend.generate")
    logger.info("Serving endpoint %s/backend/generate", namespace_name)

    backend = FluxImageBackend(args)
    try:
        await backend.initialize_model()
        await asyncio.gather(
            endpoint.serve_endpoint(backend.create_image),  # type: ignore[arg-type]
            _register_model(endpoint, backend.model_name),
        )
    finally:
        await backend.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FLUX.2-klein diffusers worker for Dynamo (t2i + i2i)"
    )
    parser.add_argument(
        "--model",
        "--model-path",
        dest="model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Name clients put in `model` (default: --model).",
    )
    parser.add_argument(
        "--discovery-backend",
        choices=("etcd", "file", "mem", "kubernetes"),
        default=None,
        help="Dynamo discovery backend. Defaults to env, then kubernetes/file.",
    )
    parser.add_argument(
        "--precision",
        choices=PRECISIONS,
        default="bf16",
        help=(
            "Transformer precision. fp8 needs Ada or newer and torchao; "
            "fp4 is an alias for nvfp4 and needs Blackwell and torchao. "
            "The text encoder and VAE stay BF16."
        ),
    )
    parser.add_argument(
        "--use-aitune",
        action="store_true",
        help="Tune the loaded transformer with AoT and JIT backends before serving (any precision).",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        dest="torch_compile",
        help="torch.compile the transformer forward (slower start, faster steady state).",
    )
    parser.add_argument(
        "--torch-compile-mode",
        choices=TORCH_COMPILE_MODE_CHOICES,
        default="default",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations per mode (t2i, i2i) at startup; 0 disables.",
    )
    parser.add_argument("--default-size", default=DEFAULT_SIZE)
    parser.add_argument(
        "--default-num-inference-steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS
    )
    parser.add_argument(
        "--default-guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE
    )
    parser.add_argument("--max-image-side", type=int, default=DEFAULT_MAX_IMAGE_SIDE)
    parser.add_argument(
        "--max-num-inference-steps",
        type=int,
        default=DEFAULT_MAX_NUM_INFERENCE_STEPS,
    )
    args = parser.parse_args()
    if args.use_aitune and args.torch_compile:
        parser.error("--use-aitune and --torch-compile are mutually exclusive")
    for name in (
        "max_image_side",
        "max_num_inference_steps",
        "default_num_inference_steps",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.default_num_inference_steps > args.max_num_inference_steps:
        parser.error(
            "--default-num-inference-steps must not exceed --max-num-inference-steps"
        )
    return args


async def main(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    discovery_backend = args.discovery_backend or os.environ.get(
        "DYN_DISCOVERY_BACKEND"
    )
    if not discovery_backend:
        discovery_backend = (
            "kubernetes" if os.environ.get("KUBERNETES_SERVICE_HOST") else "file"
        )
    logger.info("Using discovery backend: %s", discovery_backend)
    runtime = DistributedRuntime(loop, discovery_backend, "tcp")
    await backend_worker(runtime, args)


if __name__ == "__main__":
    _args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    uvloop.install()
    asyncio.run(main(_args))
