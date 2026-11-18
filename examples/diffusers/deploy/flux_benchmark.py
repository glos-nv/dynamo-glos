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
"""Benchmark a loaded FLUX worker through a temporary local Dynamo frontend.

Used by check_precision_backends.py after image checks, including mismatches.
All requests are serial:
this example's worker rejects concurrent requests rather than queueing them.
"""

import asyncio
import base64
import io
import json
import math
import socket
import statistics
import sys
import time
import urllib.error
import urllib.request
from unittest.mock import patch

PROMPTS = (
    "A red fox in a sunny alpine meadow",
    "A blue ceramic teapot on a wooden table",
    "A lighthouse on a rocky coast at sunrise",
)


def _statistics(values):
    """Summarize seconds, using nearest-rank percentiles (also for small runs)."""
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "p50": ordered[math.ceil(0.50 * len(ordered)) - 1],
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
    }


def _request(url, payload, timeout):
    """Return parsed JSON and full-response HTTP timing; bypass local proxies."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with opener.open(request, timeout=timeout) as response:
        body = response.read()
    finished = time.perf_counter()
    return json.loads(body), started, finished


def _check_image(response, size):
    from PIL import Image

    images = response.get("data", [])
    if len(images) != 1:
        raise RuntimeError("Expected exactly one image in the HTTP response")
    data = base64.b64decode(images[0]["b64_json"], validate=True)
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.size != tuple(int(side) for side in size.split("x")):
            raise RuntimeError("HTTP response image has an incorrect size")


async def _measure(base_url, backend, directory, options, generation_times):
    samples = []
    start = finish = None
    with (directory / "requests.jsonl").open("w") as stream:
        for index in range(options["warmup"] + options["requests"]):
            warming = index < options["warmup"]
            sample_index = index if warming else index - options["warmup"]
            payload = {
                "model": backend.model_name,
                "prompt": PROMPTS[sample_index % len(PROMPTS)],
                "size": options["size"],
                "n": 1,
                "response_format": "b64_json",
                "nvext": {
                    "num_inference_steps": options["steps"],
                    "guidance_scale": 1.0,
                    "seed": 42 + sample_index,
                },
            }
            generation_times.clear()
            response, sent, received = await asyncio.to_thread(
                _request,
                base_url + "/v1/images/generations",
                payload,
                options["request_timeout"],
            )
            _check_image(response, options["size"])
            if len(generation_times) != 1:
                raise RuntimeError(
                    "Expected one worker generation timing per HTTP request"
                )
            sample = {
                "index": sample_index,
                "warmup": warming,
                "prompt": payload["prompt"],
                "seed": payload["nvext"]["seed"],
                "latency_s": received - sent,
                "generation_s": generation_times[0],
            }
            stream.write(json.dumps(sample) + "\n")
            stream.flush()
            if not warming:
                if start is None:
                    start = sent
                finish = received
                samples.append(sample)
    elapsed = finish - start
    result = {
        "workload": {
            **options,
            "concurrency": 1,
            "images_per_request": 1,
            "endpoint": "/v1/images/generations",
            "prompts": PROMPTS,
        },
        "completed_requests": len(samples),
        "elapsed_s": elapsed,
        "latency_s": _statistics([sample["latency_s"] for sample in samples]),
        "generation_s": _statistics([sample["generation_s"] for sample in samples]),
        "throughput_images_s": len(samples) / elapsed,
    }
    (directory / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


async def _serve_and_measure(backend, directory, options):
    import torch
    from dynamo.runtime import DistributedRuntime
    from flux_image_worker import _get_worker_namespace, _register_model

    # The case runner supplies isolated file discovery and namespace settings.
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "file", "tcp", event_plane="zmq"
    )
    serving = None
    frontend = None
    generation_times = []
    original_run = backend._run

    def timed_run(*args, **kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        image = original_run(*args, **kwargs)
        torch.cuda.synchronize()
        generation_times.append(time.perf_counter() - started)
        return image

    with (directory / "frontend.log").open("w") as log:
        try:
            endpoint = runtime.endpoint(f"{_get_worker_namespace()}.backend.generate")
            serving = asyncio.ensure_future(
                endpoint.serve_endpoint(backend.create_image)
            )
            await _register_model(endpoint, backend.model_name)
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            frontend = await asyncio.create_subprocess_exec(
                *[
                    sys.executable,
                    "-m",
                    "dynamo.frontend",
                    "--http-host",
                    "127.0.0.1",
                    "--http-port",
                    str(port),
                    "--discovery-backend",
                    "file",
                    "--request-plane",
                    "tcp",
                    "--event-plane",
                    "zmq",
                ],
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
            base_url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 60
            while True:
                if frontend.returncode is not None:
                    raise RuntimeError("Dynamo frontend exited; see frontend.log")
                if serving.done():
                    await serving
                    raise RuntimeError("Dynamo worker stopped before the benchmark")
                try:
                    response, _, _ = await asyncio.to_thread(
                        _request, base_url + "/v1/models", None, 2
                    )
                    if any(
                        model["id"] == backend.model_name
                        for model in response.get("data", [])
                    ):
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Model not ready in Dynamo after 60s; see frontend.log"
                    )
                await asyncio.sleep(0.25)
            with patch.object(backend, "_run", timed_run):
                return await _measure(
                    base_url, backend, directory, options, generation_times
                )
        finally:
            if frontend is not None:
                if frontend.returncode is None:
                    frontend.terminate()
                try:
                    await asyncio.wait_for(frontend.wait(), timeout=10)
                except TimeoutError:
                    frontend.kill()
                    await frontend.wait()
            runtime.shutdown()
            if serving is not None:
                serving.cancel()
                await asyncio.gather(serving, return_exceptions=True)
            await backend.close()


def benchmark_dynamo(backend, directory, options):
    """Serve the loaded model and measure warmed HTTP generation.

    Latency spans the HTTP request through receipt of the full response body.
    Generation time surrounds the worker's pipeline call, synchronized on CUDA,
    excluding PNG encoding, routing and HTTP transfer. Throughput is successful
    images divided by wall time from the first measured send to the last full
    response, including client work between requests. Warmups and startup are
    excluded. Failed requests invalidate the run; raw completed samples survive.
    """
    return asyncio.run(_serve_and_measure(backend, directory, options))
