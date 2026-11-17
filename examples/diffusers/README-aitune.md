<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Serve a tuned FLUX.2-klein

Run text-to-image generation and reference-image editing with FLUX.2-klein,
optimized at startup by AI Tune and served by NVIDIA Dynamo.
Tuning selects an inference backend; it does not train or fine-tune model weights.

### Summary

#### Build

Use Docker with NVIDIA GPU support and a GPU that supports the selected precision
(see the table below). Build the example image from the repository root:

```bash
docker build \
  --build-context dynamo-src=. \
  -f examples/diffusers/Dockerfile.aitune \
  -t dynamo-glos-flux2-klein:latest examples/diffusers
```

#### Tuning and starting the service

Start the FLUX worker and Dynamo frontend:

```bash
docker run -d --name dynamo-glos-flux2-klein \
  --gpus all --shm-size 8g \
  -p 127.0.0.1:8000:8000 \
  -e DYN_DISCOVERY_BACKEND=file -e DYN_EVENT_PLANE=zmq \
  -e DYN_FILE_KV=/tmp/dynamo-discovery \
  -v dynamo-flux2-klein-hf-cache:/root/.cache/huggingface \
  dynamo-glos-flux2-klein:latest \
  python deploy/flux_image_worker.py --precision bf16 --use-aitune

docker exec -d dynamo-glos-flux2-klein \
  python -m dynamo.frontend --http-host 0.0.0.0 --http-port 8000 \
  --discovery-backend file --request-plane tcp --event-plane zmq
```

Startup includes tuning, output validation, and warmup. Wait until the response
below lists `black-forest-labs/FLUX.2-klein-4B` in `data`; HTTP 200 with an empty
list does not mean the model is ready. Use `docker logs -f dynamo-glos-flux2-klein`
to follow startup progress.

```bash
curl -fsS http://localhost:8000/v1/models
```

#### Image generation

Generate an image and save it as `generated.png`:

```bash
curl --fail-with-body -sS http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "black-forest-labs/FLUX.2-klein-4B",
    "prompt": "A red fox in a sunny alpine meadow",
    "size": "1024x1024",
    "response_format": "b64_json",
    "nvext": {"num_inference_steps": 4, "guidance_scale": 1.0, "seed": 42}
  }' > response.json

docker exec -i dynamo-glos-flux2-klein python -c \
  'import sys,json,base64; r=json.load(sys.stdin); sys.stdout.buffer.write(base64.b64decode(r["data"][0]["b64_json"]))' \
  < response.json > generated.png
```

For editing, send JSON to `/v1/images/edits` with `input_reference` containing
an image data URL or an allowed HTTPS image URL. Both endpoints use the selected
transformer backend, including eager fallback if compilation fails.
Ahead-of-time (AoT) input-shape bounds follow the
worker's `--max-image-side` setting and include both generation and editing.

### Check precisions and backends

After building the image, open a container with an output directory mounted from
the host. Run this on an idle GPU; each combination loads the full model and
compilation can take several minutes.

```bash
mkdir -p precision-results
docker run --rm -it --gpus all --shm-size 8g \
  -v dynamo-flux2-klein-hf-cache:/root/.cache/huggingface \
  -v "$(pwd)/precision-results:/results" \
  dynamo-glos-flux2-klein:latest bash
```

Inside the container, run:

```bash
python deploy/check_precision_backends.py --output-dir /results
```

The script tests BF16, FP16, FP8, and FP4 (NVFP4), each against eager execution
and every default candidate discovered from the installed AI Tune package.
It isolates candidates for this diagnostic without overriding their compiler
settings or changing normal service startup. Unsupported hardware combinations
are marked `SKIP`, so FP4 is skipped on Ada.

The table is printed after each combination and saved as `results.md` in a new
timestamped directory under `precision-results` on the host. `results.json`
also records the GPU, package versions, candidate configurations, timings, and
errors. Each combination has its own `run.log`, AI Tune artifacts, and output PNGs.
No frontend or HTTP service is started.

The columns have distinct meanings:

- **Compile**: `PASS` means AI Tune built, checked, profiled, and selected the
  isolated candidate. This exercises deferred JIT compilation too. Eager fallback
  is a failure for that candidate, and eager's own row shows `N/A`.
- **Generation / Editing**: `PASS` means the selected backend produced and saved
  that image. It does not establish numerical correctness.
- **Validation**: `PASS` means the worker's generation, editing, and held-out
  images passed the same-seed, same-precision eager comparison described below.
  Eager's own row shows `N/A` and checks image dimensions and nonuniform pixels.
- **Result**: the overall outcome, including loading failures. `NOT RUN` in a
  stage means an earlier failure prevented that check. See the row's log for
  the underlying error.

The timeout defaults to 1,800 seconds per combination, including model loading.
A failure or timeout does not stop subsequent combinations. Exit status is `1`
if any combination failed, `2` for setup errors, and `130` for an interrupted run;
completed and skipped results remain saved. Each combination uses a fresh AI Tune
cache, while lower-level compiler caches and downloaded weights may be reused.

To limit a run or identify individual default candidates:

```bash
python deploy/check_precision_backends.py --list-backends --output-dir /results
python deploy/check_precision_backends.py --precisions fp8 fp4 --output-dir /results
python deploy/check_precision_backends.py --precisions bf16 --backends eager --output-dir /results
```

`--backends` also accepts the numeric IDs printed by `--list-backends`. IDs
distinguish different configurations of the same backend class. Use `--timeout`
to allow longer compilation, or `--size` and `--steps` to change the workload
(defaults: `1024x1024`, four steps). Results apply to the tested workload and
environment, not all image sizes or GPU architectures.

### Details

#### Build

`Dockerfile.aitune` builds an independent FLUX image directly from NVIDIA's CUDA
13.1.1 development image. It installs AI Tune 0.6.0 from PyPI; no AI Tune checkout
or other example image is needed.
Use the launch commands above; the existing Docker Compose configuration does
not launch this FLUX example.

The image installs Python 3.12, PyTorch 2.11.0, Diffusers, Transformers, and TorchAO
explicitly, together with AI Tune's `torch211` extra.

This FLUX worker uses image-accounting types absent from the tested released
Dynamo wheels, so the image also builds Dynamo from this checkout.
The source build defaults to Ada (`CUDA_ARCHS=89`); set
`--build-arg CUDA_ARCHS=<architecture>` for another GPU.

Installing AI Tune does not enable tuning by itself: the worker's
`--use-aitune` flag controls that separately.

#### Precision and tuning

Choose the transformer precision with `--precision`:

| `--precision` | Transformer execution | Hardware |
| --- | --- | --- |
| `bf16` (default) | BF16 weights and computation | BF16-capable CUDA GPU |
| `fp16` | FP16 weights and computation, with BF16 pipeline inputs/outputs converted at the transformer boundary | BF16-capable CUDA GPU for the text encoder and VAE |
| `fp8` | Pretrained BFL FP8 E4M3 transformer weights with saved weight and activation scales; unquantized layers remain BF16 | Ada or newer (compute capability 8.9+) |
| `fp4` or `nvfp4` | The same pretrained NVIDIA NVFP4 transformer checkpoint | Blackwell (compute capability 10.0+) |

For example, use `--precision fp8 --use-aitune` to tune the FP8 transformer.
The text encoder and VAE remain BF16 for every choice. Precision is applied before
tuning, and eager fallback keeps that precision. Tuning does not automatically
choose among precisions.

FP8 and NVFP4 linear layers keep packed weights and scales as ordinary tensors.
A custom PyTorch operator invokes TorchAO's native quantization and matrix
multiplication, keeping its Python tensor wrappers outside compiler tracing.
This permits Inductor export without converting the weights back to BF16, but
prevents compiler fusion across those linear operations. Both precisions use
the calibration scales saved in their checkpoints. `fp8` downloads
`black-forest-labs/FLUX.2-klein-4b-fp8`; it does not quantize BF16 weights at startup.
Quantized transformers also retain native LayerNorm/RMSNorm kernels to limit
numerical differences at quantization boundaries.

Earlier FP8 generation and editing checks used explicit Inductor compiler
settings; those results do not establish correctness with the defaults.
NVFP4 checkpoint loading and full-model strict export were validated; native
NVFP4 execution and image quality require a Blackwell run.

AI Tune uses its default backend candidates and compiler settings. The example
does not specify a backend list or compiler overrides. It selects the fastest
successful candidate, even if eager execution is faster. If all candidates fail,
it logs an `ERROR` with the candidate
results and log paths, then falls back to eager execution at the selected precision.
The selected backend must pass output validation before the model is registered.
Validation compares generation, editing, and a held-out prompt against eager
outputs at the same precision and seeds, with a maximum pixel RMSE of 0.05 on
values scaled to [0, 1]. A validation failure stops startup.
Successful image generation alone does not prove that a compiled backend worked:
check the `Serving transformer` log entry for the selected backend.

`--use-aitune` and the existing JIT `--torch-compile` option are mutually exclusive.
`--torch-compile-mode` applies only to the latter.

There is no separate tuning command, checkpoint argument, or tuning volume.
With AI Tune enabled, each worker start selects and validates a transformer backend.
Build artifacts and compiler caches use the existing Hugging Face cache volume.
JIT backends compile during tuning and warmup and may compile again for new inputs.
