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
| `fp8` | TorchAO dynamic FP8 E4M3 weights and activations for large linear layers; embeddings, output projections, and smaller layers remain BF16 | Ada or newer (compute capability 8.9+) |
| `fp4` or `nvfp4` | The same pretrained NVIDIA NVFP4 transformer checkpoint | Blackwell (compute capability 10.0+) |

For example, use `--precision fp8 --use-aitune` to tune the FP8 transformer.
The text encoder and VAE remain BF16 for every choice. Precision is applied before
tuning, and eager fallback keeps that precision. Tuning does not automatically
choose among precisions.

AI Tune compares TorchInductor and Torch-TensorRT in AoT and just-in-time (JIT)
modes. It selects the fastest successful compiled candidate, even if eager
execution is faster. If all candidates fail, it logs an `ERROR` with the candidate
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
