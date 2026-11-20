<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Tune, serve, and benchmark FLUX.2-klein

AI Tune optimizes inference execution; it does not train model weights.
Tuning runs offline. The server either loads a saved AI Tune artifact or uses
`torch.compile` without AI Tune. It never tunes during startup.

Choose the workflow after building the same image:

| Goal | Entry point | Result |
| --- | --- | --- |
| Tune and serve | [Tune offline](#tune-offline), then [start the server](#start-the-server) | AI Tune selects a backend and saves an artifact; the server loads it for generation and editing |
| Compare configurations | [Benchmark all configurations](#benchmark-all-configurations) | The matrix tunes each precision/backend separately, starts servers, and prints and saves comparison tables |

Benchmarking is optional for serving. Both workflows use the same tuning
implementation and artifact format. An artifact saved by either workflow can
be passed directly to `flux_image_worker.py --artifact-dir`.

### Build

From the repository root:

```bash
docker build \
  --build-context dynamo-src=. \
  -f examples/diffusers/Dockerfile.aitune \
  -t dynamo-glos-flux2-klein:latest examples/diffusers
```

The image independently builds Dynamo and installs AI Tune 0.6.0 from PyPI.
Its default CUDA target is Ada (`CUDA_ARCHS=89`). On another GPU, add
`--build-arg CUDA_ARCHS=<compute capability without the dot>`; for example,
`120` for a GPU reporting capability `12.0`. Use Docker with NVIDIA GPU support.
The existing Docker Compose configuration does not launch this FLUX example.

### Tune offline

Create persistent host directories and open the image:

```bash
mkdir -p flux-artifacts precision-results
docker run --rm -it --gpus all --shm-size 8g \
  -v dynamo-flux2-klein-hf-cache:/root/.cache/huggingface \
  -v "$(pwd)/flux-artifacts:/artifacts" \
  -v "$(pwd)/precision-results:/results" \
  dynamo-glos-flux2-klein:latest bash
```

Inside the container:

```bash
python deploy/tune_flux.py --precision bf16 --artifact-dir /artifacts/bf16
```

Every invocation loads the model and runs AI Tune again. Existing artifacts do
not skip tuning. Without `--artifact-dir`, the default is
`flux-artifacts/<precision>` relative to the working directory.
Use `--size`, `--steps`, and `--max-image-side` to set the tuning workload and
shape bounds (defaults: `1024x1024`, four steps, maximum side 2048).

Normal tuning evaluates AI Tune's default candidates with their default compiler
settings. It selects the fastest successful candidate, even if eager is faster.
An optional `--backend <ID>` isolates a candidate for diagnostics; discover IDs
with `python deploy/check_precision_backends.py --list-backends`.

The artifact directory contains `manifest.json` and versioned checkpoint/build
subdirectories under `runs/`. A successful save atomically publishes the new
manifest. A failed tuning attempt leaves an earlier published artifact intact;
its logs/report identify the failed attempt. Eager fallback is never published
as an optimized artifact. Each attempt records overall tuning time and backend
build times in `runs/<run>/tuning.json`; the manifest records successful results.

Image checks compare generation, editing, and a held-out prompt with eager at
the same precision and seeds. Image agreement uses RGB structural similarity
(SSIM), averaged across channels with an 11-pixel Gaussian window (sigma 1.5),
population covariance, and pixel values scaled to [0, 1]. Every case must have
SSIM at least `--min-ssim` (default `0.90`, supported by both tuning and benchmark
commands). This default is a starting point to calibrate against reviewed image
pairs, not a universal image-quality threshold. Scores and the threshold are
saved under `image_validation` in the JSON report and artifact manifest.
Image mismatches are logged and saved as `Validation: FAIL`, but the
compiled artifact is still published. The command exits with status 1 for a
mismatch; the printed artifact path remains usable and serving logs a warning.

CLIPScore additionally measures prompt alignment for each eager/tuned image pair.
It reports `100 * max(cosine(image_embedding, text_embedding), 0)`, on a 0–100
scale, plus the tuned-minus-eager difference. The scorer uses
`openai/clip-vit-base-patch32` at a pinned revision, with standard CLIP image/text
preprocessing, FP32 inference, and CPU execution after image validation.
No extra Python package is required; the first run downloads the CLIP weights
into the Hugging Face cache. Cache them before using offline mode.

CLIPScore has no pass/fail threshold and does not affect SSIM validation, artifact
saving, or benchmarks. Download/scoring errors are logged and shown in the report.
Editing scores use the edit instruction, so they do not assess preservation of
the reference image. Scores, prompts, model revision, and scoring duration are
saved under `clip_score` in reports and the artifact manifest. Overall tuning
time includes CLIP loading/scoring; backend build and server timings exclude it.

### Start the server

Exit the tuning container. On the host, start the worker with the artifact path:

```bash
docker run -d --name dynamo-glos-flux2-klein \
  --gpus all --shm-size 8g -p 127.0.0.1:8000:8000 \
  -e DYN_DISCOVERY_BACKEND=file -e DYN_EVENT_PLANE=zmq \
  -e DYN_FILE_KV=/tmp/dynamo-discovery \
  -v dynamo-flux2-klein-hf-cache:/root/.cache/huggingface \
  -v "$(pwd)/flux-artifacts:/artifacts:ro" \
  dynamo-glos-flux2-klein:latest \
  python deploy/flux_image_worker.py --artifact-dir /artifacts/bf16

docker exec -d dynamo-glos-flux2-klein \
  python -m dynamo.frontend --http-host 0.0.0.0 --http-port 8000 \
  --discovery-backend file --request-plane tcp --event-plane zmq
```

The artifact supplies the model, precision, startup size, denoising steps, and
maximum image side unless overridden. Explicit model/precision mismatches fail.
Missing files, failed integrity checks, incompatible package versions or GPU
targets, and activation/warmup failures also stop startup. There is no retuning
or eager fallback. Build and serve with the same image and GPU type.

To run **without AI Tune**, replace the worker command above with:

```bash
python deploy/flux_image_worker.py --precision bf16
```

Without `--artifact-dir`, `torch.compile` is always enabled. Use
`--torch-compile-mode` to select its mode (`default` by default). The legacy
`--torch-compile` flag is still accepted but unnecessary. `--use-aitune` has
been replaced by `--artifact-dir`.

Both modes warm generation and editing before registration; `--warmup` must be
at least one. Worker logs show model-loading time, warmup time, total pipeline
startup time, and PyTorch-reported startup compilation time.

**AoT versus JIT artifacts:** ahead-of-time artifacts load compiled code. JIT
artifacts still compile when activated, even though backend selection is saved.
The default strategy can select either; inspect `backends[].build_mode` in the
manifest. Persistent compiler caches can shorten JIT startup, but do not turn it
into AoT loading. The text encoder and VAE are still loaded for every server.

Wait until the model appears in `data`, then generate an image:

```bash
curl -fsS http://localhost:8000/v1/models
curl --fail-with-body -sS http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"black-forest-labs/FLUX.2-klein-4B","prompt":"A red fox in a sunny alpine meadow","size":"1024x1024","response_format":"b64_json","nvext":{"num_inference_steps":4,"guidance_scale":1.0,"seed":42}}' \
  > response.json

docker exec -i dynamo-glos-flux2-klein python -c \
  'import sys,json,base64; r=json.load(sys.stdin); sys.stdout.buffer.write(base64.b64decode(r["data"][0]["b64_json"]))' \
  < response.json > generated.png
```

Editing uses `/v1/images/edits` with an `input_reference` image data URL or an
allowed HTTPS image URL. An empty model list does not mean the worker is ready.
Follow startup with `docker logs -f dynamo-glos-flux2-klein`.

### Precision

| Precision | Transformer | Hardware |
| --- | --- | --- |
| `bf16` | BF16 weights and computation | BF16-capable CUDA GPU |
| `fp16` | FP16 transformer, with BF16 pipeline boundary conversion | BF16-capable CUDA GPU |
| `fp8` | Calibrated BFL E4M3 checkpoint with saved weight and activation scales | Compute capability 8.9+ |
| `fp4` / `nvfp4` | Calibrated NVIDIA NVFP4 checkpoint | Blackwell, compute capability 10.0+ |

The text encoder and VAE remain BF16. FP8/NVFP4 use TorchAO native linear
operations behind custom PyTorch operators, with native LayerNorm/RMSNorm.
NVFP4 uses Triton for activation quantization and scale swizzling.
Inductor can compile around these operations. TensorRT candidates may fail to
export unsupported custom operations or FP64 positional embeddings; the matrix
records these failures. Quantized compiled images can differ from eager and
remain benchmarkable with the mismatch visible.

### Benchmark all configurations

Open the interactive container with `/artifacts` and `/results` mounted as above,
then run:

```bash
python deploy/check_precision_backends.py \
  --artifact-dir /artifacts/matrix \
  --output-dir /results \
  --benchmark-warmup 3 --benchmark-requests 30
```

The default matrix includes BF16, FP16, FP8, FP4, an eager diagnostic baseline,
a **no-AI-Tune torch.compile server**, and every installed default AI Tune
candidate. Unsupported precisions are skipped. Each AI Tune combination is tuned
and serialized in one process. That process exits before a fresh server loads
the artifact and is benchmarked through Dynamo HTTP. Image mismatches do not
block benchmarks. Compilation failures and eager fallback do.

The matrix calls `tune_flux.run_tuning()`, the same function used by the
standalone `tune_flux.py` command, with one default backend selected per case.
Both paths share precision loading, workload recording, compilation, SSIM and
CLIP scoring, and artifact saving. Benchmark servers use the same
`FluxImageBackend` artifact loading, warmup, and request handling as the public
worker; only the diagnostic eager baseline bypasses compilation.

The script prints and saves these tables:

| Table | Measurement |
| --- | --- |
| Checks | Compilation, generation, editing, validation, lowest measured SSIM, artifact save, startup, and benchmark status |
| CLIPScore | Eager/tuned prompt alignment and their difference for each validation case; diagnostic only |
| HTTP latency | Mean/P50/P95 milliseconds from request send through complete response transfer |
| Throughput | Successful images/second, first measured send to last complete response |
| Generation time | Mean/P50/P95 milliseconds for the synchronized worker pipeline call, excluding PNG encoding and transport |
| Tuning and startup | Overall tuning, backend builds, model loading, warmup, startup compilation, and total server startup, in seconds |

HTTP latency, generation time, and throughput also have summary tables with
precisions as rows and backends as columns, printed and saved in `results.md`.
Cells show mean latency/generation time or the run's aggregate images/second.
Unavailable benchmarks show `--`. Validation failures retain their measured
values; consult the detailed tables for validation and benchmark status.

The timing table separates **Server mode** (`AI Tune`, `No AI Tune`, or
`Diagnostic eager`) from **Compilation type** (`AoT`, `JIT`, or `None`).
Failed candidates still show their declared compilation type; this does not
indicate successful compilation. Check the compilation and startup status columns.

**Timing boundaries:** overall tuning includes process launch, imports, loading,
recording, building/profiling candidates, validation, and serialization. Backend
build time surrounds AI Tune's backend build/export calls, including failures;
it excludes strategy profiling and image comparison. Startup spans fresh-process
launch to model visibility in `/v1/models`, after loading and generation/editing
warmup, including frontend startup. PyTorch-reported frame compilation time is
shown separately within startup. Warmup includes inference as well as any JIT
compilation; these columns overlap and must not be added together.

The default `--compiler-cache cold` gives each tuning and server process separate
Inductor/Triton caches. This exposes compilation costs without reusing caches
populated during tuning. Use `--compiler-cache warm` to reuse the environment's
compiler caches and measure cached restarts. Downloaded model weights are reused
in either mode. Disk/OS caches are not flushed. The eager baseline is diagnostic;
the public no-artifact server always uses torch.compile.

Performance requests are serial, one image per request, using the same three
prompts and deterministic seeds. Startup and HTTP warmups are excluded from the
performance metrics. Throughput includes client work between requests and is not
maximum-concurrency throughput. Defaults are two warmups and ten measured
requests; percentiles use nearest rank.

Results are saved in `/results/<timestamp>/results.md` and `results.json`.
Artifacts are saved under `/artifacts/matrix/<timestamp>/<precision>-<backend>/`;
pass that exact directory to the server's `--artifact-dir`. Each case has
`tune.log`, `serve.log`, `frontend.log`, `benchmark.json`, and `requests.jsonl`
when those stages run. Individual build timings and image mismatches are retained.
Default host-relative artifact root: `flux-artifacts`.

Useful options:

```bash
python deploy/check_precision_backends.py --list-backends
python deploy/check_precision_backends.py --precisions fp8 fp4 --artifact-dir /artifacts/matrix --output-dir /results
python deploy/check_precision_backends.py --precisions bf16 --backends torch_compile 3 --artifact-dir /artifacts/matrix --output-dir /results
```

Numeric backend IDs come from `--list-backends`; `eager` and `torch_compile` are
named baselines. `--size` and `--steps` select the workload. `--timeout` defaults
to 1,800 seconds per combination, covering tuning and serving; individual HTTP
requests use `--request-timeout` (120 seconds). `--benchmark-requests 0` tunes
and saves artifacts without starting servers or measuring startup/performance.
Failures do not stop subsequent combinations. Exit codes: 1 for any failed check
(including image mismatches), 2 for argument/discovery errors, 130 for interruption.
Completed tables and partial diagnostics remain saved.
