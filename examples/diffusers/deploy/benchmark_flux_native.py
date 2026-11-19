# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the old FLUX worker without modifying its code or precision loaders.

Run in the worker's Python/container environment. This branch supports BF16 and
NVFP4; FP16/FP8 are reported as unsupported. One fresh process per precision.
Generation time covers the synchronized pipeline call, including text encoding,
denoising, VAE decoding and PIL conversion, excluding load, warmup and PNG saving.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

PROMPTS = (
    "A red fox in a sunny alpine meadow",
    "A blue ceramic teapot on a wooden table",
    "A lighthouse on a rocky coast at sunrise",
)
NATIVE_PRECISIONS = {"bf16": "bf16", "fp4": "nvfp4"}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_case(config_path):
    """Use the checked-out worker's loader and generation method unmodified."""
    config = json.loads(config_path.read_text())
    directory = config_path.parent
    precision = config["precision"]
    result = {"precision": precision, "status": "FAIL", "phase": "environment"}
    backend = None
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run in a GPU-enabled container")
        capability = torch.cuda.get_device_capability()
        result["environment"] = {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(capability),
            "versions": {},
        }
        for package in ("torch", "torchao", "diffusers", "transformers", "ai-dynamo"):
            try:
                result["environment"]["versions"][package] = importlib.metadata.version(
                    package
                )
            except importlib.metadata.PackageNotFoundError:
                result["environment"]["versions"][package] = None
        if precision == "fp4" and capability < (10, 0):
            result.update(
                status="SKIP",
                reason="Native NVFP4 requires Blackwell (compute capability 10.0+)",
            )
            return

        sys.path.insert(0, config["worker_dir"])
        import flux_image_worker as worker

        argv = [
            str(Path(worker.__file__)),
            "--precision",
            NATIVE_PRECISIONS[precision],
            "--default-size",
            config["size"],
            "--default-num-inference-steps",
            str(config["steps"]),
            "--warmup",
            "0",
        ]
        if config["torch_compile"]:
            argv.extend(
                [
                    "--torch-compile",
                    "--torch-compile-mode",
                    config["torch_compile_mode"],
                ]
            )
        original_argv = sys.argv
        try:
            sys.argv = argv
            args = worker._parse_args()
        finally:
            sys.argv = original_argv
        backend = worker.FluxImageBackend(args)
        result["phase"] = "load"
        started = time.perf_counter()
        backend.pipe = backend._load()
        torch.cuda.synchronize()
        result["load_seconds"] = time.perf_counter() - started
        width, height = backend._parse_size(config["size"])

        def generate(index):
            return backend._run(
                prompt=PROMPTS[index % len(PROMPTS)],
                image=None,
                width=width,
                height=height,
                steps=config["steps"],
                guidance_scale=args.default_guidance_scale,
                seed=42 + index,
            )

        result["phase"] = "warmup"
        started = time.perf_counter()
        for index in range(config["warmup"]):
            generate(index)
        torch.cuda.synchronize()
        result["warmup_seconds"] = time.perf_counter() - started
        result["phase"] = "generation"
        durations = []
        with (directory / "samples.jsonl").open("w") as stream:
            for index in range(config["requests"]):
                torch.cuda.synchronize()
                started = time.perf_counter()
                image = generate(index)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if image.size != (width, height):
                    raise RuntimeError(f"Unexpected image size: {image.size}")
                durations.append(elapsed)
                stream.write(
                    json.dumps(
                        {
                            "index": index,
                            "prompt": PROMPTS[index % len(PROMPTS)],
                            "seed": 42 + index,
                            "generation_seconds": elapsed,
                        }
                    )
                    + "\n"
                )
                stream.flush()
                print(
                    f"{precision}: image {index + 1}/{config['requests']}: {elapsed:.3f}s",
                    flush=True,
                )
                if index == 0:
                    image.save(directory / "sample.png")
        ordered = sorted(durations)
        result.update(
            status="PASS",
            images=len(durations),
            generation_seconds={
                "mean": statistics.mean(durations),
                "p50": ordered[math.ceil(len(ordered) * 0.50) - 1],
                "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
            },
        )
    except Exception as error:  # noqa: BLE001 - retain diagnostics and continue precisions
        result["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        write_json(directory / "result.json", result)
        if backend is not None:
            backend._executor.shutdown(wait=True)


def table(rows):
    headings = ("Precision", "Mean (s)", "P50 (s)", "P95 (s)", "Images", "Status")
    records = []
    for row in rows:
        metrics = row.get("generation_seconds", {})
        records.append(
            (
                row["precision"],
                *(
                    f"{metrics[key]:.3f}" if key in metrics else "--"
                    for key in ("mean", "p50", "p95")
                ),
                str(row.get("images", "--")),
                row["status"],
            )
        )
    widths = [
        max(len(cell[i]) for cell in [headings, *records]) for i in range(len(headings))
    ]

    def line(values):
        return (
            "| "
            + " | ".join(value.ljust(width) for value, width in zip(values, widths))
            + " |"
        )

    return "\n".join(
        [
            line(headings),
            line(tuple("-" * width for width in widths)),
            *(line(row) for row in records),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-dir",
        type=Path,
        help="Directory containing this branch's flux_image_worker.py",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("flux-generation-results")
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=("bf16", "fp16", "fp8", "fp4"),
        default=["bf16", "fp16", "fp8", "fp4"],
    )
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--steps", type=int, choices=range(1, 51), default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Seconds per precision, including load and warmup",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Use the old worker's optional torch.compile path; default is eager",
    )
    parser.add_argument(
        "--torch-compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--_case", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args._case:
        run_case(args._case)
        return 0
    if min(args.requests, args.warmup, args.timeout) <= 0:
        parser.error("--requests, --warmup and --timeout must be positive")
    try:
        width, height = (int(value) for value in args.size.split("x"))
        if any(side <= 0 or side > 2048 or side % 16 for side in (width, height)):
            raise ValueError
    except ValueError:
        parser.error(
            "--size must be WxH, with sides divisible by 16 and between 16 and 2048"
        )
    candidates = (
        [args.worker_dir]
        if args.worker_dir
        else [Path("deploy"), Path("examples/diffusers/deploy")]
    )
    worker_dir = next(
        (
            path.resolve()
            for path in candidates
            if (path / "flux_image_worker.py").is_file()
        ),
        None,
    )
    if worker_dir is None:
        parser.error("Pass --worker-dir pointing to this branch's deploy directory")
    directory = args.output_dir.resolve() / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    directory.mkdir(parents=True)
    config = {
        **vars(args),
        "worker_dir": str(worker_dir),
        "output_dir": str(directory),
        "_case": None,
    }
    config["source_sha256"] = {
        name: hashlib.sha256((worker_dir / name).read_bytes()).hexdigest()
        for name in ("flux_image_worker.py", "nvfp4_loader.py")
        if (worker_dir / name).is_file()
    }
    write_json(directory / "config.json", config)
    rows = []
    interrupted = False
    print(f"Results: {directory}", flush=True)
    for precision in dict.fromkeys(args.precisions):
        case = directory / precision
        case.mkdir()
        row = {"precision": precision, "status": "SKIP"}
        if precision not in NATIVE_PRECISIONS:
            row["reason"] = (
                "Not implemented by this branch's native worker (supports bf16 and nvfp4 only)"
            )
        else:
            write_json(case / "config.json", {**config, "precision": precision})
            print(f"Benchmarking {precision}; log: {case / 'run.log'}", flush=True)
            failure = None
            with (case / "run.log").open("w") as stream:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--_case",
                        str(case / "config.json"),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={
                        **os.environ,
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )
                try:
                    code = process.wait(timeout=args.timeout)
                    if code:
                        failure = f"Child exited with code {code}; see run.log"
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                    interrupted = isinstance(error, KeyboardInterrupt)
                    failure = (
                        "Interrupted"
                        if interrupted
                        else f"Timeout after {args.timeout}s"
                    )
                    # Kill the entire group, including any compiler subprocesses.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            if (case / "result.json").is_file():
                row = json.loads((case / "result.json").read_text())
            else:
                failure = failure or "Child exited without results; see run.log"
            if failure:
                row.update(status="FAIL", error=failure)
        write_json(case / "result.json", row)
        rows.append(row)
        rendered = table(rows)
        execution = (
            f"torch.compile ({args.torch_compile_mode})"
            if args.torch_compile
            else "Eager"
        )
        details = "\n".join(
            f"- {item['precision']}: {item.get('error') or item['reason']}"
            for item in rows
            if item.get("error") or item.get("reason")
        )
        report = (
            f"Execution: {execution}. Size: {args.size}; steps: {args.steps}; "
            f"warmups: {args.warmup}; measured images: {args.requests}; concurrency: 1.\n\n"
            "Generation time covers the synchronized pipeline call, including PIL conversion. "
            "Model loading, warmup, PNG saving and HTTP transport are excluded. "
            "Percentiles use nearest rank. No AI Tune or precision adapters are used.\n\n"
            f"{rendered}\n\n{details}\n"
        )
        (directory / "results.md").write_text(report)
        write_json(directory / "results.json", {"config": config, "results": rows})
        print(rendered, flush=True)
        if interrupted:
            return 130
    print(f"Saved table: {directory / 'results.md'}", flush=True)
    return int(any(row["status"] == "FAIL" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
