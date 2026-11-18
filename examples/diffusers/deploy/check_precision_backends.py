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
"""Check and benchmark FLUX precisions with AI Tune's default candidates.

Run inside Dockerfile.aitune's image. Each combination gets a fresh process and
AI Tune cache. The worker's actual loading and startup validation code is reused;
diagnostic patches isolate one default candidate without changing serving code.
Heavy dependencies are imported only in child processes, so --help needs no GPU.
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

STAGES = ("load", "compile", "generation", "editing", "validation", "benchmark")
logger = logging.getLogger(__name__)


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _discover(path):
    import importlib.metadata

    import aitune.torch as ait
    import torch
    from flux_precision import check_precision_support

    unsupported = {}
    for precision in ("bf16", "fp16", "fp8", "fp4"):
        try:
            check_precision_support(precision)
        except ValueError as error:
            unsupported[precision] = str(error)
    # AI Tune 0.6.0 exposes resolved defaults through this private property.
    # Keep the library's configurations intact, including same-class variants.
    candidates = ait.MinLatencyStrategy()._backends
    _write_json(
        path,
        {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("aitune", "torch", "torchao", "diffusers", "transformers")
            },
            "unsupported": unsupported,
            "backends": [
                {"id": "eager", "name": "Eager", "description": "Untuned baseline"},
                *[
                    {
                        "id": str(index),
                        "name": f"{index}: {type(backend).__name__}",
                        "description": backend.describe(),
                    }
                    for index, backend in enumerate(candidates, 1)
                ],
            ],
        },
    )


def _run_case(path):
    """Record completed phases even if a later compiler or image check fails."""
    row = json.loads(path.read_text())
    directory = path.parent
    phase = "load"

    def save():
        row["phase"] = phase
        _write_json(path, row)

    try:
        import flux_aitune
        import flux_image_worker as worker
        import numpy as np
        import torch
        from aitune.torch.backend import TorchEagerBackend

        eager = row["backend"]["id"] == "eager"
        argv = [
            "flux_image_worker.py",
            "--precision",
            row["precision"],
            "--default-size",
            row["size"],
            "--default-num-inference-steps",
            str(row["steps"]),
            "--warmup",
            "0",
        ]
        if not eager:
            argv.append("--use-aitune")
        with patch.object(sys, "argv", argv):
            args = worker._parse_args()
        backend = worker.FluxImageBackend(args)
        width, height = backend._parse_size(args.default_size)
        real_strategy = flux_aitune.ait.MinLatencyStrategy
        real_tune = flux_aitune.tune_transformer
        real_selection = flux_aitune._validate_backend_selection
        real_validate = flux_aitune._validate_image
        checked = 0

        def strategy_factory(**kwargs):
            strategy = real_strategy(**kwargs)
            candidate = strategy._backends[int(row["backend"]["id"]) - 1]
            if candidate.describe() != row["backend"]["description"]:
                raise RuntimeError(
                    "Default backend configuration changed since discovery"
                )
            strategy._backends_override = [candidate]
            return strategy

        def tune(pipe, **kwargs):
            nonlocal phase
            row["load"] = "PASS"
            phase = "compile"
            save()
            return real_tune(pipe, **kwargs)

        def selection(transformer, strategy):
            nonlocal phase
            selected = real_selection(transformer, strategy)
            row["candidate_results"] = strategy.backend_results
            if any(isinstance(item, TorchEagerBackend) for item in selected):
                raise RuntimeError("Candidate failed; AI Tune selected eager fallback")
            # Selection follows build, correctness checks and profiling. This
            # also exercises lazy JIT compilation, unlike build() alone.
            row["compile"] = "PASS"
            phase = "generation"
            save()
            return selected

        def validate(actual, expected):
            nonlocal checked, phase
            label = ("generation", "editing", "held-out")[checked]
            actual.save(directory / f"{label}.png")
            if checked < 2:
                row[label] = "PASS"
            phase = "validation"
            save()
            try:
                real_validate(actual, expected)
            except RuntimeError as error:
                # Image agreement and execution speed are independent results.
                # Keep the compiled candidate for diagnostic benchmarking.
                row["validation"] = "FAIL"
                row.setdefault("validation_errors", []).append(
                    {"case": label, "error": str(error)}
                )
                logger.warning(
                    "%s image validation failed: %s; continuing diagnostic checks and benchmark",
                    label,
                    error,
                )
            checked += 1
            if checked == 3 and row["validation"] != "FAIL":
                row["validation"] = "PASS"
            phase = "editing" if checked == 1 else "validation"
            save()

        try:
            with (
                patch.object(flux_aitune.ait, "MinLatencyStrategy", strategy_factory),
                patch.object(flux_aitune, "tune_transformer", tune),
                patch.object(flux_aitune, "_validate_backend_selection", selection),
                patch.object(flux_aitune, "_validate_image", validate),
            ):
                backend.pipe = backend._load()
            if eager:
                row["load"] = "PASS"
                reference = None
                for phase, prompt in (
                    ("generation", "A red fox in a sunny alpine meadow"),
                    ("editing", "Turn this image into a watercolor painting"),
                ):
                    save()
                    with torch.inference_mode():
                        result = backend._run(
                            prompt,
                            reference,
                            width,
                            height,
                            args.default_num_inference_steps,
                            args.default_guidance_scale,
                            42,
                        )
                    pixels = np.asarray(result)
                    if result.size != (width, height) or pixels.std() == 0:
                        raise RuntimeError(
                            "Image has an incorrect size or is uniformly colored"
                        )
                    result.save(directory / f"{phase}.png")
                    row[phase] = "PASS"
                    reference = result
            if row["benchmark_options"]["requests"]:
                from flux_benchmark import benchmark_dynamo

                phase = "benchmark"
                save()
                row["benchmark_metrics"] = benchmark_dynamo(
                    backend, directory, row["benchmark_options"]
                )
                row["benchmark"] = "PASS"
            else:
                row["benchmark"] = "DISABLED"
            row["status"] = "FAIL" if row["validation"] == "FAIL" else "PASS"
        finally:
            backend._executor.shutdown(wait=True)
    except Exception as error:  # noqa: BLE001 - record failures from optional compilers
        row[phase] = "FAIL"
        row["status"] = "FAIL"
        row["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        save()


def _subprocess(arguments, log, timeout, env=None):
    """Bound a child and its compiler subprocesses; preserve the full log."""
    with log.open("w") as stream:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *arguments],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            # The parent can exit before a compiler subprocess. Kill any
            # remaining group members before allowing the next GPU workload.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def _table(columns, records):
    cells = [columns, *records]
    widths = [max(len(cell[index]) for cell in cells) for index in range(len(columns))]

    def line(cell):
        return (
            "| "
            + " | ".join(value.ljust(width) for value, width in zip(cell, widths))
            + " |"
        )

    return "\n".join(
        [
            line(columns),
            line(tuple("-" * width for width in widths)),
            *(line(cell) for cell in records),
        ]
    )


def _metric_tables(rows):
    tables = []
    for title, metric in (
        ("HTTP latency (ms)", "latency_s"),
        ("Worker generation time (ms)", "generation_s"),
        ("Throughput (images/s, concurrency 1)", "throughput_images_s"),
    ):
        throughput = metric == "throughput_images_s"
        headings = (
            ("Images/s", "Images", "Elapsed (s)")
            if throughput
            else ("Mean", "P50", "P95")
        )
        records = []
        for row in rows:
            values = ("--", "--", "--")
            if row["benchmark"] == "PASS":
                result = row["benchmark_metrics"]
                values = (
                    (
                        f"{result[metric]:.3f}",
                        str(result["completed_requests"]),
                        f"{result['elapsed_s']:.3f}",
                    )
                    if throughput
                    else tuple(
                        f"{result[metric][stat] * 1000:.2f}"
                        for stat in ("mean", "p50", "p95")
                    )
                )
            records.append(
                (
                    row["precision"],
                    row["backend"]["name"],
                    *values,
                    row["validation"],
                    row["benchmark"],
                )
            )
        tables.append(
            f"### {title}\n\n"
            + _table(
                ("Precision", "Backend", *headings, "Validation", "Benchmark"), records
            )
        )
    return "\n\n".join(tables)


def _report(directory, metadata, rows):
    columns = (
        "Precision",
        "Backend",
        "Compile",
        "Generation",
        "Editing",
        "Validation",
        "Benchmark",
        "Result",
    )
    records = [
        (
            row["precision"],
            row["backend"]["name"],
            row["compile"],
            row["generation"],
            row["editing"],
            row["validation"],
            row["benchmark"],
            row["status"],
        )
        for row in rows
    ]
    table = _table(columns, records)
    metrics = _metric_tables(rows)
    details = "\n".join(
        f"- {row['precision']} / {row['backend']['name']}: {row['error']}"
        + (f" (log: {row['directory']}/run.log)" if row["status"] != "SKIP" else "")
        for row in rows
        if row.get("error")
    )
    validation_details = "\n".join(
        f"- {row['precision']} / {row['backend']['name']} / {failure['case']}: {failure['error']}"
        for row in rows
        for failure in row.get("validation_errors", [])
    )
    report = (
        f"GPU: {metadata['gpu']}\n\n"
        f"Workload: {rows[0]['size']}, {rows[0]['steps']} steps, batch size 1.\n\n"
        f"{table}\n\n"
        "Compile includes AI Tune build, checks, and profiling. Eager fallback "
        "does not count as success. Validation compares images with eager at "
        "the same precision; it is not a comparison with BF16.\n\n"
        f"{metrics}\n\n"
        "Benchmarks use serial text-to-image HTTP requests. Startup and warmups "
        "are excluded. Generation time excludes PNG encoding and transport; "
        "latency includes the full response transfer. Throughput includes client "
        "work between requests and is not a maximum-concurrency measurement.\n\n"
        "Image validation failures do not block diagnostic benchmarks; consult "
        "the Validation column before interpreting performance results.\n\n"
        f"{details}\n{validation_details}\n"
    )
    (directory / "results.md").write_text(report)
    _write_json(directory / "results.json", {"environment": metadata, "results": rows})
    print(table, flush=True)
    print("\n" + metrics, flush=True)


def main():
    """Print and save a sequential precision/backend matrix; failures return 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("precision-results"))
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=("bf16", "fp16", "fp8", "fp4"),
        default=["bf16", "fp16", "fp8", "fp4"],
    )
    parser.add_argument(
        "--backends", nargs="+", help="IDs from --list-backends; default: all"
    )
    parser.add_argument("--list-backends", action="store_true")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--steps", type=int, choices=range(1, 51), default=4)
    parser.add_argument(
        "--benchmark-requests",
        type=int,
        default=10,
        help="Measured HTTP requests per passing configuration; 0 disables benchmarking",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=2,
        help="Unmeasured HTTP warmup requests",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for each benchmark HTTP request",
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Seconds per combination"
    )
    parser.add_argument("--_discover", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_case", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args._discover:
        _discover(args._discover)
        return 0
    if args._case:
        _run_case(args._case)
        return 0
    try:
        width, height = (int(part) for part in args.size.split("x"))
        if any(side <= 0 or side > 2048 or side % 16 for side in (width, height)):
            raise ValueError
    except ValueError:
        parser.error(
            "--size must be WxH, with sides divisible by 16 and between 16 and 2048"
        )
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.benchmark_requests < 0 or args.benchmark_warmup < 0:
        parser.error("Benchmark request and warmup counts must be nonnegative")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    directory = args.output_dir.resolve() / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    directory.mkdir(parents=True)
    print(f"Results: {directory}", flush=True)
    environment = directory / "environment.json"
    try:
        code = _subprocess(
            ["--_discover", str(environment)], directory / "discovery.log", 120
        )
    except subprocess.TimeoutExpired:
        parser.exit(
            2, f"Backend discovery timed out; see {directory / 'discovery.log'}\n"
        )
    except KeyboardInterrupt:
        return 130
    if code:
        parser.exit(2, f"Backend discovery failed; see {directory / 'discovery.log'}\n")
    metadata = json.loads(environment.read_text())
    if args.list_backends:
        for backend in metadata["backends"]:
            print(f"{backend['id']}: {backend['description']}")
        return 0
    if args.backends and set(args.backends) - {
        item["id"] for item in metadata["backends"]
    }:
        parser.error("Unknown backend ID; use --list-backends")
    rows = []
    for precision in dict.fromkeys(args.precisions):
        for backend in metadata["backends"]:
            if args.backends and backend["id"] not in args.backends:
                continue
            case_dir = directory / f"{precision}-{backend['id']}"
            case_dir.mkdir()
            row = dict.fromkeys(STAGES, "NOT RUN")
            row.update(
                precision=precision,
                backend=backend,
                size=args.size,
                steps=args.steps,
                benchmark_options={
                    "size": args.size,
                    "steps": args.steps,
                    "requests": args.benchmark_requests,
                    "warmup": args.benchmark_warmup,
                    "request_timeout": args.request_timeout,
                },
                directory=case_dir.name,
                status="PENDING",
                phase="load",
            )
            if backend["id"] == "eager":
                row.update(compile="N/A", validation="N/A")
            case_path = case_dir / "result.json"
            _write_json(case_path, row)
            rows.append(row)
            if precision in metadata["unsupported"]:
                row.update(dict.fromkeys(STAGES, "SKIP"))
                row.update(status="SKIP", error=metadata["unsupported"][precision])
            else:
                print(
                    f"Checking {precision} / {backend['name']} (log: {case_dir / 'run.log'})",
                    flush=True,
                )
                started = time.monotonic()
                interrupted = False
                failure = None
                try:
                    code = _subprocess(
                        ["--_case", str(case_path)],
                        case_dir / "run.log",
                        args.timeout,
                        env={
                            **os.environ,
                            "AITUNE_CACHE_DIR": str(case_dir / "aitune"),
                            "PYTHONUNBUFFERED": "1",
                            "DYN_DISCOVERY_BACKEND": "file",
                            "DYN_EVENT_PLANE": "zmq",
                            "DYN_FILE_KV": str(case_dir / "discovery"),
                            "DYN_NAMESPACE": "flux-bench-" + uuid.uuid4().hex,
                            "DYN_SYSTEM_PORT": "-1",
                        },
                    )
                    if code:
                        failure = f"Child exited with code {code}"
                except subprocess.TimeoutExpired:
                    failure = f"TIMEOUT after {args.timeout} seconds"
                except KeyboardInterrupt:
                    failure, interrupted = "INTERRUPTED", True
                row.update(json.loads(case_path.read_text()))
                row["seconds"] = round(time.monotonic() - started, 1)
                if failure or row["status"] == "PENDING":
                    row.update(status="FAIL", error=failure or "Child did not finish")
                    row[row["phase"]] = "FAIL"
                if interrupted:
                    _write_json(case_path, row)
                    _report(directory, metadata, rows)
                    return 130
            _write_json(case_path, row)
            _report(directory, metadata, rows)
    print(f"Saved table: {directory / 'results.md'}", flush=True)
    return int(any(row["status"] == "FAIL" for row in rows))


if __name__ == "__main__":
    sys.exit(main())
