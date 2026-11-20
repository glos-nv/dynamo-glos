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

from tune_flux import DEFAULT_MIN_SSIM

STAGES = (
    "load",
    "compile",
    "generation",
    "editing",
    "validation",
    "artifact",
    "startup",
    "benchmark",
)
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
                {
                    "id": "eager",
                    "name": "Eager",
                    "description": "Untuned diagnostic baseline",
                },
                {
                    "id": "torch_compile",
                    "name": "torch.compile (no AI Tune)",
                    "description": "Default server without artifacts",
                },
                *[
                    {
                        "id": str(index),
                        "name": f"{index}: {type(backend).__name__}",
                        "description": backend.describe(),
                        "build_mode": backend.build_mode.name,
                    }
                    for index, backend in enumerate(candidates, 1)
                ],
            ],
        },
    )


def _run_tune_case(path):
    """Call the standalone tuner's shared implementation for one candidate.

    Only candidate selection and report destination differ from normal tuning.
    Exit this process before starting the server, which loads the saved artifact.
    """
    from tune_flux import run_tuning

    row = json.loads(path.read_text())
    run_tuning(
        artifact_dir=row["artifact_dir"],
        precision=row["precision"],
        size=row["size"],
        steps=row["steps"],
        backend_id=row["backend"]["id"],
        min_ssim=row["min_ssim"],
        report=row,
        report_path=path,
        started=float(os.environ["FLUX_PROCESS_STARTED_AT"]),
    )


def _run_serve_case(path):
    """Measure a new server process, never reusing a pipeline from tuning."""
    row = json.loads(path.read_text())
    row.update(phase="startup", status="PENDING")
    _write_json(path, row)
    try:
        from flux_benchmark import benchmark_dynamo
        from flux_image_worker import FluxImageBackend, _parse_args

        argv = [
            "--precision",
            row["precision"],
            "--default-size",
            row["size"],
            "--default-num-inference-steps",
            str(row["steps"]),
        ]
        if row["server_mode"] == "aitune":
            argv.extend(["--artifact-dir", row["artifact_dir"]])
        backend = FluxImageBackend(_parse_args(argv))
        if row["backend"]["id"] == "eager":
            # Explicit diagnostic baseline only; the public server defaults to JIT.
            backend._load = backend._load_pipeline
            backend.execution_mode = "eager (diagnostic)"

        def ready(startup_seconds, timings):
            row.update(startup="PASS", startup_seconds=startup_seconds, **timings)
            if row["server_mode"] != "aitune":
                row.update(load="PASS", generation="PASS", editing="PASS")
                row["compile"] = (
                    "PASS" if row["backend"]["id"] == "torch_compile" else "N/A"
                )
            row["phase"] = "benchmark"
            _write_json(path, row)

        row["benchmark_metrics"] = benchmark_dynamo(
            backend,
            path.parent,
            row["benchmark_options"],
            startup_started=float(os.environ["FLUX_PROCESS_STARTED_AT"]),
            ready_callback=ready,
        )
        row["benchmark"] = "PASS"
        row["status"] = "FAIL" if row["validation"] == "FAIL" else "PASS"
    except Exception as error:  # noqa: BLE001 - persist subprocess diagnostics
        row[row["phase"]] = "FAIL"
        row.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        _write_json(path, row)


def _subprocess(arguments, log, timeout, env=None):
    """Bound a child and its compiler subprocesses; preserve the full log."""
    with log.open("w") as stream:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *arguments],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env={
                **(env if env is not None else os.environ),
                "FLUX_PROCESS_STARTED_AT": str(time.perf_counter()),
            },
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


def _precision_backend_table(rows, values):
    """Pivot formatted values into precision rows and distinct backend columns."""
    precisions = dict.fromkeys(row["precision"] for row in rows)
    backends = {row["backend"]["id"]: row["backend"]["name"] for row in rows}
    cells = {
        (row["precision"], row["backend"]["id"]): value
        for row, value in zip(rows, values, strict=True)
    }
    return _table(
        ("Precision", *backends.values()),
        [
            (
                precision,
                *(cells.get((precision, backend), "--") for backend in backends),
            )
            for precision in precisions
        ],
    )


def _metric_tables(rows):
    """Render detailed benchmarks and precision/backend summaries of their means."""
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
        summary_title = title if throughput else f"{title}, mean"
        tables.append(
            f"### {summary_title} — precision/backend summary\n\n"
            + _precision_backend_table(rows, [record[2] for record in records])
        )
    return "\n\n".join(tables)


def _startup_table(rows):
    """Keep server selection separate from the backend's compilation lifecycle."""

    def seconds(row, key):
        return f"{row[key]:.3f}" if key in row else "--"

    records = []
    for row in rows:
        if row["backend"]["id"] == "eager":
            server_mode, compilation_type = "Diagnostic eager", "None"
        elif row["server_mode"] == "no-aitune":
            server_mode, compilation_type = "No AI Tune", "JIT"
        else:
            server_mode = "AI Tune"
            # Discovery records the candidate type even when its build fails.
            backends = row.get("backends") or [row["backend"]]
            build_modes = sorted(
                {
                    backend["build_mode"]
                    for backend in backends
                    if "build_mode" in backend
                }
            )
            labels = {"AHEAD_OF_TIME": "AoT", "JUST_IN_TIME": "JIT"}
            compilation_type = (
                "/".join(labels.get(mode, mode) for mode in build_modes) or "--"
            )
        records.append(
            (
                row["precision"],
                row["backend"]["name"],
                server_mode,
                compilation_type,
                seconds(row, "tuning_seconds"),
                seconds(row, "compilation_seconds"),
                seconds(row, "model_load_seconds"),
                seconds(row, "warmup_seconds"),
                seconds(row, "startup_compilation_seconds"),
                seconds(row, "startup_seconds"),
                row["startup"],
            )
        )
    return "### Tuning and server startup (seconds)\n\n" + _table(
        (
            "Precision",
            "Backend",
            "Server mode",
            "Compilation type",
            "Overall tuning",
            "Backend build",
            "Model load",
            "Warmup",
            "Startup compile",
            "Server startup",
            "Startup",
        ),
        records,
    )


def _clip_table(rows):
    """Show prompt alignment separately from SSIM and execution success."""
    records = []
    for row in rows:
        result = row.get("clip_score", {})
        scores = result.get("cases", {})
        for label, values in scores.items():
            records.append(
                (
                    row["precision"],
                    row["backend"]["name"],
                    label,
                    f"{values['eager']:.4f}",
                    f"{values['tuned']:.4f}",
                    f"{values['delta']:+.4f}",
                    "OK",
                )
            )
        if not scores or result.get("status") == "ERROR":
            records.append(
                (
                    row["precision"],
                    row["backend"]["name"],
                    "--",
                    "--",
                    "--",
                    "--",
                    result.get("status", "NOT RUN"),
                )
            )
    return "### CLIPScore prompt alignment (0–100, diagnostic only)\n\n" + _table(
        ("Precision", "Backend", "Case", "Eager", "Tuned", "Delta", "Scoring"),
        records,
    )


def _report(directory, metadata, rows):
    def lowest_ssim(row):
        scores = row.get("image_validation", {}).get("scores", {})
        return f"{min(scores.values()):.6f}" if scores else "--"

    columns = (
        "Precision",
        "Backend",
        "Compile",
        "Generation",
        "Editing",
        "Validation",
        "Lowest SSIM",
        "Artifact",
        "Startup",
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
            lowest_ssim(row),
            row["artifact"],
            row["startup"],
            row["benchmark"],
            row["status"],
        )
        for row in rows
    ]
    table = _table(columns, records)
    metrics = _metric_tables(rows)
    startup = _startup_table(rows)
    clip = _clip_table(rows)
    details = "\n".join(
        f"- {row['precision']} / {row['backend']['name']}: {row['error']}"
        + (
            f" (logs: {row['directory']}/tune.log, {row['directory']}/serve.log)"
            if row["status"] != "SKIP"
            else ""
        )
        for row in rows
        if row.get("error")
    )
    validation_details = "\n".join(
        f"- {row['precision']} / {row['backend']['name']} / {failure['case']}: {failure['error']}"
        for row in rows
        for failure in row.get("validation_errors", [])
    )
    clip_errors = "\n".join(
        f"- {row['precision']} / {row['backend']['name']} / CLIPScore: {row['clip_score']['error']}"
        for row in rows
        if row.get("clip_score", {}).get("error")
    )
    report = (
        f"GPU: {metadata['gpu']}\n\n"
        f"Workload: {rows[0]['size']}, {rows[0]['steps']} steps, batch size 1.\n\n"
        f"Total recorded offline tuning: {sum(row.get('tuning_seconds', 0) for row in rows):.3f} seconds.\n\n"
        f"{table}\n\n"
        "Compile includes AI Tune build, checks, and profiling. Eager fallback "
        "does not count as success. Validation compares images with eager at "
        "the same precision; it is not a comparison with BF16.\n\n"
        f"Image agreement requires RGB SSIM >= {rows[0]['min_ssim']:.6f} for every case. "
        "Lowest SSIM is the minimum measured generation, editing, or held-out score. "
        "This is a similarity check, not a general image-quality assessment.\n\n"
        f"{clip}\n\n"
        "CLIPScore is diagnostic prompt alignment; delta is tuned minus eager. "
        "Scores use a fixed CLIP checkpoint on CPU during offline tuning, outside "
        "backend profiling and server benchmarks. Editing scores use the edit instruction, "
        "which is not a full description of the image. No CLIPScore threshold is applied.\n\n"
        f"{metrics}\n\n"
        f"{startup}\n\n"
        f"Compiler cache policy: {rows[0]['compiler_cache']}. Startup is measured in a fresh process "
        "from launch to model visibility in /v1/models, after generation/editing warmup. "
        "AoT loads compiled artifacts; JIT artifacts and no-AI-Tune servers compile on startup. "
        "Backend build measures AI Tune's build/export calls, including failed attempts. "
        "Startup compile is PyTorch-reported frame compilation time within startup. "
        "Warmup includes inference and any JIT compilation; it is not compiler-only time.\n\n"
        "Benchmarks use serial text-to-image HTTP requests. Startup and warmups "
        "are excluded. Generation time excludes PNG encoding and transport; "
        "latency includes the full response transfer. Throughput includes client "
        "work between requests and is not a maximum-concurrency measurement.\n\n"
        "Image validation failures do not block diagnostic benchmarks; consult "
        "the Validation column before interpreting performance results.\n\n"
        f"{details}\n{validation_details}\n{clip_errors}\n"
    )
    (directory / "results.md").write_text(report)
    _write_json(directory / "results.json", {"environment": metadata, "results": rows})
    print(table, flush=True)
    print("\n" + clip, flush=True)
    print("\n" + metrics, flush=True)
    print("\n" + startup, flush=True)


def main():
    """Print and save a sequential precision/backend matrix; failures return 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("precision-results"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("flux-artifacts"),
        help="Root for persistent artifacts, grouped by run and configuration",
    )
    parser.add_argument(
        "--compiler-cache",
        choices=("cold", "warm"),
        default="cold",
        help="Cold isolates compiler caches for each tuning and server process; warm reuses environment caches",
    )
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
        "--min-ssim",
        type=float,
        default=DEFAULT_MIN_SSIM,
        help="Minimum RGB SSIM for image agreement (default: %(default)s)",
    )
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
    parser.add_argument("--_tune-case", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_serve-case", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 <= args.min_ssim <= 1:
        parser.error("--min-ssim must be between 0 and 1")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args._discover:
        _discover(args._discover)
        return 0
    if args._tune_case:
        _run_tune_case(args._tune_case)
        return 0
    if args._serve_case:
        _run_serve_case(args._serve_case)
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
                min_ssim=args.min_ssim,
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
            row["server_mode"] = (
                "aitune"
                if backend["id"] not in ("eager", "torch_compile")
                else "no-aitune"
            )
            row["compiler_cache"] = args.compiler_cache
            row["artifact_dir"] = str(
                args.artifact_dir.resolve() / directory.name / case_dir.name
            )
            if row["server_mode"] != "aitune":
                row.update(artifact="N/A", validation="N/A")
            if backend["id"] == "eager":
                row["compile"] = "N/A"
            case_path = case_dir / "result.json"
            _write_json(case_path, row)
            rows.append(row)
            if precision in metadata["unsupported"]:
                row.update(dict.fromkeys(STAGES, "SKIP"))
                row.update(status="SKIP", error=metadata["unsupported"][precision])
            else:
                print(
                    f"Checking {precision} / {backend['name']} (logs: {case_dir})",
                    flush=True,
                )
                started = time.monotonic()
                interrupted = False
                failure = None
                try:
                    env = {
                        **os.environ,
                        "PYTHONUNBUFFERED": "1",
                        "DYN_DISCOVERY_BACKEND": "file",
                        "DYN_EVENT_PLANE": "zmq",
                        "DYN_FILE_KV": str(case_dir / "discovery"),
                        "DYN_NAMESPACE": "flux-bench-" + uuid.uuid4().hex,
                        "DYN_SYSTEM_PORT": "-1",
                    }
                    phases = (
                        ["tune", "serve"]
                        if row["server_mode"] == "aitune"
                        else ["serve"]
                    )
                    for phase in phases:
                        if phase == "serve" and not args.benchmark_requests:
                            row["benchmark"] = "DISABLED"
                            if row["status"] == "PENDING":
                                row["status"] = "PASS"
                            _write_json(case_path, row)
                            break
                        phase_env = dict(env)
                        if args.compiler_cache == "cold":
                            phase_env.update(
                                TORCHINDUCTOR_CACHE_DIR=str(
                                    case_dir / phase / "inductor"
                                ),
                                TRITON_CACHE_DIR=str(case_dir / phase / "triton"),
                            )
                        phase_started = time.monotonic()
                        remaining = args.timeout - (phase_started - started)
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(phase, args.timeout)
                        code = _subprocess(
                            [f"--_{phase}-case", str(case_path)],
                            case_dir / f"{phase}.log",
                            remaining,
                            env=phase_env,
                        )
                        row.update(json.loads(case_path.read_text()))
                        if phase == "tune":
                            row["tuning_seconds"] = time.monotonic() - phase_started
                            _write_json(case_path, row)
                        if code:
                            failure = f"{phase} process exited with code {code}"
                            break
                        if phase == "tune" and not row.get("artifact_saved"):
                            failure = "Tuning did not publish an artifact"
                            break

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
