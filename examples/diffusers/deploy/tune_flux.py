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
"""Always tune FLUX offline, then publish an artifact for load-only serving."""

import argparse
import logging
import os
import time
import uuid
from pathlib import Path

from flux_artifacts import save_artifact, write_json

logger = logging.getLogger(__name__)
DEFAULT_MIN_SSIM = 0.90


def run_tuning(
    *,
    artifact_dir,
    precision,
    size,
    steps,
    model=None,
    backend_id=None,
    max_image_side=2048,
    min_ssim=DEFAULT_MIN_SSIM,
    report=None,
    report_path=None,
    started=None,
):
    """Run a new tuning attempt even if a previous artifact exists.

    Shared by the standalone tuning CLI and the precision/backend benchmark.
    With no backend_id, select among all AI Tune defaults; the benchmark passes
    one candidate ID per case. Both produce the same load-only serving artifact.
    Each attempt has its own build directory. The root manifest is published
    only after checkpoint saving succeeds, so an unsuccessful attempt leaves
    an existing artifact intact. Image mismatches are warnings in the artifact.
    Overall tuning time includes loading, recording, backend builds/profiling,
    SSIM/CLIP image checks, and checkpoint serialization/integrity indexing.
    """
    if not 0 <= min_ssim <= 1:
        raise ValueError("min_ssim must be between 0 and 1")
    started = time.perf_counter() if started is None else started
    artifact_dir = Path(artifact_dir).resolve()
    run_directory = artifact_dir / "runs" / uuid.uuid4().hex
    run_directory.mkdir(parents=True)
    os.environ["AITUNE_CACHE_DIR"] = str(run_directory / "build")
    report = {} if report is None else report
    report_path = Path(report_path) if report_path else run_directory / "tuning.json"
    report.update(phase="load", artifact_dir=str(artifact_dir), artifact_saved=False)

    def progress():
        write_json(report_path, report)

    progress()
    backend = None
    try:
        from flux_aitune import tune_transformer
        from flux_image_worker import SIZE_MULTIPLE, FluxImageBackend, _parse_args

        argv = [
            "--precision",
            precision,
            "--default-size",
            size,
            "--default-num-inference-steps",
            str(steps),
            "--max-image-side",
            str(max_image_side),
        ]
        if model:
            argv.extend(["--model", model])
        args = _parse_args(argv)
        backend = FluxImageBackend(args)
        width, height = backend._parse_size(size)
        # Offline tuning starts with eager precision conversion, never torch.compile.
        backend.pipe = backend._load_pipeline()
        report.update(load="PASS", phase="compile")
        progress()
        tune_transformer(
            backend.pipe,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=args.default_guidance_scale,
            max_image_side=max_image_side,
            size_multiple=SIZE_MULTIPLE,
            report=report,
            progress=progress,
            output_dir=run_directory,
            backend_id=backend_id,
            min_ssim=min_ssim,
        )
        report["phase"] = "artifact"
        progress()
        manifest = save_artifact(
            backend.pipe,
            artifact_dir,
            run_directory,
            {
                "model": args.model,
                "precision": "fp4" if precision == "nvfp4" else precision,
                "workload": {
                    "size": size,
                    "steps": steps,
                    "max_image_side": max_image_side,
                },
                "validation": report["validation"],
                "validation_errors": report.get("validation_errors", []),
                "image_validation": report["image_validation"],
                "clip_score": report["clip_score"],
                "backend_builds": report["backend_builds"],
                "compilation_seconds": report.get("compilation_seconds", 0),
            },
        )
        report["tuning_seconds"] = time.perf_counter() - started
        manifest["tuning_seconds"] = report["tuning_seconds"]
        write_json(artifact_dir / "manifest.json", manifest)
        report.update(
            artifact="PASS",
            artifact_saved=True,
            backends=manifest["backends"],
            status="FAIL" if report["validation"] == "FAIL" else "PASS",
        )
        logger.info(
            "Saved artifact: %s; overall tuning %.3fs; backend builds %.3fs",
            artifact_dir,
            report["tuning_seconds"],
            report.get("compilation_seconds", 0),
        )
        return report
    except Exception as error:
        report[report["phase"]] = "FAIL"
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        logger.exception("Offline tuning failed; no new artifact was published")
        raise
    finally:
        report["tuning_seconds"] = time.perf_counter() - started
        progress()
        if backend is not None:
            backend._executor.shutdown(wait=True)


def main():
    """Create an artifact directory; never load an old artifact to skip tuning."""
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir", type=Path, help="Default: flux-artifacts/<precision>"
    )
    parser.add_argument(
        "--precision", choices=("bf16", "fp16", "fp8", "fp4", "nvfp4"), default="bf16"
    )
    parser.add_argument("--model")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-image-side", type=int, default=2048)
    parser.add_argument(
        "--min-ssim",
        type=float,
        default=DEFAULT_MIN_SSIM,
        help="Minimum RGB SSIM for image agreement (default: %(default)s)",
    )
    parser.add_argument(
        "--backend",
        help="Optional default-candidate ID from check_precision_backends.py --list-backends",
    )
    args = parser.parse_args()
    if not 0 <= args.min_ssim <= 1:
        parser.error("--min-ssim must be between 0 and 1")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    precision = "fp4" if args.precision == "nvfp4" else args.precision
    report = run_tuning(
        artifact_dir=args.artifact_dir or Path("flux-artifacts") / precision,
        precision=precision,
        size=args.size,
        steps=args.steps,
        model=args.model,
        backend_id=args.backend,
        max_image_side=args.max_image_side,
        min_ssim=args.min_ssim,
        started=started,
    )
    return int(report["status"] == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
