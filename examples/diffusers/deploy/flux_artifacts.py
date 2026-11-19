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
"""Save and load versioned AI Tune artifacts; loading never invokes tuning."""

import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
PACKAGES = ("aitune", "torch", "torchao", "diffusers", "transformers")


def write_json(path: Path, value: dict) -> None:
    """Publish a complete JSON file, preserving the old file on write failure."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def environment() -> dict:
    """Return the runtime versions and GPU target required by this artifact."""
    import torch

    return {
        "versions": {name: importlib.metadata.version(name) for name in PACKAGES},
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
    }


def _inside(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Artifact path escapes its directory: {relative}")
    return path


def read_manifest(directory: Path) -> dict:
    """Read a completed artifact manifest without importing AI Tune or Torch."""
    path = directory / "manifest.json"
    if not path.is_file():
        raise ValueError(
            f"No usable AI Tune artifact: missing {path}. Run tune_flux.py first."
        )
    manifest = json.loads(path.read_text())
    required = {
        "schema_version",
        "model",
        "precision",
        "workload",
        "environment",
        "checkpoint",
        "files",
        "backends",
        "validation",
    }
    if (
        not isinstance(manifest, dict)
        or required - manifest.keys()
        or manifest["schema_version"] != 1
    ):
        raise ValueError(f"Unsupported or incomplete artifact manifest: {path}")
    if not manifest["files"] or not manifest["backends"]:
        raise ValueError(f"Empty artifact: {path}")
    if manifest["precision"] not in ("bf16", "fp16", "fp8", "fp4"):
        raise ValueError(f"Invalid artifact precision in {path}")
    if manifest["validation"] not in ("PASS", "FAIL"):
        raise ValueError(f"Artifact image checks are incomplete: {path}")
    if not _inside(directory, manifest["checkpoint"]).is_dir():
        raise ValueError(f"Missing checkpoint directory in {directory}")
    return manifest


def validate_artifact(directory: Path, manifest: dict) -> None:
    """Reject incompatible runtime/GPU targets and missing or corrupted files."""
    actual = environment()
    if manifest["environment"] != actual:
        raise ValueError(
            f"AI Tune artifact environment mismatch. Saved: {manifest['environment']}; current: {actual}. Retune on this runtime/GPU."
        )
    for relative, expected in manifest["files"].items():
        path = _inside(directory, relative)
        if not path.is_file():
            raise ValueError(f"Missing artifact file: {path}")
        with path.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_hash != expected:
            raise ValueError(f"Corrupted artifact file: {path}")


def selected_backends(transformer) -> list[dict]:
    """Describe the actual selected backends, rejecting eager fallback."""
    from aitune.torch.backend import TorchEagerBackend
    from aitune.torch.module.wrapper_module import Module, ModuleState

    if not isinstance(transformer, Module) or transformer.state != ModuleState.TUNED:
        raise RuntimeError("Artifact does not contain a tuned transformer")
    backends = list(transformer.module.backends.values())
    if not backends or any(
        isinstance(backend, TorchEagerBackend) for backend in backends
    ):
        raise RuntimeError(
            "No usable optimized artifact: AI Tune selected eager fallback"
        )
    return [
        {
            "name": type(backend).__name__,
            "description": backend.describe(),
            "build_mode": backend.build_mode.name,
        }
        for backend in backends
    ]


def save_artifact(pipe, directory: Path, run_directory: Path, metadata: dict) -> dict:
    """Save a relocatable checkpoint; the caller publishes its manifest last."""
    import aitune.torch as ait

    backends = selected_backends(pipe.transformer)
    storage = ait.LocalTorchStorage(
        base_folder=run_directory,
        compress_checkpoint=False,
        overwrite_checkpoint_on_tune=False,
    )
    ait.save(pipe.transformer, "transformer", storage=storage)
    checkpoint = run_directory / "transformer"
    files = {}
    for path in sorted(checkpoint.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                files[str(path.relative_to(directory))] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    if not files:
        raise RuntimeError("AI Tune did not save a checkpoint")
    return {
        **metadata,
        "schema_version": 1,
        "environment": environment(),
        "checkpoint": str(checkpoint.relative_to(directory)),
        "files": files,
        "backends": backends,
    }


def load_artifact(pipe, directory: Path, manifest: dict) -> None:
    """Install the saved transformer. AoT loads compiled code; JIT reactivates JIT.

    The caller validates integrity before loading model weights. Original precision
    hooks must already be installed on the pipeline. No tuning or eager fallback
    is allowed here; deserialization/activation failures propagate to startup.
    """
    import aitune.torch as ait
    import torch
    from flux_aitune import _contiguous_inputs

    pipe.transformer.register_forward_pre_hook(_contiguous_inputs, with_kwargs=True)
    pipe.transformer = ait.load(
        pipe.transformer,
        manifest["checkpoint"],
        storage=ait.LocalTorchStorage(base_folder=directory, compress_checkpoint=False),
        device_map={"": torch.device("cuda:0")},
    )
    backends = selected_backends(pipe.transformer)
    if backends != manifest["backends"]:
        raise RuntimeError(
            "Loaded backend selection does not match the artifact manifest"
        )
    if manifest["validation"] == "FAIL":
        logger.warning(
            "Artifact image validation failed during tuning: %s",
            manifest.get("validation_errors", []),
        )
    logger.info("Loaded AI Tune artifact from %s: %s", directory, backends)
