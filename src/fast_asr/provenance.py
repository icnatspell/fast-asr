"""Reproducibility metadata and structural checks for benchmark artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import onnx


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_artifact(model_directory: Path) -> dict[str, int]:
    """Fail fast when encoder/decoder cache contracts disagree."""
    required = ["encoder.onnx", "decoder.onnx", "tokenizer.json"]
    missing = [name for name in required if not (model_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing artifact files: {', '.join(missing)}.")
    encoder = onnx.load(model_directory / "encoder.onnx", load_external_data=False)
    decoder = onnx.load(model_directory / "decoder.onnx", load_external_data=False)
    encoder_caches = {
        output.name.replace("present_", "past_", 1): output
        for output in encoder.graph.output
        if output.name.startswith(("present_key_cross_", "present_value_cross_"))
    }
    decoder_caches = {
        value.name: value
        for value in decoder.graph.input
        if value.name.startswith(("past_key_cross_", "past_value_cross_"))
    }
    if not encoder_caches:
        raise ValueError("Encoder exposes no cross-attention caches.")
    if encoder_caches.keys() != decoder_caches.keys():
        raise ValueError("Encoder outputs and decoder cross-attention inputs do not match.")
    lengths: set[int] = set()
    for name, encoder_value in encoder_caches.items():
        encoder_length = encoder_value.type.tensor_type.shape.dim[2].dim_value
        decoder_length = decoder_caches[name].type.tensor_type.shape.dim[2].dim_value
        if encoder_length != decoder_length:
            raise ValueError(f"Cross-attention length mismatch for {name}.")
        lengths.add(encoder_length)
    if len(lengths) != 1:
        raise ValueError("Cross-attention caches do not share one sequence length.")
    return {"cross_attention_cache_count": len(encoder_caches), "encoder_frames": lengths.pop()}


def write_provenance(
    output_path: Path,
    model_directory: Path,
    *,
    split: str,
    max_samples: int,
    threads: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Write stable inputs, environment, and artifact hashes for one run."""
    validation = validate_model_artifact(model_directory)
    revision = None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    deployment_files = {
        "encoder.onnx",
        "encoder.onnx.data",
        "decoder.onnx",
        "decoder.onnx.data",
        "config.json",
        "generation_config.json",
        "genai_config.json",
        "audio_processor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    model_files = sorted(
        path
        for path in model_directory.iterdir()
        if path.is_file() and path.name in deployment_files
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "git_revision": revision,
        "settings": {
            "split": split,
            "max_samples": max_samples,
            "threads": threads,
            "max_tokens": max_tokens,
            "execution_provider": "CPUExecutionProvider",
        },
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("onnxruntime", "onnx", "numpy", "transformers", "datasets")
        },
        "artifact": {
            "location": str(model_directory.resolve()),
            "files": {path.name: _sha256(path) for path in model_files},
            **validation,
        },
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
