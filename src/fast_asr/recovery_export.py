"""Validated Olive export workflows for merged recovery checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx


def load_recovery_config(checkpoint_directory: Path) -> dict[str, Any]:
    """Load and validate architecture metadata needed for reproducible export."""
    model_directory = checkpoint_directory / "merged-model"
    config_path = checkpoint_directory / "recovery_config.json"
    if not model_directory.is_dir() or not (model_directory / "config.json").is_file():
        raise FileNotFoundError(f"Missing merged model below {checkpoint_directory}.")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Recovery configuration must be a JSON object.")
    stride = config.get("encoder_stride_factor")
    pool = config.get("hidden_pool_factor")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("encoder_stride_factor must be a positive integer.")
    if not isinstance(pool, int) or pool < 1:
        raise ValueError("hidden_pool_factor must be a positive integer.")
    if pool != 1:
        raise ValueError(
            "Hidden pooling uses a runtime hook and cannot yet be exported safely; use factor 1."
        )
    return config


def write_recovery_export_workflow(
    checkpoint_directory: Path, workflow_path: Path, output_directory: Path
) -> None:
    """Write FP32 export plus the study's 8-bit K-quant deployment baseline."""
    load_recovery_config(checkpoint_directory)
    model_directory = checkpoint_directory / "merged-model"
    workflow = {
        "input_model": {
            "type": "HfModel",
            "model_path": str(model_directory),
            "task": "automatic-speech-recognition",
        },
        "systems": {
            "local_system": {
                "type": "LocalSystem",
                "accelerators": [
                    {"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}
                ],
            }
        },
        "engine": {"target": "local_system"},
        "passes": {
            "builder": {"type": "ModelBuilder", "precision": "fp32"},
            "full_int8": {
                "type": "OnnxKQuantQuantization",
                "bits": 8,
                "block_size": 32,
                "accuracy_level": 4,
            },
        },
        "output_dir": str(output_directory),
    }
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")


def finalize_recovery_export(checkpoint_directory: Path, model_directory: Path) -> None:
    """Restore architecture changes that Transformers config cannot serialize.

    Whisper does not expose the second convolution stride as a config field, so
    ModelBuilder reconstructs the default stride even though the recovered
    positional table and max-source length are compressed.
    """
    recovery = load_recovery_config(checkpoint_directory)
    factor = int(recovery["encoder_stride_factor"])
    if factor == 1:
        return
    encoder_path = model_directory / "encoder.onnx"
    if not encoder_path.is_file():
        raise FileNotFoundError(encoder_path)
    model = onnx.load(encoder_path)
    metadata = {item.key: item.value for item in model.metadata_props}
    marker = "whisper_recovery_encoder_stride_factor"
    if marker in metadata:
        if int(metadata[marker]) != factor:
            raise ValueError("Export was already finalized with a different stride factor.")
        return
    conv2 = next((node for node in model.graph.node if node.name.endswith("/Conv_2")), None)
    if conv2 is None:
        raise ValueError("Could not find Whisper's second encoder convolution.")
    stride = next((attribute for attribute in conv2.attribute if attribute.name == "strides"), None)
    if stride is None or len(stride.ints) != 1:
        raise ValueError("Second encoder convolution has no one-dimensional stride.")
    stride.ints[0] *= factor
    property_value = model.metadata_props.add()
    property_value.key = marker
    property_value.value = str(factor)
    del model.graph.value_info[:]
    onnx.save(model, encoder_path)
    onnx.checker.check_model(encoder_path)
