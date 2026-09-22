"""Static U8S8 QOperator quantization through Olive's INC SmoothQuant pass."""

from __future__ import annotations

import shutil
from pathlib import Path

import onnx
from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.inc_quantization import IncStaticQuantization

from fast_asr.calibration import create_calibration_config


def copy_onnx_model_for_quantization(input_model: Path, workspace: Path) -> Path:
    """Copy an ONNX model and its external weights before INC mutates graph metadata."""
    copied_model = workspace / input_model.name
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_model, copied_model)

    model = onnx.load(input_model, load_external_data=False)
    source_root = input_model.parent.resolve()
    workspace_root = workspace.resolve()
    for initializer in model.graph.initializer:
        external_data = {entry.key: entry.value for entry in initializer.external_data}
        location = external_data.get("location")
        if location is None:
            continue
        source_data = (source_root / location).resolve()
        copied_data = (workspace_root / location).resolve()
        safe_source = source_data.is_relative_to(source_root)
        safe_destination = copied_data.is_relative_to(workspace_root)
        if not safe_source or not safe_destination:
            raise ValueError(f"Unsafe external-data location {location!r} in {input_model}.")
        copied_data.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_data, copied_data)
    return copied_model


def quantize_with_smoothquant(
    input_model: Path,
    calibration_directory: Path,
    output_directory: Path,
    *,
    smoothquant_alpha: float = 0.5,
    calibration_samples: int = 128,
) -> Path:
    """Create an INT8-weight/U8-activation QOperator model with SmoothQuant.

    QOperator is intentionally U8S8: signed-activation QOperator is unsupported or slow
    on x86-64. ``input_model`` is never mutated; Neural Compressor intermediates are
    written below ``output_directory``.
    """
    if not input_model.is_file():
        raise FileNotFoundError(input_model)
    if not 0.0 <= smoothquant_alpha <= 1.0:
        raise ValueError("smoothquant_alpha must be within [0.0, 1.0].")
    if calibration_samples < 1:
        raise ValueError("calibration_samples must be positive.")

    output_directory.mkdir(parents=True, exist_ok=True)
    quantization_input = copy_onnx_model_for_quantization(
        input_model, output_directory / "unquantized-copy"
    )
    config = {
        "approach": "static",
        "device": "cpu",
        "backend": "default",
        "domain": "nlp",
        "quant_format": "QOperator",
        "calibration_sampling_size": [calibration_samples],
        "data_config": create_calibration_config(calibration_directory, calibration_samples),
        "recipes": {
            "smooth_quant": True,
            "smooth_quant_args": {"alpha": smoothquant_alpha},
        },
        "workspace": str(output_directory / "neural-compressor-workspace"),
        "tuning_criterion": {"strategy": "basic", "max_trials": 1, "objective": "performance"},
        "save_as_external_data": True,
        "all_tensors_to_one_file": True,
    }
    olive_pass = create_pass_from_dict(IncStaticQuantization, config, disable_search=True)
    quantized_model = olive_pass.run(
        ONNXModelHandler(str(quantization_input)), str(output_directory)
    )
    return Path(quantized_model.model_path)
