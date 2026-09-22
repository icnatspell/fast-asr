from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import onnx
from olive.workflows.run.config import RunConfig

from fast_asr.quantize import copy_onnx_model_for_quantization
from fast_asr.workflow import write_whisper_base_en_workflow


def test_base_en_workflow_preserves_official_export_stage(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.json"
    write_whisper_base_en_workflow(
        workflow_path,
        tmp_path / "calibration",
        tmp_path / "output",
        smoothquant_alpha=0.6,
        calibration_samples=16,
    )

    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    assert workflow["input_model"]["model_path"] == "openai/whisper-base.en"
    assert workflow["systems"]["local_system"]["accelerators"][0]["execution_providers"] == [
        "CPUExecutionProvider"
    ]
    assert workflow["passes"]["builder"] == {"type": "ModelBuilder", "precision": "fp32"}
    quantizer = workflow["passes"]["static_smoothquant_int8"]
    assert quantizer["type"] == "IncStaticQuantization"
    assert quantizer["quant_format"] == "QOperator"
    assert quantizer["recipes"]["smooth_quant_args"]["alpha"] == 0.6

    parsed = RunConfig.model_validate(json.loads(workflow_path.read_text(encoding="utf-8")))
    parsed_quantizer = parsed.passes["static_smoothquant_int8"][0]
    assert parsed_quantizer.type == "incstaticquantization"
    assert parsed_quantizer.config is not None
    parsed_config = cast(dict[str, object], parsed_quantizer.config)
    assert parsed_config["quant_format"] == "QOperator"


def test_quantization_source_copy_is_separate(tmp_path: Path) -> None:
    source = tmp_path / "component.onnx"
    graph = onnx.helper.make_graph([], "empty", [], [])
    onnx.save(onnx.helper.make_model(graph), source)

    copied = copy_onnx_model_for_quantization(source, tmp_path / "quantization-input")

    assert copied != source
    assert copied.read_bytes() == source.read_bytes()
