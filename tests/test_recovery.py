from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest
import torch
from olive.workflows.run.config import RunConfig
from onnx import TensorProto, helper

from fast_asr.compression_candidates import (
    LearnedTemporalDownsampler,
    adaptive_merge_hidden_states,
)
from fast_asr.recovery_export import (
    finalize_recovery_export,
    write_recovery_export_workflow,
)
from fast_asr.recovery_training import RecoveryConfig


def _checkpoint(path: Path, *, hidden_pool_factor: int = 1) -> Path:
    model = path / "merged-model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "recovery_config.json").write_text(
        json.dumps({"encoder_stride_factor": 2, "hidden_pool_factor": hidden_pool_factor}),
        encoding="utf-8",
    )
    return path


def test_recovery_workflow_is_valid_and_uses_full_int8(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.json"
    write_recovery_export_workflow(
        _checkpoint(tmp_path / "checkpoint"), workflow_path, tmp_path / "output"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    assert workflow["passes"]["full_int8"]["bits"] == 8
    RunConfig.model_validate(workflow)


def test_recovery_export_rejects_runtime_only_pooling(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="runtime hook"):
        write_recovery_export_workflow(
            _checkpoint(tmp_path / "checkpoint", hidden_pool_factor=2),
            tmp_path / "workflow.json",
            tmp_path / "output",
        )


def test_recovery_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="Positive values"):
        RecoveryConfig(lora_rank=0)


def test_finalize_recovery_export_restores_stride_idempotently(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    model_directory = tmp_path / "export"
    model_directory.mkdir()
    value = helper.make_tensor_value_info("value", TensorProto.FLOAT, [1])
    conv2 = helper.make_node(
        "Conv", ["value", "value"], ["output"], name="model/Conv_2", strides=[2]
    )
    graph = helper.make_graph(
        [conv2],
        "encoder",
        [value],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    onnx.save(helper.make_model(graph), model_directory / "encoder.onnx")
    finalize_recovery_export(checkpoint, model_directory)
    finalize_recovery_export(checkpoint, model_directory)
    model = onnx.load(model_directory / "encoder.onnx")
    stride = next(item for item in model.graph.node[0].attribute if item.name == "strides")
    assert list(stride.ints) == [4]


def test_compression_candidates_preserve_shape_and_constant_signal() -> None:
    hidden = torch.ones(1, 12, 8)
    downsampler = LearnedTemporalDownsampler(hidden_size=8, factor=3)
    assert downsampler(hidden).shape == (1, 4, 8)
    assert torch.allclose(downsampler(hidden), torch.ones(1, 4, 8))
    merged = adaptive_merge_hidden_states(hidden, 0.5)
    assert merged.shape == (1, 6, 8)
    assert torch.allclose(merged, torch.ones(1, 6, 8))
