from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper

from fast_asr.provenance import validate_model_artifact, write_provenance


def _write_artifact(path: Path) -> None:
    path.mkdir()
    audio = helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 80, 3000])
    present = helper.make_tensor_value_info(
        "present_key_cross_0", TensorProto.FLOAT, [1, 8, 750, 64]
    )
    encoder = helper.make_graph(
        [helper.make_node("Identity", ["audio"], ["present_key_cross_0"])],
        "encoder",
        [audio],
        [present],
    )
    past = helper.make_tensor_value_info(
        "past_key_cross_0", TensorProto.FLOAT, [1, 8, 750, 64]
    )
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, 1])
    decoder = helper.make_graph(
        [helper.make_node("Identity", ["past_key_cross_0"], ["logits"])],
        "decoder",
        [past],
        [logits],
    )
    onnx.save(helper.make_model(encoder), path / "encoder.onnx")
    onnx.save(helper.make_model(decoder), path / "decoder.onnx")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_validation_and_provenance_capture_contract(tmp_path: Path) -> None:
    artifact = tmp_path / "model"
    _write_artifact(artifact)
    assert validate_model_artifact(artifact)["encoder_frames"] == 750
    report = write_provenance(
        tmp_path / "provenance.json",
        artifact,
        split="test.clean",
        max_samples=64,
        threads=4,
        max_tokens=448,
    )
    assert report["settings"]["threads"] == 4
    assert set(report["artifact"]["files"]) == {
        "decoder.onnx",
        "encoder.onnx",
        "tokenizer.json",
    }
