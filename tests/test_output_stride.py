import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from fast_asr.output_stride import _load_external_model_safely, create_output_stride_candidate


def _write_model(
    path: Path,
    name: str,
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", [inputs[0].name], [output.name]) for output in outputs],
        name,
        inputs,
        outputs,
    )
    onnx.save(helper.make_model(graph), path)


def test_create_output_stride_candidate_updates_all_cross_attention_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    audio = helper.make_tensor_value_info("audio_features", TensorProto.FLOAT, [1, 80, 3000])
    encoder_outputs = [
        helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [1, 1500, 512]),
        helper.make_tensor_value_info("present_key_cross_0", TensorProto.FLOAT, [1, 8, 1500, 64]),
        helper.make_tensor_value_info("present_value_cross_0", TensorProto.FLOAT, [1, 8, 1500, 64]),
    ]
    _write_model(source / "encoder.onnx", "encoder", [audio], encoder_outputs)
    decoder_input = helper.make_tensor_value_info(
        "past_key_cross_0", TensorProto.FLOAT, [1, 8, 1500, 64]
    )
    decoder_output = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, [1, 8, 1500, 64]
    )
    _write_model(source / "decoder.onnx", "decoder", [decoder_input], [decoder_output])

    candidate = tmp_path / "candidate"
    create_output_stride_candidate(source, candidate)

    encoder = onnx.load(candidate / "encoder.onnx")
    decoder = onnx.load(candidate / "decoder.onnx")
    assert {output.name for output in encoder.graph.output} == {
        "hidden_states",
        "present_key_cross_0",
        "present_value_cross_0",
    }
    assert sum(node.op_type == "Slice" for node in encoder.graph.node) == 3
    assert decoder.graph.input[0].type.tensor_type.shape.dim[2].dim_value == 750


def test_safe_loader_accepts_hardlinked_external_data(tmp_path: Path) -> None:
    tensor = onnx.numpy_helper.from_array(np.ones((32,), dtype=np.float32), "weight")
    output = helper.make_tensor_value_info("weight", TensorProto.FLOAT, [32])
    model = helper.make_model(helper.make_graph([], "external", [], [output], [tensor]))
    model_path = tmp_path / "encoder.onnx"
    onnx.save_model(
        model,
        model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="encoder.onnx.data",
        size_threshold=0,
    )
    os.link(tmp_path / "encoder.onnx.data", tmp_path / "hardlink-backup.data")

    loaded = _load_external_model_safely(model_path)

    assert onnx.numpy_helper.to_array(loaded.graph.initializer[0]).shape == (32,)
