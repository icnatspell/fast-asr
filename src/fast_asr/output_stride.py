"""ONNX graph transform for an explicit Whisper encoder output stride experiment."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper


def _unique_name(model: onnx.ModelProto, stem: str) -> str:
    names = {
        value.name
        for collection in (model.graph.input, model.graph.output, model.graph.value_info)
        for value in collection
    }
    names.update(initializer.name for initializer in model.graph.initializer)
    candidate = stem
    suffix = 0
    while candidate in names:
        suffix += 1
        candidate = f"{stem}_{suffix}"
    return candidate


def _add_stride_slice(
    model: onnx.ModelProto, output_name: str, axis: int, stride: int
) -> None:
    output = next((item for item in model.graph.output if item.name == output_name), None)
    if output is None:
        raise ValueError(f"Encoder output {output_name!r} was not found.")
    dimension = output.type.tensor_type.shape.dim[axis]
    if not dimension.HasField("dim_value") or dimension.dim_value % stride:
        message = f"Encoder output {output_name!r} must have a divisible static sequence length."
        raise ValueError(message)
    dimension.dim_value //= stride
    prefix = _unique_name(model, f"{output_name}_stride_{stride}")
    starts_name = f"{prefix}_starts"
    ends_name = f"{prefix}_ends"
    axes_name = f"{prefix}_axes"
    steps_name = f"{prefix}_steps"
    raw_name = f"{prefix}_raw"
    producers = [node for node in model.graph.node if output_name in node.output]
    if len(producers) != 1:
        raise ValueError(f"Expected one producer for encoder output {output_name!r}.")
    producer = producers[0]
    output_index = next(index for index, name in enumerate(producer.output) if name == output_name)
    producer.output[output_index] = raw_name
    for node in model.graph.node:
        for input_index, name in enumerate(node.input):
            if name == output_name:
                node.input[input_index] = raw_name
    model.graph.initializer.extend(
        [
            helper.make_tensor(starts_name, TensorProto.INT64, [1], [0]),
            helper.make_tensor(ends_name, TensorProto.INT64, [1], [2**63 - 1]),
            helper.make_tensor(axes_name, TensorProto.INT64, [1], [axis]),
            helper.make_tensor(steps_name, TensorProto.INT64, [1], [stride]),
        ]
    )
    model.graph.node.append(
        helper.make_node(
            "Slice",
            [raw_name, starts_name, ends_name, axes_name, steps_name],
            [output_name],
            name=f"{prefix}_slice",
        )
    )


def _set_dimension(value: onnx.ValueInfoProto, axis: int, dimension: int) -> None:
    shape = value.type.tensor_type.shape
    if len(shape.dim) <= axis:
        raise ValueError(f"{value.name!r} does not have axis {axis}.")
    shape.dim[axis].ClearField("dim_param")
    shape.dim[axis].dim_value = dimension


def _update_max_source_positions(model_directory: Path, sequence_length: int) -> None:
    """Keep Whisper metadata aligned with a graph whose encoder length changed."""
    for filename in ("config.json", "model_config.json"):
        path = model_directory / filename
        if not path.is_file():
            continue
        content = json.loads(path.read_text(encoding="utf-8"))

        def update(value: object) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "max_source_positions":
                        value[key] = sequence_length
                    else:
                        update(child)
            elif isinstance(value, list):
                for child in value:
                    update(child)

        update(content)
        path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def create_output_stride_candidate(
    source_model_directory: Path, output_model_directory: Path, *, stride: int = 2
) -> None:
    """Copy a GenAI Whisper artifact and subsample all encoder sequence outputs.

    Whisper's exported encoder has a hidden-state output plus cross-attention
    K/V outputs for every decoder layer. All must be reduced together; decoder
    input metadata is updated to the new fixed sequence length. This is an
    intentionally lossy, training-free P2 candidate and needs full ASR eval.
    """
    if stride < 2:
        raise ValueError("stride must be at least 2; use the source artifact for stride 1.")
    if output_model_directory.exists():
        raise FileExistsError(f"Refusing to overwrite {output_model_directory}.")
    encoder_path = source_model_directory / "encoder.onnx"
    decoder_path = source_model_directory / "decoder.onnx"
    if not encoder_path.is_file() or not decoder_path.is_file():
        raise FileNotFoundError("Expected encoder.onnx and decoder.onnx in the source artifact.")

    shutil.copytree(source_model_directory, output_model_directory)
    encoder = onnx.load(output_model_directory / "encoder.onnx", load_external_data=False)
    decoder = onnx.load(output_model_directory / "decoder.onnx", load_external_data=False)
    cross_output = next(
        value for value in encoder.graph.output if value.name.startswith("present_key_cross_")
    )
    sequence_length = cross_output.type.tensor_type.shape.dim[2].dim_value // stride

    for output in list(encoder.graph.output):
        if output.name == "hidden_states":
            _add_stride_slice(encoder, output.name, axis=1, stride=stride)
        elif output.name.startswith(("present_key_cross_", "present_value_cross_")):
            _add_stride_slice(encoder, output.name, axis=2, stride=stride)

    for input_value in decoder.graph.input:
        if input_value.name.startswith(("past_key_cross_", "past_value_cross_")):
            _set_dimension(input_value, axis=2, dimension=sequence_length)

    _update_max_source_positions(output_model_directory, sequence_length)

    onnx.save(encoder, output_model_directory / "encoder.onnx")
    onnx.save(decoder, output_model_directory / "decoder.onnx")
    onnx.checker.check_model(output_model_directory / "encoder.onnx")
    onnx.checker.check_model(output_model_directory / "decoder.onnx")


def create_hidden_state_pool_candidate(
    source_model_directory: Path, output_model_directory: Path, *, factor: int = 2
) -> None:
    """Mean-pool final encoder states before decoder cross-attention projections."""
    if factor < 2:
        raise ValueError("factor must be at least 2.")
    if output_model_directory.exists():
        raise FileExistsError(f"Refusing to overwrite {output_model_directory}.")
    shutil.copytree(source_model_directory, output_model_directory)
    encoder_path = output_model_directory / "encoder.onnx"
    decoder_path = output_model_directory / "decoder.onnx"
    encoder = onnx.load(encoder_path, load_external_data=False)
    decoder = onnx.load(decoder_path, load_external_data=False)
    hidden_output = next(value for value in encoder.graph.output if value.name == "hidden_states")
    source_length = hidden_output.type.tensor_type.shape.dim[1].dim_value
    if not source_length or source_length % factor:
        raise ValueError("Hidden-state length must divide evenly by factor.")
    target_length = source_length // factor
    producers = [node for node in encoder.graph.node if "hidden_states" in node.output]
    if len(producers) != 1:
        raise ValueError("Expected one hidden_states producer.")
    producer = producers[0]
    producer_index = list(encoder.graph.node).index(producer)
    output_index = next(i for i, name in enumerate(producer.output) if name == "hidden_states")
    raw_name = f"hidden_states_pool_{factor}_raw"
    transposed_name = f"hidden_states_pool_{factor}_channels_first"
    pooled_name = f"hidden_states_pool_{factor}_pooled"
    producer.output[output_index] = raw_name
    pooling_nodes = [
        helper.make_node(
            "Transpose",
            [raw_name],
            [transposed_name],
            perm=[0, 2, 1],
            name=f"hidden_states_pool_{factor}_transpose_in",
        ),
        helper.make_node(
            "AveragePool",
            [transposed_name],
            [pooled_name],
            kernel_shape=[factor],
            strides=[factor],
            name=f"hidden_states_pool_{factor}_average",
        ),
        helper.make_node(
            "Transpose",
            [pooled_name],
            ["hidden_states"],
            perm=[0, 2, 1],
            name=f"hidden_states_pool_{factor}_transpose_out",
        ),
    ]
    for offset, node in enumerate(pooling_nodes, start=1):
        encoder.graph.node.insert(producer_index + offset, node)

    _set_dimension(hidden_output, axis=1, dimension=target_length)
    for node in encoder.graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.name != "value" or attribute.type != onnx.AttributeProto.TENSOR:
                continue
            values = numpy_helper.to_array(attribute.t)
            if values.ndim == 1 and values.size == 4 and values[1] == source_length:
                replacement = values.copy()
                replacement[1] = target_length
                attribute.t.CopyFrom(numpy_helper.from_array(replacement))
    for output in encoder.graph.output:
        if output.name.startswith(("present_key_cross_", "present_value_cross_")):
            _set_dimension(output, axis=2, dimension=target_length)
    for input_value in decoder.graph.input:
        if input_value.name.startswith(("past_key_cross_", "past_value_cross_")):
            _set_dimension(input_value, axis=2, dimension=target_length)
    _update_max_source_positions(output_model_directory, target_length)
    del encoder.graph.value_info[:]
    onnx.save(encoder, encoder_path)
    onnx.save(decoder, decoder_path)
    onnx.checker.check_model(encoder_path)
    onnx.checker.check_model(decoder_path)


def create_encoder_stride_candidate(
    source_model_directory: Path, output_model_directory: Path, *, factor: int = 2
) -> None:
    """Increase Whisper's second encoder-convolution stride by ``factor``."""
    if factor < 2:
        raise ValueError("factor must be at least 2; use the source artifact for factor 1.")
    if output_model_directory.exists():
        raise FileExistsError(f"Refusing to overwrite {output_model_directory}.")
    source_encoder = source_model_directory / "encoder.onnx"
    source_decoder = source_model_directory / "decoder.onnx"
    if not source_encoder.is_file() or not source_decoder.is_file():
        raise FileNotFoundError("Expected encoder.onnx and decoder.onnx in the source artifact.")

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_encoder = Path(temporary_directory) / "encoder.onnx"
        shutil.copyfile(source_encoder, temporary_encoder)
        shutil.copyfile(
            source_encoder.with_suffix(".onnx.data"), temporary_encoder.with_suffix(".onnx.data")
        )
        encoder = onnx.load(temporary_encoder)
    decoder = onnx.load(source_decoder, load_external_data=False)
    shutil.copytree(
        source_model_directory,
        output_model_directory,
        ignore=shutil.ignore_patterns("encoder.onnx", "encoder.onnx.data"),
    )
    conv2 = next((node for node in encoder.graph.node if node.name.endswith("/Conv_2")), None)
    if conv2 is None:
        raise ValueError("Could not find Whisper's second encoder convolution.")
    stride_attribute = next((item for item in conv2.attribute if item.name == "strides"), None)
    if stride_attribute is None or len(stride_attribute.ints) != 1:
        raise ValueError("Second encoder convolution has no one-dimensional stride.")
    stride_attribute.ints[0] *= factor

    position = next(
        (
            item
            for item in encoder.graph.initializer
            if item.name == "encoder.embed_positions.weight"
        ),
        None,
    )
    if position is None:
        raise ValueError("Could not find Whisper encoder positional embeddings.")
    position_values = numpy_helper.to_array(position)
    if position_values.shape[0] % factor:
        raise ValueError("Positional embedding length must divide evenly by factor.")
    sequence_length = position_values.shape[0] // factor
    encoder.graph.initializer.remove(position)
    encoder.graph.initializer.append(
        numpy_helper.from_array(position_values[::factor].copy(), position.name)
    )
    for node in encoder.graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.name != "value" or attribute.type != onnx.AttributeProto.TENSOR:
                continue
            values = numpy_helper.to_array(attribute.t)
            if values.ndim == 1 and values.size == 4 and values[1] == position_values.shape[0]:
                replacement = values.copy()
                replacement[1] = sequence_length
                attribute.t.CopyFrom(numpy_helper.from_array(replacement))
    for output in encoder.graph.output:
        if output.name == "hidden_states":
            _set_dimension(output, axis=1, dimension=sequence_length)
        elif output.name.startswith(("present_key_cross_", "present_value_cross_")):
            _set_dimension(output, axis=2, dimension=sequence_length)
    for input_value in decoder.graph.input:
        if input_value.name.startswith(("past_key_cross_", "past_value_cross_")):
            _set_dimension(input_value, axis=2, dimension=sequence_length)
    _update_max_source_positions(output_model_directory, sequence_length)
    # Intermediate ValueInfo entries were inferred for the original 1500-frame
    # graph. ORT can safely re-infer them, whereas retaining them makes the new
    # positional Add appear dimensionally inconsistent.
    del encoder.graph.value_info[:]
    onnx.save_model(
        encoder,
        output_model_directory / "encoder.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="encoder.onnx.data",
        size_threshold=0,
    )
    onnx.save(decoder, output_model_directory / "decoder.onnx")
    onnx.checker.check_model(output_model_directory / "encoder.onnx")
    onnx.checker.check_model(output_model_directory / "decoder.onnx")
