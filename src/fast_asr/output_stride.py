"""ONNX graph transform for an explicit Whisper encoder output stride experiment."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

AntiAliasKernel = Literal["average", "binomial3", "binomial5"]
PoolMethod = Literal[
    "mean", "max", "binomial3", "binomial5", "left-weighted", "right-weighted"
]

_ANTI_ALIAS_KERNELS: dict[AntiAliasKernel, tuple[float, ...]] = {
    "average": (0.5, 0.5),
    "binomial3": (0.25, 0.5, 0.25),
    "binomial5": (0.0625, 0.25, 0.375, 0.25, 0.0625),
}

_POOL_KERNELS: dict[PoolMethod, tuple[float, ...]] = {
    "binomial3": (0.25, 0.5, 0.25),
    "binomial5": (0.0625, 0.25, 0.375, 0.25, 0.0625),
    "left-weighted": (0.75, 0.25),
    "right-weighted": (0.25, 0.75),
}


def _load_external_model_safely(model_path: Path) -> onnx.ModelProto:
    """Load external tensors through private copies, avoiding hard-link rejection."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_model = Path(temporary_directory) / model_path.name
        shutil.copyfile(model_path, temporary_model)
        data_path = model_path.with_suffix(model_path.suffix + ".data")
        if data_path.is_file():
            shutil.copyfile(data_path, temporary_model.with_suffix(model_path.suffix + ".data"))
        return onnx.load(temporary_model)


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

    encoder = _load_external_model_safely(source_encoder)
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


def create_antialiased_encoder_stride_candidate(
    source_model_directory: Path,
    output_model_directory: Path,
    *,
    kernel: AntiAliasKernel = "binomial3",
) -> None:
    """Halve Conv2 input resolution using fixed depthwise low-pass filtering.

    Original Conv2 stride remains two, producing 750 encoder frames. Unlike
    directly changing stride to four, every input phase contributes.
    """
    if output_model_directory.exists():
        raise FileExistsError(f"Refusing to overwrite {output_model_directory}.")
    coefficients = _ANTI_ALIAS_KERNELS[kernel]
    source_encoder = source_model_directory / "encoder.onnx"
    source_decoder = source_model_directory / "decoder.onnx"
    if not source_encoder.is_file() or not source_decoder.is_file():
        raise FileNotFoundError("Expected encoder.onnx and decoder.onnx in source artifact.")

    encoder = _load_external_model_safely(source_encoder)
    decoder = onnx.load(source_decoder, load_external_data=False)
    shutil.copytree(
        source_model_directory,
        output_model_directory,
        ignore=shutil.ignore_patterns("encoder.onnx", "encoder.onnx.data"),
    )
    conv2 = next((node for node in encoder.graph.node if node.name.endswith("/Conv_2")), None)
    if conv2 is None:
        raise ValueError("Could not find Whisper's second encoder convolution.")
    conv2_weight = next(
        (item for item in encoder.graph.initializer if item.name == conv2.input[1]), None
    )
    if conv2_weight is None:
        raise ValueError("Could not find second encoder convolution weights.")
    channels = numpy_helper.to_array(conv2_weight).shape[1]
    filter_name = _unique_name(encoder, f"fast_asr_antialias_{kernel}_weight")
    filtered_name = _unique_name(encoder, f"fast_asr_antialias_{kernel}_output")
    weights = np.tile(np.asarray(coefficients, dtype=np.float32), (channels, 1, 1))
    encoder.graph.initializer.append(numpy_helper.from_array(weights, filter_name))
    padding = len(coefficients) // 2
    pads = [padding, padding]
    if len(coefficients) % 2 == 0:
        pads[1] -= 1
    filter_node = helper.make_node(
        "Conv",
        [conv2.input[0], filter_name],
        [filtered_name],
        name=f"/fast_asr/Antialias_{kernel}",
        group=channels,
        kernel_shape=[len(coefficients)],
        pads=pads,
        strides=[2],
    )
    conv2_index = list(encoder.graph.node).index(conv2)
    encoder.graph.node.insert(conv2_index, filter_node)
    conv2.input[0] = filtered_name

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
    if position_values.shape[0] % 2:
        raise ValueError("Positional embedding length must divide evenly by two.")
    sequence_length = position_values.shape[0] // 2
    encoder.graph.initializer.remove(position)
    encoder.graph.initializer.append(
        numpy_helper.from_array(position_values[::2].copy(), position.name)
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


def _insert_pool_after_output(
    model: onnx.ModelProto,
    producer: onnx.NodeProto,
    output_name: str,
    factor: int,
    method: PoolMethod,
    channels: int,
) -> list[onnx.NodeProto]:
    """Replace one [batch, sequence, hidden] output with pooled equivalent."""
    output_index = next(index for index, name in enumerate(producer.output) if name == output_name)
    prefix = _unique_name(model, f"{output_name}_pool_{factor}")
    raw_name = f"{prefix}_raw"
    channels_first = f"{prefix}_channels_first"
    pooled = f"{prefix}_pooled"
    producer.output[output_index] = raw_name
    nodes = [
        helper.make_node(
            "Transpose",
            [raw_name],
            [channels_first],
            perm=[0, 2, 1],
            name=f"{prefix}_transpose_in",
        ),
    ]
    if method in {"mean", "max"}:
        nodes.append(
            helper.make_node(
                "AveragePool" if method == "mean" else "MaxPool",
                [channels_first],
                [pooled],
                kernel_shape=[factor],
                strides=[factor],
                name=f"{prefix}_{method}",
            )
        )
    else:
        if factor != 2:
            raise ValueError(f"Pooling method {method!r} only supports factor 2.")
        coefficients = _POOL_KERNELS[method]
        weight_name = f"{prefix}_weight"
        weights = np.tile(np.asarray(coefficients, dtype=np.float32), (channels, 1, 1))
        model.graph.initializer.append(numpy_helper.from_array(weights, weight_name))
        total_padding = len(coefficients) - factor
        nodes.append(
            helper.make_node(
                "Conv",
                [channels_first, weight_name],
                [pooled],
                group=channels,
                kernel_shape=[len(coefficients)],
                pads=[(total_padding + 1) // 2, total_padding // 2],
                strides=[factor],
                name=f"{prefix}_{method}",
            )
        )
    nodes.append(
        helper.make_node(
            "Transpose",
            [pooled],
            [output_name],
            perm=[0, 2, 1],
            name=f"{prefix}_transpose_out",
        )
    )
    return nodes


def create_intermediate_pool_candidate(
    source_model_directory: Path,
    output_model_directory: Path,
    *,
    after_layer: int,
    factor: int = 2,
    method: PoolMethod = "mean",
) -> None:
    """Pool encoder tokens after a one-indexed count of completed layers."""
    if after_layer not in range(1, 6):
        raise ValueError("after_layer must be between 1 and 5 for Whisper base.")
    if factor < 2:
        raise ValueError("factor must be at least 2.")
    if output_model_directory.exists():
        raise FileExistsError(f"Refusing to overwrite {output_model_directory}.")
    source_encoder = source_model_directory / "encoder.onnx"
    source_decoder = source_model_directory / "decoder.onnx"
    if not source_encoder.is_file() or not source_decoder.is_file():
        raise FileNotFoundError("Expected encoder.onnx and decoder.onnx in source artifact.")

    encoder = _load_external_model_safely(source_encoder)
    decoder = onnx.load(source_decoder, load_external_data=False)
    shutil.copytree(
        source_model_directory,
        output_model_directory,
        ignore=shutil.ignore_patterns("encoder.onnx", "encoder.onnx.data"),
    )
    boundary_name = f"/model/layers.{after_layer}/input_layernorm/SkipLayerNorm"
    boundary = next((node for node in encoder.graph.node if node.name == boundary_name), None)
    if boundary is None or len(boundary.output) < 4:
        raise ValueError(f"Could not find supported encoder boundary {boundary_name!r}.")
    boundary_index = list(encoder.graph.node).index(boundary)
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
    channels = numpy_helper.to_array(position).shape[1]
    inserted: list[onnx.NodeProto] = []
    for output_name in (boundary.output[0], boundary.output[3]):
        if not output_name:
            raise ValueError(f"Boundary {boundary_name!r} lacks required residual outputs.")
        inserted.extend(
            _insert_pool_after_output(
                encoder, boundary, output_name, factor, method, channels
            )
        )
    for offset, node in enumerate(inserted, start=1):
        encoder.graph.node.insert(boundary_index + offset, node)

    hidden_output = next(value for value in encoder.graph.output if value.name == "hidden_states")
    source_length = hidden_output.type.tensor_type.shape.dim[1].dim_value
    if not source_length or source_length % factor:
        raise ValueError("Hidden-state length must divide evenly by factor.")
    target_length = source_length // factor
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
    _set_dimension(hidden_output, axis=1, dimension=target_length)
    for output in encoder.graph.output:
        if output.name.startswith(("present_key_cross_", "present_value_cross_")):
            _set_dimension(output, axis=2, dimension=target_length)
    for input_value in decoder.graph.input:
        if input_value.name.startswith(("past_key_cross_", "past_value_cross_")):
            _set_dimension(input_value, axis=2, dimension=target_length)
    _update_max_source_positions(output_model_directory, target_length)
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


def create_content_aware_merge_candidate(
    source_model_directory: Path,
    output_model_directory: Path,
    *,
    after_layer: int = 2,
    reduction_ratio: float = 0.5,
) -> None:
    """Merge adjacent encoder states at low-change boundaries using ONNX ops."""
    if after_layer not in range(1, 6):
        raise ValueError("after_layer must be between 1 and 5 for Whisper base.")
    if not 0 < reduction_ratio < 1:
        raise ValueError("reduction_ratio must be within (0, 1).")
    if output_model_directory.exists():
        raise FileExistsError(f"Refusing to overwrite {output_model_directory}.")
    source_encoder = source_model_directory / "encoder.onnx"
    source_decoder = source_model_directory / "decoder.onnx"
    if not source_encoder.is_file() or not source_decoder.is_file():
        raise FileNotFoundError("Expected encoder.onnx and decoder.onnx in source artifact.")

    encoder = _load_external_model_safely(source_encoder)
    decoder = onnx.load(source_decoder, load_external_data=False)
    boundary_name = f"/model/layers.{after_layer}/input_layernorm/SkipLayerNorm"
    boundary = next((node for node in encoder.graph.node if node.name == boundary_name), None)
    if boundary is None or len(boundary.output) < 4 or not boundary.output[3]:
        raise ValueError(f"Could not find supported encoder boundary {boundary_name!r}.")
    hidden_output = next(value for value in encoder.graph.output if value.name == "hidden_states")
    source_length = hidden_output.type.tensor_type.shape.dim[1].dim_value
    position = next(
        (
            item
            for item in encoder.graph.initializer
            if item.name == "encoder.embed_positions.weight"
        ),
        None,
    )
    if not source_length or position is None:
        raise ValueError("Encoder needs static length and positional embeddings.")
    channels = numpy_helper.to_array(position).shape[1]
    target_length = max(2, round(source_length * (1 - reduction_ratio)))
    if target_length >= source_length:
        raise ValueError("Reduction ratio must remove at least one token.")

    prefix = f"fast_asr_merge_layer_{after_layer}_{target_length}"

    def names(suffix: str) -> str:
        return f"{prefix}_{suffix}"
    initializers = {
        "start0": np.asarray([0], dtype=np.int64),
        "start1": np.asarray([1], dtype=np.int64),
        "endminus1": np.asarray([source_length - 1], dtype=np.int64),
        "endfull": np.asarray([source_length], dtype=np.int64),
        "axis1": np.asarray([1], dtype=np.int64),
        "axis2": np.asarray([2], dtype=np.int64),
        "axis0": np.asarray([0], dtype=np.int64),
        "one": np.asarray(1, dtype=np.int64),
        "topk": np.asarray([target_length - 1], dtype=np.int64),
        "flags": np.zeros((1, source_length), dtype=np.int64),
        "flag_updates": np.ones((1, target_length - 1), dtype=np.int64),
        "segment_sum": np.zeros((target_length, channels), dtype=np.float32),
        "segment_count": np.zeros((target_length, 1), dtype=np.float32),
        "count_updates": np.ones((source_length, 1), dtype=np.float32),
    }
    encoder.graph.initializer.extend(
        numpy_helper.from_array(value, names(key)) for key, value in initializers.items()
    )
    norm_name, residual_name = boundary.output[0], boundary.output[3]
    norm_raw, residual_raw = names("norm_raw"), names("residual_raw")
    boundary.output[0], boundary.output[3] = norm_raw, residual_raw
    nodes = [
        helper.make_node(
            "Slice",
            [norm_raw, names("start0"), names("endminus1"), names("axis1")],
            [names("left")],
        ),
        helper.make_node(
            "Slice",
            [norm_raw, names("start1"), names("endfull"), names("axis1")],
            [names("right")],
        ),
        helper.make_node("Sub", [names("right"), names("left")], [names("difference")]),
        helper.make_node("Mul", [names("difference"), names("difference")], [names("square")]),
        helper.make_node(
            "ReduceSum", [names("square"), names("axis2")], [names("change")], keepdims=0
        ),
        helper.make_node(
            "TopK",
            [names("change"), names("topk")],
            [names("largest_changes"), names("boundary_indices")],
            largest=1,
            sorted=0,
        ),
        helper.make_node(
            "Add", [names("boundary_indices"), names("one")], [names("boundaries")]
        ),
        helper.make_node(
            "ScatterElements",
            [names("flags"), names("boundaries"), names("flag_updates")],
            [names("boundary_flags")],
            axis=1,
        ),
        helper.make_node(
            "CumSum", [names("boundary_flags"), names("one")], [names("segment_ids")]
        ),
        helper.make_node(
            "Squeeze", [names("segment_ids"), names("axis0")], [names("flat_ids")]
        ),
        helper.make_node(
            "Unsqueeze", [names("flat_ids"), names("axis1")], [names("scatter_indices")]
        ),
        helper.make_node(
            "ScatterND",
            [names("segment_count"), names("scatter_indices"), names("count_updates")],
            [names("counts")],
            reduction="add",
        ),
    ]
    for original_name, raw_name, label in (
        (norm_name, norm_raw, "norm"),
        (residual_name, residual_raw, "residual"),
    ):
        nodes.extend(
            [
                helper.make_node(
                    "Squeeze", [raw_name, names("axis0")], [names(f"{label}_flat")]
                ),
                helper.make_node(
                    "ScatterND",
                    [names("segment_sum"), names("scatter_indices"), names(f"{label}_flat")],
                    [names(f"{label}_sum")],
                    reduction="add",
                ),
                helper.make_node(
                    "Div", [names(f"{label}_sum"), names("counts")], [names(f"{label}_mean")]
                ),
                helper.make_node(
                    "Unsqueeze", [names(f"{label}_mean"), names("axis0")], [original_name]
                ),
            ]
        )
    boundary_index = list(encoder.graph.node).index(boundary)
    for offset, node in enumerate(nodes, start=1):
        node.name = names(f"node_{offset}")
        encoder.graph.node.insert(boundary_index + offset, node)

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
    _set_dimension(hidden_output, axis=1, dimension=target_length)
    for output in encoder.graph.output:
        if output.name.startswith(("present_key_cross_", "present_value_cross_")):
            _set_dimension(output, axis=2, dimension=target_length)
    for input_value in decoder.graph.input:
        if input_value.name.startswith(("past_key_cross_", "past_value_cross_")):
            _set_dimension(input_value, axis=2, dimension=target_length)
    shutil.copytree(
        source_model_directory,
        output_model_directory,
        ignore=shutil.ignore_patterns("encoder.onnx", "encoder.onnx.data"),
    )
    _update_max_source_positions(output_model_directory, target_length)
    del encoder.graph.value_info[:]
    onnx.save_model(
        encoder,
        output_model_directory / "encoder.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="encoder.onnx.data",
        size_threshold=1024,
    )
    onnx.save(decoder, output_model_directory / "decoder.onnx")
    onnx.checker.check_model(output_model_directory / "encoder.onnx")
    onnx.checker.check_model(output_model_directory / "decoder.onnx")
