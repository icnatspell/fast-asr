"""Bounded-thread ONNX Runtime profiling with a calibration-shaped NPZ input."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort


def load_npz_inputs(input_file: Path) -> dict[str, np.ndarray]:
    """Load one named ONNX input tensor per key from an NPZ fixture."""
    with np.load(input_file, allow_pickle=False) as sample:
        if not sample.files:
            raise ValueError(f"{input_file} does not contain any ONNX inputs.")
        return {name: sample[name] for name in sample.files}


def profile_cpu_model(
    model_path: Path,
    input_file: Path,
    output_directory: Path,
    *,
    threads: int = 4,
    warmup_runs: int = 10,
    measured_runs: int = 50,
) -> dict[str, Any]:
    """Run a bounded-thread batch-one profile and persist its JSON summary and ORT trace."""
    if threads < 1 or warmup_runs < 0 or measured_runs < 1:
        raise ValueError(
            "threads and measured_runs must be positive; warmup_runs cannot be negative."
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    inputs = load_npz_inputs(input_file)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(threads)

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_profiling = True
    options.profile_file_prefix = str(output_directory / "onnxruntime-profile")
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])

    for _ in range(warmup_runs):
        session.run(None, inputs)

    durations_ms: list[float] = []
    for _ in range(measured_runs):
        started = time.perf_counter_ns()
        session.run(None, inputs)
        durations_ms.append((time.perf_counter_ns() - started) / 1_000_000)

    profile_file = Path(session.end_profiling())
    copied_trace = output_directory / "onnxruntime-profile.json"
    shutil.copy2(profile_file, copied_trace)
    durations = np.asarray(durations_ms)
    result: dict[str, Any] = {
        "model": str(model_path),
        "input": str(input_file),
        "execution_provider": "CPUExecutionProvider",
        "intra_op_threads": threads,
        "inter_op_threads": 1,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "latency_ms": {
            "mean": float(durations.mean()),
            "median": float(np.median(durations)),
            "p95": float(np.percentile(durations, 95)),
        },
        "profile_trace": str(copied_trace),
    }
    (output_directory / "profile-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def profile_model_artifact(
    model_directory: Path,
    output_directory: Path,
    *,
    threads: int = 4,
    warmup_runs: int = 3,
    measured_runs: int = 10,
) -> dict[str, Any]:
    """Profile encoder and one cached decoder step with deterministic inputs."""
    if threads < 1 or warmup_runs < 0 or measured_runs < 1:
        raise ValueError("Invalid profiling run counts or thread count.")
    output_directory.mkdir(parents=True, exist_ok=True)

    def session(component: str) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_profiling = True
        options.profile_file_prefix = str(output_directory / component)
        return ort.InferenceSession(
            str(model_directory / f"{component}.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )

    encoder = session("encoder")
    encoder_feeds = {"audio_features": np.zeros((1, 80, 3000), dtype=np.float32)}
    encoded = dict(zip(
        [item.name for item in encoder.get_outputs()],
        cast(list[np.ndarray], encoder.run(None, encoder_feeds)),
        strict=True,
    ))
    decoder = session("decoder")
    decoder_feeds: dict[str, np.ndarray] = {
        "input_ids": np.asarray([[50257]], dtype=np.int32)
    }
    for item in decoder.get_inputs():
        if item.name.startswith(("past_key_cross_", "past_value_cross_")):
            decoder_feeds[item.name] = encoded[item.name.replace("past_", "present_", 1)]
        elif item.name.startswith(("past_key_self_", "past_value_self_")):
            decoder_feeds[item.name] = np.empty((1, 8, 0, 64), dtype=np.float32)

    report: dict[str, Any] = {
        "model_directory": str(model_directory),
        "threads": threads,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "components": {},
    }
    for component, runtime, feeds in (
        ("encoder", encoder, encoder_feeds),
        ("decoder", decoder, decoder_feeds),
    ):
        for _ in range(warmup_runs):
            runtime.run(None, feeds)
        durations = []
        for _ in range(measured_runs):
            start = time.perf_counter_ns()
            runtime.run(None, feeds)
            durations.append((time.perf_counter_ns() - start) / 1_000_000)
        trace = Path(runtime.end_profiling())
        events = json.loads(trace.read_text(encoding="utf-8"))
        by_operator: dict[str, float] = {}
        for event in events:
            if event.get("cat") != "Node" or not isinstance(event.get("dur"), (int, float)):
                continue
            operator = str(event.get("args", {}).get("op_name", "unknown"))
            by_operator[operator] = by_operator.get(operator, 0.0) + event["dur"] / 1000
        profiled_runs = warmup_runs + measured_runs + (1 if component == "encoder" else 0)
        trace_target = output_directory / f"{component}-trace.json"
        shutil.copy2(trace, trace_target)
        report["components"][component] = {
            "median_ms": float(np.median(durations)),
            "p95_ms": float(np.percentile(durations, 95)),
            "operator_ms_per_run": dict(
                sorted(
                    (
                        (operator, duration / profiled_runs)
                        for operator, duration in by_operator.items()
                    ),
                    key=lambda pair: -pair[1],
                )
            ),
            "trace": str(trace_target),
        }
    (output_directory / "profile-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
