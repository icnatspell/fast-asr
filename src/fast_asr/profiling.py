"""Bounded-thread ONNX Runtime profiling with a calibration-shaped NPZ input."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

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
