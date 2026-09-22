"""Declarative, resumable benchmark matrices for candidate screening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fast_asr.benchmark import benchmark_librispeech


def write_screening_plan(
    model_directories: list[Path],
    output_root: Path,
    plan_path: Path,
    *,
    splits: list[str],
    max_samples: int,
    threads: int,
) -> None:
    """Write an inspectable matrix without starting compute-heavy evaluation."""
    if not model_directories or not splits:
        raise ValueError("At least one model and split are required.")
    if max_samples < 0 or threads < 1:
        raise ValueError("max_samples must be non-negative; threads must be positive.")
    jobs = [
        {
            "model_directory": str(model),
            "output_directory": str(output_root / model.name / split.replace(".", "-")),
            "split": split,
            "max_samples": max_samples,
            "threads": threads,
        }
        for model in model_directories
        for split in splits
    ]
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps({"schema_version": 1, "jobs": jobs}, indent=2) + "\n", encoding="utf-8"
    )


def run_screening_plan(plan_path: Path) -> list[dict[str, Any]]:
    """Run unfinished jobs sequentially; existing summaries make the plan resumable."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or not isinstance(plan.get("jobs"), list):
        raise ValueError("Unsupported screening plan.")
    reports: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        output = Path(job["output_directory"])
        summary = output / "summary.json"
        if summary.is_file():
            reports.append(json.loads(summary.read_text(encoding="utf-8")))
            continue
        reports.append(
            benchmark_librispeech(
                Path(job["model_directory"]),
                output,
                split=str(job["split"]),
                max_samples=int(job["max_samples"]),
                threads=int(job["threads"]),
            )
        )
    return reports
