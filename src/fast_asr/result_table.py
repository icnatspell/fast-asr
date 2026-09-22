"""Create reusable benchmark ledgers from evaluation summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

RESULT_COLUMNS = (
    "config",
    "split",
    "utterances",
    "wer_percent",
    "cer_percent",
    "rtfx",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "tps",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "truncated_percent",
    "artifact_size_mb",
)


def _nested(report: dict[str, Any], *keys: str) -> Any:
    value: Any = report
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _scaled(value: Any, scale: float = 1.0) -> float | None:
    return None if value is None else float(value) * scale


def summary_row(summary_path: Path) -> dict[str, str | int | float | None]:
    """Flatten one versioned evaluation summary into the ledger schema."""
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1:
        raise ValueError(f"Unsupported summary schema in {summary_path}.")
    split = summary_path.parent.name
    config = summary_path.parent.parent.name
    return {
        "config": config,
        "split": split,
        "utterances": report.get("utterance_count"),
        "wer_percent": _scaled(_nested(report, "quality", "wer"), 100),
        "cer_percent": _scaled(_nested(report, "quality", "cer"), 100),
        "rtfx": _scaled(_nested(report, "latency", "rtfx")),
        "ttft_p50_ms": _scaled(_nested(report, "latency", "first_token_ms", "p50")),
        "ttft_p95_ms": _scaled(_nested(report, "latency", "first_token_ms", "p95")),
        "tps": _scaled(_nested(report, "latency", "decoder_tokens_per_second")),
        "tpot_p50_ms": _scaled(
            _nested(report, "latency", "time_per_output_token_ms", "p50")
        ),
        "tpot_p95_ms": _scaled(
            _nested(report, "latency", "time_per_output_token_ms", "p95")
        ),
        "truncated_percent": _scaled(_nested(report, "diagnostics", "truncation_rate"), 100),
        "artifact_size_mb": _scaled(
            _nested(report, "resources", "artifact_size_bytes"), 1 / 1_000_000
        ),
    }


def write_result_tables(summary_paths: list[Path], csv_path: Path, markdown_path: Path) -> None:
    """Write deterministic CSV and Markdown tables for any evaluation collection."""
    rows = sorted(
        (summary_row(path) for path in summary_paths),
        key=lambda row: (str(row["config"]), str(row["split"])),
    )
    if not rows:
        raise ValueError("At least one summary is required.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(RESULT_COLUMNS) + " |"
    divider = "| " + " | ".join("---" for _ in RESULT_COLUMNS) + " |"
    body = [
        "| "
        + " | ".join(
            "—" if row[column] is None else str(row[column]) for column in RESULT_COLUMNS
        )
        + " |"
        for row in rows
    ]
    markdown_path.write_text("\n".join([header, divider, *body]) + "\n", encoding="utf-8")


def refresh_result_tables(results_root: Path, csv_path: Path, markdown_path: Path) -> None:
    """Discover summaries deterministically and regenerate both ledger formats."""
    summaries = sorted(results_root.rglob("summary.json"))
    write_result_tables(summaries, csv_path, markdown_path)
