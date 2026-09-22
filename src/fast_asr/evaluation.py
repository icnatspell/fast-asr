"""Runtime-neutral ASR evaluation from a JSONL per-utterance record contract."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

REQUIRED_RECORD_FIELDS = {"utterance_id", "reference", "prediction", "audio_duration_s"}
TEXT_NORMALIZER = BasicTextNormalizer()


def _normalized_text(record: dict[str, Any], field: str) -> str:
    return TEXT_NORMALIZER(str(record[field])).strip()


def load_records(records_path: Path) -> list[dict[str, Any]]:
    """Load and validate one JSON object per evaluated utterance."""
    records: list[dict[str, Any]] = []
    lines = records_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_number} is not a JSON object.")
        missing_fields = REQUIRED_RECORD_FIELDS.difference(record)
        if missing_fields:
            raise ValueError(f"Line {line_number} is missing {sorted(missing_fields)}.")
        records.append(record)
    if not records:
        raise ValueError("No evaluation records were found.")
    return records


def _numeric_values(records: Iterable[dict[str, Any]], field: str) -> list[float]:
    return [float(record[field]) for record in records if record.get(field) is not None]


def _time_to_first_token_values(records: Iterable[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for record in records:
        if record.get("ttft_ms") is not None:
            values.append(float(record["ttft_ms"]))
        elif all(
            record.get(field) is not None
            for field in ("e2e_latency_ms", "decode_latency_ms", "first_token_latency_ms")
        ):
            values.append(
                float(record["e2e_latency_ms"])
                - float(record["decode_latency_ms"])
                + float(record["first_token_latency_ms"])
            )
    return values


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def _artifact_size_bytes(artifact_path: Path | None) -> int | None:
    if artifact_path is None:
        return None
    if artifact_path.is_file():
        return artifact_path.stat().st_size
    if artifact_path.is_dir():
        return sum(path.stat().st_size for path in artifact_path.rglob("*") if path.is_file())
    raise FileNotFoundError(artifact_path)


def _error_rates(records: Sequence[dict[str, Any]]) -> dict[str, float] | None:
    scored = [record for record in records if _normalized_text(record, "reference")]
    if not scored:
        return None
    output = jiwer.process_words(
        [_normalized_text(record, "reference") for record in scored],
        [_normalized_text(record, "prediction") for record in scored],
    )
    reference_words = output.hits + output.substitutions + output.deletions
    return {
        "wer": float(output.wer),
        "substitution_rate": output.substitutions / reference_words,
        "deletion_rate": output.deletions / reference_words,
        "insertion_rate": output.insertions / reference_words,
        "exact_match_rate": sum(
            _normalized_text(record, "reference") == _normalized_text(record, "prediction")
            for record in scored
        )
        / len(scored),
    }


def _character_error_rate(records: Sequence[dict[str, Any]]) -> float | None:
    scored = [record for record in records if _normalized_text(record, "reference")]
    if not scored:
        return None
    return float(
        jiwer.process_characters(
            [_normalized_text(record, "reference") for record in scored],
            [_normalized_text(record, "prediction") for record in scored],
        ).cer
    )


def build_evaluation_report(
    records: Sequence[dict[str, Any]], artifact_path: Path | None = None
) -> dict[str, Any]:
    """Aggregate quality, latency, resource, and diagnostic metrics.

    Missing optional fields produce ``null`` sections instead of fabricated values.
    This lets the same schema cover any ASR runtime while requiring runners to
    expose richer instrumentation as it becomes available.
    """
    error_rates = _error_rates(records)
    audio_seconds = sum(float(record["audio_duration_s"]) for record in records)
    e2e_latencies_ms = _numeric_values(records, "e2e_latency_ms")
    total_e2e_seconds = sum(e2e_latencies_ms) / 1_000
    silence_records = [record for record in records if record.get("is_silence", False)]
    generated_tokens = _numeric_values(records, "generated_tokens")
    decode_latencies_ms = _numeric_values(records, "decode_latency_ms")

    latency = {
        "end_to_end_ms": _distribution(e2e_latencies_ms),
        "encoder_ms": _distribution(_numeric_values(records, "encoder_latency_ms")),
        "first_token_ms": _distribution(_time_to_first_token_values(records)),
        "decoder_first_token_ms": _distribution(
            _numeric_values(records, "first_token_latency_ms")
        ),
        "time_per_output_token_ms": _distribution(_numeric_values(records, "tpot_ms")),
        "decode_ms": _distribution(decode_latencies_ms),
        "rtf": total_e2e_seconds / audio_seconds if audio_seconds and e2e_latencies_ms else None,
        "rtfx": audio_seconds / total_e2e_seconds if total_e2e_seconds else None,
        "decoder_tokens_per_second": (
            sum(generated_tokens) / (sum(decode_latencies_ms) / 1_000)
            if generated_tokens and decode_latencies_ms and sum(decode_latencies_ms)
            else None
        ),
    }
    return {
        "schema_version": 1,
        "utterance_count": len(records),
        "audio_duration_s": audio_seconds,
        "quality": {
            **(error_rates or {}),
            "cer": _character_error_rate(records),
            "mean_token_agreement": _distribution(_numeric_values(records, "token_agreement")),
            "mean_kld": _distribution(_numeric_values(records, "kld")),
        },
        "latency": latency,
        "resources": {
            "peak_rss_bytes": max(_numeric_values(records, "peak_rss_bytes"), default=None),
            "peak_vram_bytes": max(_numeric_values(records, "peak_vram_bytes"), default=None),
            "artifact_size_bytes": _artifact_size_bytes(artifact_path),
        },
        "diagnostics": {
            "timestamp_mae_ms": _distribution(_numeric_values(records, "timestamp_mae_ms")),
            "timestamp_coverage": (
                sum(record.get("timestamp_mae_ms") is not None for record in records) / len(records)
            ),
            "silence_hallucination_rate": (
                sum(bool(str(record["prediction"]).strip()) for record in silence_records)
                / len(silence_records)
                if silence_records
                else None
            ),
            "truncation_rate": (
                sum(bool(record.get("truncated", False)) for record in records) / len(records)
            ),
        },
    }


def write_evaluation_report(
    records_path: Path, output_path: Path, artifact_path: Path | None = None
) -> dict[str, Any]:
    """Score a JSONL record file and write one portable JSON report."""
    report = build_evaluation_report(load_records(records_path), artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
