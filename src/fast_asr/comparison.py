"""Paired statistical comparison and explicit candidate promotion gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from fast_asr.evaluation import load_records

NORMALIZER = BasicTextNormalizer()


@dataclass(frozen=True)
class PromotionGate:
    """Acceptance thresholds stated before inspecting candidate results."""

    max_wer_regression: float = 0.01
    min_speedup: float = 1.1
    max_truncation_regression: float = 0.005


def _error_statistics(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Return per-utterance error and reference-word counts for fast resampling."""
    errors: list[int] = []
    reference_words: list[int] = []
    for record in records:
        reference = NORMALIZER(str(record["reference"])).strip()
        prediction = NORMALIZER(str(record["prediction"])).strip()
        if not reference:
            errors.append(0)
            reference_words.append(0)
            continue
        result = jiwer.process_words(reference, prediction)
        errors.append(result.substitutions + result.deletions + result.insertions)
        reference_words.append(result.hits + result.substitutions + result.deletions)
    return np.asarray(errors, dtype=np.int64), np.asarray(reference_words, dtype=np.int64)


def compare_records(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    gate: PromotionGate,
    bootstrap_samples: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute paired deltas, percentile CIs, and a machine-readable gate decision."""
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100.")
    if (
        gate.max_wer_regression < 0
        or gate.min_speedup <= 0
        or gate.max_truncation_regression < 0
    ):
        raise ValueError("Promotion thresholds must be non-negative and speedup positive.")
    baseline_by_id = {str(record["utterance_id"]): record for record in baseline_records}
    candidate_by_id = {str(record["utterance_id"]): record for record in candidate_records}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("Paired comparison requires identical utterance IDs.")
    identifiers = sorted(baseline_by_id)
    baseline = [baseline_by_id[identifier] for identifier in identifiers]
    candidate = [candidate_by_id[identifier] for identifier in identifiers]
    baseline_errors, reference_words = _error_statistics(baseline)
    candidate_errors, candidate_reference_words = _error_statistics(candidate)
    if not np.array_equal(reference_words, candidate_reference_words):
        raise ValueError("Paired records have different normalized references.")
    total_reference_words = int(reference_words.sum())
    if total_reference_words == 0:
        raise ValueError("Paired records contain no scorable reference words.")
    baseline_wer = float(baseline_errors.sum() / total_reference_words)
    candidate_wer = float(candidate_errors.sum() / total_reference_words)
    baseline_latencies = np.asarray(
        [float(record["e2e_latency_ms"]) for record in baseline], dtype=np.float64
    )
    candidate_latencies = np.asarray(
        [float(record["e2e_latency_ms"]) for record in candidate], dtype=np.float64
    )
    baseline_latency = float(baseline_latencies.sum())
    candidate_latency = float(candidate_latencies.sum())
    if baseline_latency <= 0 or candidate_latency <= 0:
        raise ValueError("Paired latencies must be positive.")
    speedup = baseline_latency / candidate_latency
    baseline_truncation = np.mean([bool(record.get("truncated", False)) for record in baseline])
    candidate_truncation = np.mean([bool(record.get("truncated", False)) for record in candidate])
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(identifiers), size=(bootstrap_samples, len(identifiers)), dtype=np.int32
    )
    sampled_words = reference_words[indices].sum(axis=1)
    valid = sampled_words > 0
    wer_deltas = (
        candidate_errors[indices].sum(axis=1)[valid]
        - baseline_errors[indices].sum(axis=1)[valid]
    ) / sampled_words[valid]
    speedups = baseline_latencies[indices].sum(axis=1) / candidate_latencies[indices].sum(axis=1)
    wer_delta = candidate_wer - baseline_wer
    truncation_delta = float(candidate_truncation - baseline_truncation)
    checks = {
        "wer": wer_delta <= gate.max_wer_regression,
        "speed": speedup >= gate.min_speedup,
        "truncation": truncation_delta <= gate.max_truncation_regression,
    }
    return {
        "schema_version": 1,
        "utterance_count": len(identifiers),
        "baseline_wer": baseline_wer,
        "candidate_wer": candidate_wer,
        "wer_delta": wer_delta,
        "wer_delta_ci95": list(map(float, np.percentile(wer_deltas, [2.5, 97.5]))),
        "speedup": speedup,
        "speedup_ci95": list(map(float, np.percentile(speedups, [2.5, 97.5]))),
        "truncation_delta": truncation_delta,
        "thresholds": gate.__dict__,
        "checks": checks,
        "promote": all(checks.values()),
    }


def write_comparison(
    baseline_path: Path,
    candidate_path: Path,
    output_path: Path,
    *,
    gate: PromotionGate,
    bootstrap_samples: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    report = compare_records(
        load_records(baseline_path),
        load_records(candidate_path),
        gate=gate,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
