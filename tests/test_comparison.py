from __future__ import annotations

from fast_asr.comparison import PromotionGate, compare_records


def _record(identifier: str, prediction: str, latency: float) -> dict[str, object]:
    return {
        "utterance_id": identifier,
        "reference": "a short sentence",
        "prediction": prediction,
        "audio_duration_s": 2.0,
        "e2e_latency_ms": latency,
        "truncated": False,
    }


def test_paired_comparison_is_deterministic_and_applies_gate() -> None:
    baseline = [_record("a", "a short sentence", 100), _record("b", "a sentence", 100)]
    candidate = [_record("a", "a short sentence", 50), _record("b", "a sentence", 50)]
    first = compare_records(
        baseline, candidate, gate=PromotionGate(max_wer_regression=0), bootstrap_samples=100
    )
    second = compare_records(
        baseline, candidate, gate=PromotionGate(max_wer_regression=0), bootstrap_samples=100
    )
    assert first == second
    assert first["speedup"] == 2
    assert first["promote"] is True
