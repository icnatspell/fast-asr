from __future__ import annotations

from fast_asr.evaluation import build_evaluation_report


def test_report_contains_quality_latency_and_diagnostics() -> None:
    report = build_evaluation_report(
        [
            {
                "utterance_id": "speech",
                "reference": "a small test",
                "prediction": "a test",
                "audio_duration_s": 2.0,
                "e2e_latency_ms": 100.0,
                "encoder_latency_ms": 30.0,
                "first_token_latency_ms": 45.0,
                "decode_latency_ms": 70.0,
                "tpot_ms": 35.0,
                "generated_tokens": 2,
                "peak_rss_bytes": 1000,
                "token_agreement": 0.8,
                "kld": 0.1,
                "timestamp_mae_ms": 12.0,
            },
            {
                "utterance_id": "silence",
                "reference": "",
                "prediction": "hallucination",
                "audio_duration_s": 1.0,
                "e2e_latency_ms": 50.0,
                "is_silence": True,
                "truncated": True,
            },
        ]
    )

    assert report["quality"]["wer"] == 1 / 3
    assert report["quality"]["deletion_rate"] == 1 / 3
    assert report["latency"]["rtfx"] == 20.0
    assert report["resources"]["peak_rss_bytes"] == 1000.0
    assert report["diagnostics"]["silence_hallucination_rate"] == 1.0
    assert report["diagnostics"]["truncation_rate"] == 0.5
