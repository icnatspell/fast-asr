from __future__ import annotations

import json
from pathlib import Path

from fast_asr.result_table import summary_row, write_result_tables


def test_result_table_flattens_and_writes_summary(tmp_path: Path) -> None:
    summary = tmp_path / "candidate" / "dev-clean" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "utterance_count": 10,
                "quality": {"wer": 0.1, "cer": 0.05},
                "latency": {
                    "rtfx": 40,
                    "first_token_ms": {"p50": 10, "p95": 20},
                    "decoder_tokens_per_second": 200,
                    "time_per_output_token_ms": {"p50": 4, "p95": 5},
                },
                "diagnostics": {"truncation_rate": 0.01},
                "resources": {"artifact_size_bytes": 2_000_000},
            }
        ),
        encoding="utf-8",
    )
    row = summary_row(summary)
    assert row["experiment_id"] == "unknown"
    assert row["config"] == "candidate"
    assert row["wer_percent"] == 10
    csv_path = tmp_path / "results.csv"
    markdown_path = tmp_path / "results.md"
    write_result_tables([summary], csv_path, markdown_path)
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "wer_percent" in csv_text
    assert "\r" not in csv_text
    assert "candidate" in markdown_path.read_text(encoding="utf-8")
