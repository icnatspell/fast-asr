from __future__ import annotations

import json
from pathlib import Path

from fast_asr.screening import write_screening_plan


def test_screening_plan_is_candidate_by_split_matrix(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    models = [tmp_path / "model-a", tmp_path / "model-b"]
    write_screening_plan(
        models,
        tmp_path / "results",
        plan_path,
        splits=["validation.clean", "validation.other"],
        max_samples=128,
        threads=4,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(plan["jobs"]) == 4
    assert {job["max_samples"] for job in plan["jobs"]} == {128}
    assert {job["threads"] for job in plan["jobs"]} == {4}
