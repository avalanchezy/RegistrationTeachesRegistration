from __future__ import annotations

from collections import Counter

from scripts.select_candidate_pool import limited_candidate_pool


def test_limited_candidate_pool_keeps_equal_per_run_budget() -> None:
    rows = []
    for run_name in ("direct", "probability"):
        rows.extend(
            {
                "source_candidate_run": run_name,
                "selection_score_mm": float(index),
            }
            for index in range(25)
        )

    selected = limited_candidate_pool(rows, "lower", top_per_run=20)

    assert len(selected) == 40
    assert Counter(row["source_candidate_run"] for row in selected) == {
        "direct": 20,
        "probability": 20,
    }


def test_limited_candidate_pool_zero_preserves_all_rows() -> None:
    rows = [{"selection_score_mm": 1.0}, {"selection_score_mm": 2.0}]
    assert limited_candidate_pool(rows, "upper", top_per_run=0) is rows
