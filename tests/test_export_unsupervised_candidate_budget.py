from __future__ import annotations

from collections import Counter

from scripts.export_unsupervised_candidate_scores import export_indices


def test_export_indices_uses_per_run_budget() -> None:
    rows = []
    for run_name in ("direct", "guided", "probability"):
        rows.extend(
            {
                "source_candidate_run": run_name,
                "selection_score_mm": float(index),
            }
            for index in range(24)
        )

    indices = export_indices(rows, "lower", top_per_run=20)
    selected = [rows[index] for index in indices]

    assert len(indices) == 60
    assert Counter(row["source_candidate_run"] for row in selected) == {
        "direct": 20,
        "guided": 20,
        "probability": 20,
    }


def test_export_indices_zero_keeps_original_order() -> None:
    rows = [{"selection_score_mm": 2.0}, {"selection_score_mm": 1.0}]
    assert export_indices(rows, "upper", top_per_run=0) == [0, 1]


def test_export_indices_can_match_upper_deployment_filter() -> None:
    rows = [
        {
            "source_candidate_run": "guided",
            "selection_score_mm": 0.1,
            "target_metadata": {"axial_assignment": "lower"},
        },
        {
            "source_candidate_run": "guided",
            "selection_score_mm": 0.2,
            "target_metadata": {"axial_assignment": "upper"},
        },
    ]

    assert export_indices(
        rows,
        "upper",
        top_per_run=20,
        exclude_upper_opposite_axial=True,
    ) == [1]
