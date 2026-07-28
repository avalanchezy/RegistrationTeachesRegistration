from __future__ import annotations

import pytest

from scripts.run_geometry_benchmark import _summary_row_from_result_payload


def result_payload() -> dict[str, object]:
    return {
        "record": {"case_id": "115", "jaw": "upper"},
        "target": {"name": "crown_probability_p0.50"},
        "source_variant": "pca_high_0.35",
        "registration": {
            "method": "pca+basin",
            "chirality": 1,
            "score": 0.8,
            "median_distance": 0.7,
            "p90_distance": 1.4,
            "overlap_2mm": 0.9,
        },
        "selection": {"score": 0.6, "rank_score": 0.7, "prior_angle_deg": 12.0},
        "metrics": {"mean_tre_mm": 1.1, "median_tre_mm": 1.0},
        "oracle_diagnostic": {
            "mean_tre_mm": 0.9,
            "unsupervised_rank": 3,
            "method": "pca+basin",
            "target": "crown_probability_p0.35",
        },
    }


def test_resume_payload_reconstructs_summary_row() -> None:
    row = _summary_row_from_result_payload(
        result_payload(),
        target_mode="crown-probability",
        chirality_mode="metadata",
        expected_case_id="115",
        expected_jaw="upper",
    )

    assert row["case_id"] == "115"
    assert row["target_mode"] == "crown-probability"
    assert row["mean_tre_mm"] == pytest.approx(1.1)
    assert row["oracle_mean_tre_mm"] == pytest.approx(0.9)


def test_resume_payload_rejects_another_case() -> None:
    with pytest.raises(ValueError, match="different case or jaw"):
        _summary_row_from_result_payload(
            result_payload(),
            target_mode="crown-probability",
            chirality_mode="metadata",
            expected_case_id="116",
            expected_jaw="upper",
        )
