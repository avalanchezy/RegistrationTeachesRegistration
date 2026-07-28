from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_official_registration_metrics import (
    official_metrics,
    parse_transform_key,
    rank_modes,
)


def test_parse_transform_key_roundtrip() -> None:
    transform = np.arange(16, dtype=np.float64).reshape(4, 4)
    parsed = parse_transform_key(",".join(str(value) for value in transform.reshape(-1)))
    assert np.array_equal(parsed, transform)


def test_official_metrics_reports_translation_and_rotation() -> None:
    truth = np.eye(4)
    prediction = np.eye(4)
    prediction[:3, 3] = [3.0, 4.0, 0.0]
    translation, rotation = official_metrics(prediction, truth)
    assert np.isclose(translation, 5.0)
    assert np.isclose(rotation, 0.0)


def test_rank_modes_averages_translation_and_rotation_ranks() -> None:
    summaries = {
        "direct": {
            "mean_translation_error_mm": 2.0,
            "mean_rotation_error_deg": 1.0,
        },
        "direct_probability": {
            "mean_translation_error_mm": 1.0,
            "mean_rotation_error_deg": 2.0,
        },
        "direct_guided": {
            "mean_translation_error_mm": 3.0,
            "mean_rotation_error_deg": 3.0,
        },
    }
    ranked = rank_modes(summaries)
    by_name = {row["mode"]: row for row in ranked}
    assert by_name["direct"]["official_average_rank"] == 1.5
    assert by_name["direct_probability"]["official_average_rank"] == 1.5
    assert by_name["direct_guided"]["official_average_rank"] == 3.0
    assert ranked[0]["mode"] == "direct"
