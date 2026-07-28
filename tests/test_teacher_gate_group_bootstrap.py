from __future__ import annotations

import numpy as np

from scripts.calibrate_teacher_predicted_tre_gate import grouped_bootstrap_diagnostics


def test_grouped_bootstrap_keeps_duplicate_cbct_cases_in_one_unit() -> None:
    rows = [
        {"case_id": "001", "true_tre_mm": 0.5},
        {"case_id": "001", "true_tre_mm": 0.7},
        {"case_id": "002", "true_tre_mm": 1.0},
        {"case_id": "002", "true_tre_mm": 1.2},
        {"case_id": "003", "true_tre_mm": 4.0},
        {"case_id": "003", "true_tre_mm": 4.2},
    ]
    case_groups = {"001": "shared", "002": "shared", "003": "independent"}
    result = grouped_bootstrap_diagnostics(
        rows,
        np.asarray([True, True, True, True, False, False]),
        case_groups,
        samples=200,
        seed=7,
    )

    assert result["cbct_groups"] == 2
    assert result["accepted_cbct_groups"] == 1
    assert 100 <= result["bootstrap_samples"] <= 200
    assert result["bootstrap_p90_true_tre_ci95_mm"][1] <= 1.2
