from __future__ import annotations

import numpy as np

from scripts.evaluate_paired_crown_configs import robust_selection_diagnostics


def test_robust_selection_accepts_distributed_candidate_gain() -> None:
    result = robust_selection_diagnostics(
        np.asarray([-0.12, -0.08, -0.10, -0.09, -0.11]),
        ["001", "002", "003", "004", "005"],
        (-0.12, -0.07),
        "candidate",
        "reference",
    )

    assert result["candidate_selected"] is True
    assert result["recommended_label"] == "candidate"
    assert all(result["criteria"].values())


def test_robust_selection_rejects_single_case_driven_mean_gain() -> None:
    result = robust_selection_diagnostics(
        np.asarray([-1.0, 0.10, 0.10, 0.10, 0.10]),
        ["outlier", "002", "003", "004", "005"],
        (-0.40, 0.05),
        "candidate",
        "reference",
    )

    assert result["candidate_selected"] is False
    assert result["recommended_label"] == "reference"
    assert result["most_influential_case"]["case_id"] == "outlier"
    assert (
        result["criteria"][
            "candidate_gain_survives_every_leave_one_case_out_check"
        ]
        is False
    )
