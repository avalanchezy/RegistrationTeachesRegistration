from __future__ import annotations

import numpy as np

from scripts.evaluate_nested_global_mode_selection import (
    MODE_COMPLEXITY,
    apply_exact_fallback,
    mode_complexity,
    robust_mode_decision,
)


def test_toothseg_hybrid_has_higher_complexity_than_crown_baseline() -> None:
    assert MODE_COMPLEXITY["crown"] < MODE_COMPLEXITY["crown_toothseg"]


def test_dynamic_tuned_family_modes_share_tuned_complexity() -> None:
    assert mode_complexity("tuned_random_forest_extra_trees") == 1


def test_guided_profiles_and_combination_have_expected_complexity() -> None:
    assert mode_complexity("direct_guided_fine") == 2
    assert mode_complexity("direct_guided_high") == 2
    assert mode_complexity("direct_guided_all") == 4


def test_robust_mode_accepts_distributed_group_gain() -> None:
    result = robust_mode_decision(
        np.asarray([-0.12, -0.08, -0.10, -0.09, -0.11]),
        (-0.12, -0.07),
        "direct_probability",
    )

    assert result["candidate_selected"] is True
    assert result["recommended_mode"] == "direct_probability"


def test_robust_mode_rejects_outlier_driven_group_gain() -> None:
    result = robust_mode_decision(
        np.asarray([-1.0, 0.10, 0.10, 0.10, 0.10]),
        (-0.40, 0.05),
        "direct_guided",
    )

    assert result["candidate_selected"] is False
    assert result["recommended_mode"] == "direct"


def test_robust_mode_keeps_direct_without_extra_requirements() -> None:
    result = robust_mode_decision(
        np.asarray([0.0, 0.0, 0.0]),
        (0.0, 0.0),
        "direct",
    )

    assert result["candidate_selected"] is True
    assert result["recommended_mode"] == "direct"


def test_robust_mode_can_retain_incumbent_baseline() -> None:
    result = robust_mode_decision(
        np.asarray([-0.5, 0.2, 0.2, 0.2]),
        (-0.25, 0.20),
        "direct_guided",
        baseline_mode="incumbent",
    )

    assert result["candidate_selected"] is False
    assert result["recommended_mode"] == "incumbent"


def test_exact_fallback_replaces_every_mode_consistently() -> None:
    modes = {
        "incumbent": {("001", "upper"): 3.0, ("002", "upper"): 1.0},
        "direct": {("001", "upper"): 2.0, ("002", "upper"): 0.5},
    }

    replaced = apply_exact_fallback(modes, {("001", "upper"): 0.25})

    assert replaced == 1
    assert modes["incumbent"][("001", "upper")] == 0.25
    assert modes["direct"][("001", "upper")] == 0.25
    assert modes["direct"][("002", "upper")] == 0.5
