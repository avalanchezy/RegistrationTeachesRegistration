from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_joint_ensemble_scores import (
    precompute_pair_matrices,
    retained_method_order,
)
from scripts.sweep_joint_pair_reranker import PairPrior, pair_geometry


def item(angles: list[float], translation: list[float], score: float) -> dict:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    transform[:3, 3] = translation
    return {"row": {"transform": transform, "chirality": 1}, "prediction_mm": score}


def test_vectorized_pair_geometry_matches_scalar_implementation() -> None:
    upper = [item([4, -2, 8], [2, 3, 4], 0.2), item([-3, 1, 5], [4, 1, 2], 0.4)]
    lower = [item([1, 3, -2], [-1, 2, 6], 0.3), item([5, 2, 7], [0, 1, 3], 0.5)]
    priors = [
        PairPrior(
            relative_rotation=Rotation.from_euler("z", 3, degrees=True).as_matrix(),
            relative_translation=np.asarray([1.0, -2.0, 0.5]),
        )
    ]
    data = precompute_pair_matrices(upper, lower, priors)
    for upper_index, upper_item in enumerate(upper):
        for lower_index, lower_item in enumerate(lower):
            angle, translation = pair_geometry(upper_item, lower_item, priors[0])
            assert np.isclose(data["angle"][upper_index, lower_index], angle)
            assert np.isclose(data["translation"][upper_index, lower_index], translation)


def test_official_method_prefilter_uses_official_metrics() -> None:
    policies = {
        "best_tre": {
            "raw": {"mean_tre_mm": 1.0},
            "exact_fallback": {"mean_tre_mm": 1.0, "p90_tre_mm": 2.0},
            "official": {
                "exact_fallback_official_balanced_error": 0.6,
                "exact_fallback_mean_rotation_error_deg": 3.0,
                "exact_fallback_mean_translation_error_mm": 6.0,
            },
        },
        "best_official": {
            "raw": {"mean_tre_mm": 1.4},
            "exact_fallback": {"mean_tre_mm": 1.4, "p90_tre_mm": 2.4},
            "official": {
                "exact_fallback_official_balanced_error": 0.3,
                "exact_fallback_mean_rotation_error_deg": 1.5,
                "exact_fallback_mean_translation_error_mm": 3.0,
            },
        },
    }

    assert retained_method_order(policies, "tre")[0] == "best_tre"
    assert retained_method_order(policies, "official")[0] == "best_official"
