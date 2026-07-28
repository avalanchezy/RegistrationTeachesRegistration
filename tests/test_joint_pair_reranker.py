from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.sweep_joint_pair_reranker import PairPrior, pair_geometry, select_pair


def candidate(angle_deg: float, prediction_mm: float, chirality: int = 1) -> dict:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("z", angle_deg, degrees=True).as_matrix()
    if chirality < 0:
        transform[:3, :3] = transform[:3, :3] @ np.diag((-1.0, 1.0, 1.0))
    return {
        "prediction_mm": prediction_mm,
        "candidate_index": 0,
        "row": {
            "transform": transform.tolist(),
            "chirality": chirality,
        },
    }


def test_pair_geometry_removes_protocol_reflection() -> None:
    prior = PairPrior(np.eye(3), np.zeros(3))
    upper = candidate(0.0, 1.0, chirality=-1)
    lower = candidate(4.0, 1.0, chirality=-1)

    angle, translation = pair_geometry(upper, lower, prior)

    assert np.isclose(angle, 4.0)
    assert np.isclose(translation, 0.0)


def test_angle_prior_can_break_a_model_near_tie() -> None:
    prior = PairPrior(np.eye(3), np.zeros(3))
    upper = [candidate(0.0, 1.0)]
    lower = [candidate(12.0, 0.9), candidate(2.0, 1.0)]

    independent = select_pair(upper, lower, prior, 2, 0.0, 0.0, False)
    regularized = select_pair(upper, lower, prior, 2, 0.05, 0.0, False)

    assert independent["lower"] is lower[0]
    assert regularized["lower"] is lower[1]
    assert regularized["relative_angle_deg"] < independent["relative_angle_deg"]


def test_chirality_mismatch_is_not_selected() -> None:
    prior = PairPrior(np.eye(3), np.zeros(3))
    upper = [candidate(0.0, 1.0, chirality=1)]
    lower = [candidate(0.0, 0.1, chirality=-1), candidate(0.0, 1.0, chirality=1)]

    selected = select_pair(upper, lower, prior, 2, 0.0, 0.0, False)

    assert selected["lower"] is lower[1]
