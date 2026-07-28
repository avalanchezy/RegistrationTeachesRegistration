from __future__ import annotations

import numpy as np

from task2reg.threshold_consensus import (
    ConsensusConfig,
    apply_joint_gate,
    evidence_key,
    select_consensus,
)
from task2reg.candidate_learning import FEATURE_NAMES, candidate_features
from task2reg.priors import RotationPrior


def candidate(offset, view, family, threshold, side):
    transform = np.eye(4)
    transform[0, 3] = offset
    return {
        "transform": transform.tolist(),
        "candidate_view": view,
        "target": family,
        "target_metadata": {"mode": family, "threshold": threshold},
        "source_variant": f"pca_{side}_0.30",
        "selection_score_mm": 0.5,
        "fit_score_mm": 0.5,
        "fit_p90_mm": 3.0,
        "target_coverage_2mm": 0.5,
    }


def test_evidence_key_collapses_crop_fraction():
    first = candidate(0.0, "A", "threshold", 1200, "low")
    second = dict(first, source_variant="pca_low_0.40")
    assert evidence_key(first) == evidence_key(second)


def test_consensus_requires_diverse_evidence():
    rows = [
        candidate(0.0, "A", "threshold", 1000, "low"),
        candidate(0.2, "A", "threshold_aggregate", 1200, "high"),
        candidate(0.1, "B", "threshold", 1400, "low"),
        candidate(20.0, "C", "threshold_adaptive", 1600, "high"),
    ]
    distances = np.abs(np.subtract.outer([0.0, 0.2, 0.1, 20.0], [0.0, 0.2, 0.1, 20.0]))
    config = ConsensusConfig(1.0, 1.0, 2, 2, 3, 2, 3)
    selected = select_consensus(rows, distances, config)
    assert selected is not None
    assert selected["consensus_count"] == 3
    assert selected["view_count"] == 2


def test_joint_require_removes_incomplete_case():
    config = ConsensusConfig(1.0, 1.0, 1, 1, 1, 1, 1, joint_mode="require")
    upper = {"candidate": candidate(0.0, "A", "threshold", 1000, "low")}
    accepted, diagnostics = apply_joint_gate({("001", "upper"): upper}, config)
    assert accepted == {}
    assert diagnostics["001"]["joint_passed"] == 0


def test_candidate_features_accept_missing_geometry_features():
    row = candidate(0.0, "A", "threshold", 1000, "low")
    row.update(
        {
            "predicted_full_centroid": [0.0, 0.0, 0.0],
            "jaw_reference_center": [0.0, 0.0, 0.0],
            "method": "pca",
            "chirality": 1,
        }
    )
    prior = RotationPrior(np.eye(3), 1)
    assert candidate_features(row, prior).shape == (len(FEATURE_NAMES),)
