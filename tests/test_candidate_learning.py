from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.candidate_learning import (
    CROWN_CONSISTENCY_FEATURE_NAMES,
    CROWN_REFINEMENT_FEATURE_NAMES,
    GROUP_FEATURE_NAMES,
    MULTIMODAL_GROUP_FEATURE_NAMES,
    ROI_GROUP_FEATURE_NAMES,
    candidate_group_features,
    candidate_multimodal_group_features,
    is_opposite_axial_target,
)
from task2reg.priors import RotationPrior


def make_candidate(target: str, assignment: str, shift: float) -> dict:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = shift
    return {
        "transform": transform.tolist(),
        "predicted_full_centroid": [shift, 0.0, 0.0],
        "jaw_reference_center": [0.0, 0.0, 0.0],
        "selection_score_mm": 1.0 + shift,
        "fit_score_mm": 0.5 + shift,
        "target_trimmed_score_mm": 0.75 + shift,
        "source_variant": "pca_high_0.25",
        "method": "pca+basin",
        "target": target,
        "target_metadata": {
            "mode": "threshold",
            "threshold": 1600,
            "component_voxels": 10000,
            "axial_assignment": assignment,
        },
        "candidate_jaw": "upper",
    }


def test_group_features_and_opposite_arch_flag() -> None:
    candidates = [
        make_candidate("axial_upper_threshold_1600", "upper", 0.0),
        make_candidate("axial_lower_threshold_1600", "lower", 1.0),
    ]
    prior = RotationPrior(np.eye(3), np.asarray([0.0]))
    features = candidate_group_features(candidates, prior, "upper")

    assert features.shape == (2, len(GROUP_FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert not is_opposite_axial_target(candidates[0], "upper")
    assert is_opposite_axial_target(candidates[1], "upper")
    opposite_column = GROUP_FEATURE_NAMES.index("context_opposite_axial_target")
    assert features[:, opposite_column].tolist() == [0.0, 1.0]
    crown_available = GROUP_FEATURE_NAMES.index("crown_consistency_available")
    assert features[:, crown_available].tolist() == [0.0, 0.0]

    candidates[0]["crown_symmetric_trim20_mm"] = 0.75
    candidates[0]["crown_source_trim20_mm"] = 0.70
    candidates[0]["crown_target_trim20_mm"] = 0.80
    crown_features = candidate_group_features(candidates, prior, "upper")
    assert crown_features[0, crown_available] == 1.0
    assert len(CROWN_CONSISTENCY_FEATURE_NAMES) == 9
    assert len(CROWN_REFINEMENT_FEATURE_NAMES) == 6

    candidates[0]["crown_refinement_initial_trim20_mm"] = 1.0
    candidates[0]["crown_refinement_improvement_mm"] = 0.25
    candidates[0]["crown_refinement_center_motion_mm"] = 0.5
    candidates[0]["crown_refinement_angle_deg"] = 1.5
    candidates[0]["crown_refinement_alpha"] = 0.5
    refined_features = candidate_group_features(candidates, prior, "upper")
    refinement_available = GROUP_FEATURE_NAMES.index(
        "crown_refinement_available"
    )
    assert refined_features[:, refinement_available].tolist() == [1.0, 0.0]
    crown_rank = GROUP_FEATURE_NAMES.index("context_crown_symmetric_rank")
    assert np.isfinite(refined_features[:, crown_rank]).all()

    candidates[1]["candidate_run"] = "runs/view_roi_low_augmented"
    roi_features = candidate_group_features(
        candidates, prior, "upper", include_roi_view=True
    )
    assert roi_features.shape == (2, len(ROI_GROUP_FEATURE_NAMES))
    roi_column = ROI_GROUP_FEATURE_NAMES.index("context_roi_view")
    assert roi_features[:, roi_column].tolist() == [0.0, 1.0]
    low_column = ROI_GROUP_FEATURE_NAMES.index("context_roi_low_profile")
    assert roi_features[:, low_column].tolist() == [0.0, 1.0]


def test_multimodal_features_encode_toothseg_quality() -> None:
    threshold = make_candidate("axial_upper_threshold_1600", "upper", 0.0)
    toothseg = make_candidate("toothseg_crown_0.45", "", 1.0)
    toothseg["target_metadata"] = {
        "mode": "toothseg",
        "detected_teeth": 12,
        "crown_fraction": 0.45,
    }
    prior = RotationPrior(np.eye(3), np.asarray([0.0]))

    features = candidate_multimodal_group_features(
        [threshold, toothseg], prior, "upper"
    )

    assert features.shape == (2, len(MULTIMODAL_GROUP_FEATURE_NAMES))
    assert features[:, MULTIMODAL_GROUP_FEATURE_NAMES.index("target_toothseg")].tolist() == [
        0.0,
        1.0,
    ]
    assert np.allclose(
        features[:, MULTIMODAL_GROUP_FEATURE_NAMES.index("toothseg_detected_fraction")],
        [0.0, 0.75],
    )
    assert np.allclose(
        features[:, MULTIMODAL_GROUP_FEATURE_NAMES.index("toothseg_crown_fraction")],
        [0.0, 0.45],
    )


def test_cross_run_consensus_ignores_same_run_duplicates() -> None:
    candidates = [
        make_candidate("crown", "upper", 0.0),
        make_candidate("crown", "upper", 0.1),
        make_candidate("crown-probability", "upper", 0.02),
        make_candidate("crown-guided", "upper", 5.0),
    ]
    candidates[0]["candidate_run"] = "runs/direct"
    candidates[1]["candidate_run"] = "runs/direct"
    candidates[2]["candidate_run"] = "runs/probability"
    candidates[3]["candidate_run"] = "runs/guided"
    prior = RotationPrior(np.eye(3), np.asarray([0.0]))

    features = candidate_group_features(candidates, prior, "upper")
    available = GROUP_FEATURE_NAMES.index("context_cross_run_available")
    nearest = GROUP_FEATURE_NAMES.index("context_cross_run_nearest_distance")
    mean_nearest = GROUP_FEATURE_NAMES.index(
        "context_cross_run_mean_nearest_distance"
    )
    support = GROUP_FEATURE_NAMES.index("context_cross_run_support_0p25")

    assert features[:, available].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert features[0, nearest] < features[3, nearest]
    assert features[0, mean_nearest] < features[3, mean_nearest]
    assert features[0, support] > features[3, support]

    same_run = [
        make_candidate("crown", "upper", 0.0),
        make_candidate("crown", "upper", 0.1),
    ]
    for candidate in same_run:
        candidate["candidate_run"] = "runs/direct"
    single_run_features = candidate_group_features(same_run, prior, "upper")
    assert single_run_features[:, available].tolist() == [0.0, 0.0]
    assert single_run_features[:, support].tolist() == [0.0, 0.0]


def test_transferred_candidate_preserves_roi_view_provenance() -> None:
    candidate = make_candidate("threshold_1600", "upper", 0.0)
    candidate["candidate_run"] = "transferred_candidates"
    candidate["source_candidate_run"] = "runs/roi_low_profile"
    candidate["target_metadata"]["volume_path"] = "query/CBCT.nii.gz"
    candidate["target_metadata"]["source_volume_path"] = (
        "source/TrainROI/crop.nii.gz"
    )
    prior = RotationPrior(np.eye(3), np.asarray([0.0]))
    features = candidate_group_features(
        [candidate], prior, "upper", include_roi_view=True
    )
    assert features[0, ROI_GROUP_FEATURE_NAMES.index("context_roi_view")] == 1.0
    assert features[0, ROI_GROUP_FEATURE_NAMES.index("context_roi_low_profile")] == 1.0
