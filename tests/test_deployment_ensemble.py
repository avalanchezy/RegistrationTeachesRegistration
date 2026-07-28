import numpy as np

from task2reg.deployment_ensemble import (
    aggregate_seed_costs,
    blended_candidates,
    ensemble_feature_matrix,
    select_joint_pair,
    unsupervised_candidates,
)
from task2reg.priors import RotationPrior
from scripts.run_submission_inference import align_model_features


def transform(translation: tuple[float, float, float]) -> list[list[float]]:
    value = np.eye(4)
    value[:3, 3] = translation
    return value.tolist()


def test_vote_aggregation_uses_seed_winners() -> None:
    costs = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 1.0],
        ]
    )
    aggregate = aggregate_seed_costs(costs, "vote")
    assert int(np.argmin(aggregate)) == 0


def test_rank_blend_respects_alpha_orientation() -> None:
    rows = [{"id": index} for index in range(3)]
    indices = [0, 1, 2]
    regression = np.asarray([[0.0, 1.0, 2.0]])
    pairwise = np.asarray([[2.0, 1.0, 0.0]])
    regression_only = blended_candidates(
        rows, indices, regression, pairwise, "mean", "mean", 1.0
    )
    pairwise_only = blended_candidates(
        rows, indices, regression, pairwise, "mean", "mean", 0.0
    )
    assert regression_only[0]["candidate_index"] == 0
    assert pairwise_only[0]["candidate_index"] == 2


def test_unsupervised_candidates_use_recorded_geometry_score() -> None:
    rows = [
        {"rank_score_mm": 0.8},
        {"rank_score_mm": 0.2},
        {"rank_score_mm": 0.5},
    ]
    ranked = unsupervised_candidates(rows, [0, 1, 2], "rank_score_mm")
    assert [row["candidate_index"] for row in ranked] == [1, 2, 0]
    assert ranked[0]["score"] == 0.2


def test_joint_selection_can_override_independent_pair() -> None:
    upper = [
        {
            "row": {"transform": transform((0.0, 0.0, 0.0)), "chirality": 1},
            "score": 0.0,
        },
        {
            "row": {"transform": transform((10.0, 0.0, 0.0)), "chirality": 1},
            "score": 0.1,
        },
    ]
    lower = [
        {
            "row": {"transform": transform((20.0, 0.0, 0.0)), "chirality": 1},
            "score": 0.0,
        },
        {
            "row": {"transform": transform((0.0, 0.0, 0.0)), "chirality": 1},
            "score": 0.1,
        },
    ]
    pair = select_joint_pair(
        upper,
        lower,
        np.eye(3),
        np.zeros(3),
        top_k=2,
        angle_weight=0.0,
        translation_weight=1.0,
    )
    assert pair["upper"] is upper[0]
    assert pair["lower"] is lower[1]


def test_deployment_feature_width_is_checked() -> None:
    row = {
        "transform": transform((0.0, 0.0, 0.0)),
        "predicted_full_centroid": [0.0, 0.0, 0.0],
        "jaw_reference_center": [0.0, 0.0, 0.0],
        "selection_score_mm": 1.0,
        "target_metadata": {},
    }
    prior = RotationPrior(
        mean_rotation=np.eye(3),
        training_angles_deg=np.empty(0),
    )
    payload = {
        "group_context_features": True,
        "roi_view_feature": True,
        "modality_features": False,
        "feature_names": ("wrong",),
    }
    try:
        ensemble_feature_matrix([row], prior, "upper", payload)
    except ValueError as error:
        assert "unavailable features" in str(error)
    else:
        raise AssertionError("Expected a feature-width validation error")


def test_legacy_feature_payload_is_aligned_by_name() -> None:
    features = np.asarray([[10.0, 20.0, 30.0]])
    aligned = align_model_features(
        features,
        ("first", "new_middle", "last"),
        {"feature_names": ("first", "last")},
    )
    np.testing.assert_array_equal(aligned, [[10.0, 30.0]])
