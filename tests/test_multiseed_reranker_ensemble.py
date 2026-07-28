import joblib
import numpy as np
import pytest

from scripts.evaluate_multiseed_reranker_ensemble import (
    aggregate_predictions,
    atomic_joblib_dump,
    select_ensemble_pair,
    validate_prediction_layout,
)
from scripts.sweep_joint_pair_reranker import PairPrior


def test_vote_prefers_candidate_selected_by_most_models() -> None:
    predictions = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.2, 0.5], [0.3, 0.1, 0.2]])
    scores = aggregate_predictions(predictions, "vote")
    assert int(np.argmin(scores)) == 1


def test_rank_mean_is_invariant_to_seed_calibration_scale() -> None:
    predictions = np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    scores = aggregate_predictions(predictions, "rank_mean")
    assert int(np.argmin(scores)) == 0


def test_joint_ensemble_can_prefer_geometrically_consistent_pair() -> None:
    identity = np.eye(4)
    rotated = np.eye(4)
    rotated[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
    upper = [{"row": {"transform": identity, "chirality": 1}, "prediction_mm": 0.2}]
    lower = [
        {"row": {"transform": rotated, "chirality": 1}, "prediction_mm": 0.1},
        {"row": {"transform": identity, "chirality": 1}, "prediction_mm": 0.2},
    ]
    pair = select_ensemble_pair(
        upper,
        lower,
        [PairPrior(relative_rotation=np.eye(3), relative_translation=np.zeros(3))],
        top_k=2,
        angle_weight=0.01,
        translation_weight=0.0,
        allow_chirality_mismatch=False,
    )
    assert np.allclose(pair["lower"]["row"]["transform"], identity)


def test_atomic_joblib_dump_replaces_temporary_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "seed_oof.joblib"
    atomic_joblib_dump({"value": np.asarray([1.0, 2.0])}, checkpoint)
    assert np.array_equal(joblib.load(checkpoint)["value"], [1.0, 2.0])
    assert not checkpoint.with_suffix(checkpoint.suffix + ".tmp").exists()


def test_prediction_layout_rejects_stale_candidate_pool(tmp_path) -> None:
    key = ("001", "upper")
    checkpoint = tmp_path / "seed_oof.partial.joblib"
    with pytest.raises(RuntimeError, match="does not match candidate pool"):
        validate_prediction_layout(
            {key: np.zeros(2, dtype=np.float64)},
            {key: [{}, {}, {}]},
            checkpoint,
        )
