import numpy as np

from scripts.run_submission_inference import (
    CandidateSelection,
    choose_model_selection,
    predict_candidates,
)


class FixedEstimator:
    def __init__(self, log_predictions: list[float]) -> None:
        self.log_predictions = np.asarray(log_predictions, dtype=np.float64)

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert len(features) == len(self.log_predictions)
        return self.log_predictions


def test_single_model_selection_is_unchanged() -> None:
    payload = {"model": FixedEstimator([1.0, 0.2, 0.8])}
    prediction, index = predict_candidates(payload, "lower", np.zeros((3, 2)))
    assert index == 1
    assert np.allclose(prediction, np.expm1([1.0, 0.2, 0.8]))


def test_model_vote_uses_majority_choice() -> None:
    payload = {
        "model": [
            FixedEstimator([0.1, 0.5, 0.8]),
            FixedEstimator([0.2, 0.4, 0.9]),
            FixedEstimator([0.7, 0.1, 0.5]),
        ],
        "model_aggregation": "vote",
    }
    _, index = predict_candidates(payload, "lower", np.zeros((3, 2)))
    assert index == 0


def test_model_vote_breaks_tie_by_mean_prediction() -> None:
    payload = {
        "model": [
            FixedEstimator([0.1, 0.7, 0.8]),
            FixedEstimator([0.5, 0.2, 0.8]),
            FixedEstimator([0.6, 0.7, 0.3]),
        ],
        "model_aggregation": "vote",
    }
    _, index = predict_candidates(payload, "lower", np.zeros((3, 2)))
    assert index == 0


def test_heterogeneous_model_pair_uses_lower_predicted_tre() -> None:
    def selection(value: float) -> CandidateSelection:
        return CandidateSelection(
            transform=np.eye(4),
            predicted_tre_mm=value,
            unsupervised_rank=1,
            target="target",
            source_variant="source",
            method="method",
            full_p90_mm=10.0,
            candidate_run="run",
        )

    name, selected = choose_model_selection(
        [("new", selection(2.4)), ("legacy", selection(1.8))]
    )
    assert name == "legacy"
    assert selected.predicted_tre_mm == 1.8
