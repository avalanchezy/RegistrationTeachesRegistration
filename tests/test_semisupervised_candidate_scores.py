from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.train_semisupervised_candidate_reranker import candidate_predictions


class IdentityRegressor:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return features[:, 0]


def test_candidate_predictions_preserve_pool_indices() -> None:
    key = ("001", "lower")
    rows = [
        {"selection_score_mm": 0.2, "mean_tre_mm": 2.0, "candidate_run": "a"},
        {"selection_score_mm": 0.1, "mean_tre_mm": 1.0, "candidate_run": "b"},
    ]
    args = SimpleNamespace(
        exclude_upper_opposite_axial=False,
        balance_candidate_runs=False,
        eval_top_candidates=2,
        target_transform="identity",
    )
    output = candidate_predictions(
        IdentityRegressor(),
        {key: rows},
        {key: np.asarray([[0.2], [0.1]])},
        {"001"},
        args,
        "weight_0",
    )
    assert [row["candidate_index"] for row in output] == [1, 0]
    assert [row["ensemble_score"] for row in output] == [0.1, 0.2]
