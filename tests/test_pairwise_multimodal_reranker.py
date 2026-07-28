from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweep_pairwise_multimodal_reranker import (
    pairwise_target_values,
    rank_group,
    score_group,
)


class LowerFeatureWins:
    def predict_proba(self, differences: np.ndarray) -> np.ndarray:
        probability = 1.0 / (1.0 + np.exp(4.0 * differences[:, 0]))
        return np.column_stack((1.0 - probability, probability))


def test_pairwise_ranking_selects_consistent_winner() -> None:
    rows = [
        {"selection_score_mm": 0.3},
        {"selection_score_mm": 0.1},
        {"selection_score_mm": 0.2},
    ]
    features = np.asarray([[0.0], [1.0], [2.0]])
    args = SimpleNamespace(
        exclude_upper_opposite_axial=False,
        balance_candidate_runs=False,
        eval_opponents=3,
    )
    index, score = rank_group(LowerFeatureWins(), rows, features, "lower", 3, args)
    assert index == 0
    assert score > 0.9

    indices, scores = score_group(LowerFeatureWins(), rows, features, "lower", 3, args)
    assert set(indices) == {0, 1, 2}
    assert indices[int(np.argmax(scores))] == index


def test_pairwise_target_can_follow_official_metric_instead_of_tre() -> None:
    rows = [
        {"mean_tre_mm": 10.0, "official_balanced_error": 0.1},
        {"mean_tre_mm": 1.0, "official_balanced_error": 0.5},
    ]

    tre = pairwise_target_values(rows, [0, 1], "mean_tre_mm")
    official = pairwise_target_values(
        rows, [0, 1], "official_balanced_error"
    )

    assert tre[0] > tre[1]
    assert official[0] < official[1]
