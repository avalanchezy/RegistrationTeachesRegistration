from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.select_consistent_pseudolabels import _largest_consistent_cluster


def translation(x: float) -> np.ndarray:
    transform = np.eye(4)
    transform[0, 3] = x
    return transform


def test_two_of_three_cluster_rejects_one_outlier() -> None:
    points = np.zeros((32, 3), dtype=np.float64)
    cluster, distances = _largest_consistent_cluster(
        [translation(0.0), translation(0.4), translation(12.0)],
        points,
        max_disagreement_mm=2.0,
        min_size=2,
    )
    assert cluster == [0, 1]
    assert distances[0, 2] > 10.0


def test_cluster_never_mixes_chirality() -> None:
    points = np.arange(96, dtype=np.float64).reshape(32, 3) / 10.0
    reflection = np.eye(4)
    reflection[0, 0] = -1.0
    cluster, _ = _largest_consistent_cluster(
        [np.eye(4), translation(0.2), reflection],
        points,
        max_disagreement_mm=100.0,
        min_size=2,
    )
    assert cluster == [0, 1]
