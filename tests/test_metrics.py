from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.metrics import evaluate_transform, rotation_error_deg


def transform(linear: np.ndarray, translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = linear
    matrix[:3, 3] = translation
    return matrix


def test_rotation_error_matches_known_proper_rotation() -> None:
    ground_truth = transform(np.eye(3))
    prediction = transform(Rotation.from_euler("z", 37.0, degrees=True).as_matrix())
    assert np.isclose(rotation_error_deg(prediction, ground_truth), 37.0)


def test_rotation_error_handles_shared_reflection_protocol() -> None:
    reflection = np.diag([-1.0, 1.0, 1.0])
    ground_truth = transform(reflection)
    prediction = transform(
        Rotation.from_euler("x", 23.0, degrees=True).as_matrix() @ reflection
    )
    assert np.isclose(rotation_error_deg(prediction, ground_truth), 23.0)


def test_rotation_error_preserves_official_trace_semantics_for_chirality_mismatch() -> None:
    ground_truth = transform(np.eye(3))
    prediction = transform(np.diag([-1.0, 1.0, 1.0]))
    assert np.isclose(rotation_error_deg(prediction, ground_truth), 90.0)


def test_evaluate_transform_reports_official_translation_and_rotation() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    ground_truth = transform(np.eye(3), (1.0, 2.0, 3.0))
    prediction = transform(
        Rotation.from_euler("y", 12.0, degrees=True).as_matrix(),
        (4.0, 6.0, 3.0),
    )
    metrics = evaluate_transform(source, prediction, ground_truth)
    assert np.isclose(metrics["translation_error_mm"], 5.0)
    assert np.isclose(metrics["rotation_error_deg"], 12.0)
