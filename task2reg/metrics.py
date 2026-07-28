from __future__ import annotations

import numpy as np

from .geometry import transform_points


def rotation_error_deg(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Return the official trace-based geodesic error for the 3x3 blocks."""
    relative_rotation = prediction[:3, :3] @ ground_truth[:3, :3].T
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def evaluate_transform(source: np.ndarray, prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float | int]:
    predicted = transform_points(source, prediction)
    expected = transform_points(source, ground_truth)
    distances = np.linalg.norm(predicted - expected, axis=1)
    relative = prediction @ np.linalg.inv(ground_truth)
    return {
        "mean_tre_mm": float(np.mean(distances)),
        "median_tre_mm": float(np.median(distances)),
        "p95_tre_mm": float(np.quantile(distances, 0.95)),
        "translation_error_mm": float(np.linalg.norm(prediction[:3, 3] - ground_truth[:3, 3])),
        "rotation_error_deg": rotation_error_deg(prediction, ground_truth),
        "linear_frobenius_error": float(np.linalg.norm(prediction[:3, :3] - ground_truth[:3, :3])),
        "relative_linear_frobenius": float(np.linalg.norm(relative[:3, :3] - np.eye(3))),
        "predicted_chirality": int(np.sign(np.linalg.det(prediction[:3, :3]))),
        "ground_truth_chirality": int(np.sign(np.linalg.det(ground_truth[:3, :3]))),
        "chirality_correct": int(np.sign(np.linalg.det(prediction[:3, :3])) == np.sign(np.linalg.det(ground_truth[:3, :3]))),
    }
