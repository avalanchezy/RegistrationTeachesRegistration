from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.geometry import register_geometry, transform_points


def asymmetric_arch(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = np.linspace(-1.25, 1.05, 1800)
    radius = 34.0 + 3.0 * np.sin(2.3 * theta)
    points = np.column_stack(
        [
            radius * np.sin(theta),
            0.55 * radius * np.cos(theta),
            2.0 * np.sin(3.0 * theta) + 0.04 * np.arange(len(theta)),
        ]
    )
    return points + rng.normal(scale=0.03, size=points.shape)


def make_transform(chirality: int) -> np.ndarray:
    linear = Rotation.from_euler("xyz", [24.0, -17.0, 38.0], degrees=True).as_matrix()
    if chirality < 0:
        linear = linear @ np.diag([-1.0, 1.0, 1.0])
    transform = np.eye(4)
    transform[:3, :3] = linear
    transform[:3, 3] = [18.0, -31.0, 12.0]
    return transform


def test_pca_icp_recovers_proper_transform() -> None:
    source = asymmetric_arch()
    truth = make_transform(1)
    target = transform_points(source, truth)
    result = register_geometry(source, target, methods=("pca",), pca_refine_top_k=8)[0]
    error = np.linalg.norm(transform_points(source, result.transform) - target, axis=1).mean()
    assert result.chirality == 1
    assert error < 0.1


def test_pca_icp_recovers_reflection_transform() -> None:
    source = asymmetric_arch()
    truth = make_transform(-1)
    target = transform_points(source, truth)
    result = register_geometry(source, target, methods=("pca",), pca_refine_top_k=8)[0]
    error = np.linalg.norm(transform_points(source, result.transform) - target, axis=1).mean()
    assert result.chirality == -1
    assert error < 0.1


def test_open3d_failure_skips_only_feature_initialization() -> None:
    source = asymmetric_arch()
    truth = make_transform(1)
    target = transform_points(source, truth)
    with patch(
        "task2reg.geometry._open3d_feature_initialization",
        side_effect=RuntimeError("no feature correspondences"),
    ):
        results = register_geometry(source, target, methods=("pca", "fgr"), pca_refine_top_k=2)
    assert results
    assert all(result.method == "pca" for result in results)
