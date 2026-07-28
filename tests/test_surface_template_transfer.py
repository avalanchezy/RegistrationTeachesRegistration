from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from task2reg.geometry import transform_points
from task2reg.surface_template_transfer import match_surface_template


def asymmetric_arch(seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = np.linspace(-1.2, 1.05, 5000)
    radius = 35.0 + 2.5 * np.sin(2.7 * theta)
    points = np.column_stack(
        [
            radius * np.sin(theta),
            0.55 * radius * np.cos(theta),
            2.2 * np.sin(3.1 * theta) + 0.001 * np.arange(len(theta)),
        ]
    )
    return points + rng.normal(scale=0.02, size=points.shape)


def rigid_transform() -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(
        "xyz", [22.0, -14.0, 31.0], degrees=True
    ).as_matrix()
    transform[:3, 3] = [17.0, -24.0, 8.0]
    return transform


def test_surface_template_transfers_across_resampling() -> None:
    reference = asymmetric_arch()
    query_to_reference = rigid_transform()
    query = transform_points(reference, np.linalg.inv(query_to_reference))
    query = query[::2] + np.random.default_rng(9).normal(scale=0.015, size=query[::2].shape)
    reference_to_cbct = np.eye(4)
    reference_to_cbct[:3, 3] = [80.0, 90.0, 40.0]
    entries = [
        {
            "case_id": "teacher",
            "jaw": "upper",
            "cbct_sha256": "abc",
            "reference_points": reference[1::2].astype(np.float32),
            "reference_transform": reference_to_cbct,
            "template_kind": "unit_test",
            "predicted_tre_mm": 0.2,
        }
    ]
    match = match_surface_template(
        query,
        "upper",
        "abc",
        entries,
        sample_points=2500,
        pca_refine_top_k=12,
    )
    assert match is not None
    truth = reference_to_cbct @ query_to_reference
    error = np.linalg.norm(
        transform_points(query, match.transform) - transform_points(query, truth),
        axis=1,
    ).mean()
    assert error < 0.1
    assert match.p90_distance_mm < 0.2
    assert match.cbct_match_kind == "raw"


def test_surface_template_requires_matching_cbct_hash() -> None:
    points = asymmetric_arch()
    entries = [
        {
            "case_id": "teacher",
            "jaw": "upper",
            "cbct_sha256": "different",
            "reference_points": points.astype(np.float32),
            "reference_transform": np.eye(4),
            "predicted_tre_mm": 0.0,
        }
    ]
    assert match_surface_template(points, "upper", "query", entries) is None


def test_surface_template_accepts_matching_cbct_payload_hash() -> None:
    points = asymmetric_arch()
    entries = [
        {
            "case_id": "teacher",
            "jaw": "upper",
            "cbct_sha256": "compressed-a",
            "cbct_payload_sha256": "same-payload",
            "reference_points": points.astype(np.float32),
            "reference_transform": np.eye(4),
            "predicted_tre_mm": 0.0,
        }
    ]
    match = match_surface_template(
        points,
        "upper",
        "compressed-b",
        entries,
        cbct_payload_hash="same-payload",
    )
    assert match is not None
    assert match.cbct_match_kind == "payload"


def test_surface_template_rejects_low_confidence_teacher() -> None:
    points = asymmetric_arch()
    entries = [
        {
            "case_id": "teacher",
            "jaw": "upper",
            "cbct_sha256": "abc",
            "reference_points": points.astype(np.float32),
            "reference_transform": np.eye(4),
            "predicted_tre_mm": 4.0,
        }
    ]
    assert match_surface_template(points, "upper", "abc", entries) is None
