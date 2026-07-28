from types import SimpleNamespace

import numpy as np

from scripts.transfer_template_candidate_runs import (
    source_key_for_label,
    source_teacher_transform,
)


def translated(x: float) -> np.ndarray:
    transform = np.eye(4)
    transform[0, 3] = x
    return transform


def test_geometry_source_uses_pseudo_teacher_transform() -> None:
    query = {
        "case_id": "102",
        "jaw": "upper",
        "source_labeled_case_id": "",
        "source_teacher_case_id": "101",
        "transform": translated(7.0).tolist(),
    }
    source = {
        "case_id": "101",
        "jaw": "upper",
        "transform": translated(2.0).tolist(),
    }
    source_key = source_key_for_label(query)
    source_transform = source_teacher_transform(
        source_key,
        ("102", "upper"),
        translated(7.0),
        {("101", "upper"): source},
        {},
    )
    query_to_source = np.linalg.inv(source_transform) @ translated(7.0)
    assert source_key == ("101", "upper")
    assert np.allclose(query_to_source[:3, 3], [5.0, 0.0, 0.0])


def test_labeled_source_falls_back_to_ground_truth(tmp_path) -> None:
    ground_truth_path = tmp_path / "upper_gt.npy"
    np.save(ground_truth_path, translated(3.0))
    records = {
        ("001", "upper"): SimpleNamespace(transform_path=str(ground_truth_path))
    }
    result = source_teacher_transform(
        ("001", "upper"),
        ("102", "upper"),
        translated(7.0),
        {},
        records,
    )
    assert np.allclose(result, translated(3.0))
