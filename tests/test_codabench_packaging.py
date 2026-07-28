from pathlib import Path

import numpy as np
import pytest

from scripts.package_codabench_submission import validate_transform


def _write_transform(path: Path, linear: np.ndarray) -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear
    np.save(path, transform)


def test_packaging_accepts_dataset_chirality_reflection(tmp_path) -> None:
    path = tmp_path / "lower_gt.npy"
    _write_transform(path, np.diag([-1.0, 1.0, 1.0]))

    metrics = validate_transform(path)

    assert metrics["determinant"] == pytest.approx(-1.0)
    assert metrics["has_reflection"] is True


def test_packaging_rejects_nonrigid_scaling(tmp_path) -> None:
    path = tmp_path / "upper_gt.npy"
    _write_transform(path, np.diag([1.0, 1.0, 1.1]))

    with pytest.raises(ValueError, match="orthogonality"):
        validate_transform(path)
