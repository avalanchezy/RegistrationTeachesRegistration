import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from scripts.augment_candidate_geometry import (
    adaptive_target_points,
    attach_toothseg_volume_path,
    build_target_cache,
    index_toothseg_volumes,
    resolve_adaptive_volume_paths,
)


def test_adaptive_candidate_inherits_coarse_target_volume() -> None:
    candidates = [
        {
            "target": "axial_upper_threshold_1600_tracked_7",
            "target_metadata": {
                "mode": "threshold",
                "volume_path": "roi/case.nii.gz",
            },
        },
        {
            "target": "adaptive_roi_1000_from_threshold_1600_tracked_7",
            "target_metadata": {
                "mode": "adaptive_threshold_roi",
                "coarse_target": "threshold_1600_tracked_7",
            },
        },
    ]
    resolve_adaptive_volume_paths(candidates, Path("fallback.nii.gz"))
    assert candidates[1]["target_metadata"]["volume_path"] == str(
        Path("roi/case.nii.gz")
    )


def test_adaptive_target_reconstructs_local_threshold_surface() -> None:
    volume = np.zeros((32, 32, 32), dtype=np.int16)
    volume[8:24, 8:24, 8:24] = 1200
    face_y, face_z = np.meshgrid(np.arange(8, 24), np.arange(8, 24), indexing="ij")
    source = np.column_stack(
        (np.full(face_y.size, 8.0), face_y.reshape(-1), face_z.reshape(-1))
    )
    metadata = {
        "threshold": 1000,
        "roi_start_ijk": [4, 4, 4],
        "roi_stop_ijk": [28, 28, 28],
        "adaptive_radius_mm": 1.5,
    }
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "roi.nii.gz"
        nib.save(nib.Nifti1Image(volume, np.eye(4)), str(path))
        points = adaptive_target_points(path, metadata, source, np.eye(4), seed=7)
    assert points is not None
    assert len(points) >= 128
    assert np.max(np.abs(points[:, 0] - 8.0)) <= 1.0


def test_toothseg_target_cache_uses_matching_label_volume() -> None:
    labels = np.zeros((32, 32, 32), dtype=np.uint8)
    labels[8:24, 9:23, 10:22] = 1
    candidate = {
        "target": "toothseg_full_teeth",
        "target_metadata": {
            "mode": "toothseg",
            "crown_fraction": 1.0,
            "detected_teeth": 1,
        },
    }
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        path = directory / "STS2_003.nii.gz"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(path))
        indexed = index_toothseg_volumes([directory])
        attach_toothseg_volume_path([candidate], "003", indexed)
        cache = build_target_cache(
            [candidate],
            Path("fallback.nii.gz"),
            "upper",
            np.ones(3),
            np.zeros((128, 3), dtype=np.float64),
            seed=7,
        )

    key = (str(path.resolve()), "upper", "toothseg_full_teeth")
    assert candidate["target_metadata"]["volume_path"] == str(path.resolve())
    assert key in cache
    points, normals = cache[key]
    assert len(points) >= 128
    assert points.shape == normals.shape
