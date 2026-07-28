from __future__ import annotations

from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import distance_transform_edt, map_coordinates

from .data import apply_transform, ios_pca_side_variants


def fixed_world_grid(
    image: nib.spatialimages.SpatialImage,
    shape: tuple[int, int, int],
    spacing_mm: float,
) -> np.ndarray:
    """Return an axis-aligned RAS grid centered on a physical-space image box."""
    source_shape = np.asarray(image.shape[:3], dtype=np.float64)
    corners = np.asarray(
        list(product(*[(0.0, max(size - 1.0, 0.0)) for size in source_shape])),
        dtype=np.float64,
    )
    homogeneous = np.column_stack((corners, np.ones(len(corners))))
    world = (np.asarray(image.affine, dtype=np.float64) @ homogeneous.T).T[:, :3]
    center = 0.5 * (world.min(axis=0) + world.max(axis=0))

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.eye(3) * float(spacing_mm)
    affine[:3, 3] = center - 0.5 * (np.asarray(shape, dtype=np.float64) - 1.0) * spacing_mm
    return affine


def resample_hu(
    image: nib.spatialimages.SpatialImage,
    shape: tuple[int, int, int],
    affine: np.ndarray,
) -> np.ndarray:
    resampled = resample_from_to(
        image,
        (shape, affine),
        order=1,
        mode="constant",
        cval=-1000.0,
    )
    return np.asarray(resampled.dataobj, dtype=np.float32)


def sample_hu_at_world(
    image: nib.spatialimages.SpatialImage,
    points_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_world = np.asarray(points_world, dtype=np.float64)
    homogeneous = np.column_stack((points_world, np.ones(len(points_world))))
    ijk = (np.linalg.inv(np.asarray(image.affine)) @ homogeneous.T).T[:, :3]
    shape = np.asarray(image.shape[:3], dtype=np.float64)
    inside = np.all((ijk >= 0.0) & (ijk <= shape - 1.0), axis=1)
    values = np.full(len(points_world), -1000.0, dtype=np.float32)
    if np.any(inside):
        volume = np.asanyarray(image.dataobj)
        values[inside] = map_coordinates(
            volume,
            ijk[inside].T,
            order=1,
            mode="constant",
            cval=-1000.0,
            prefilter=False,
        ).astype(np.float32)
    return values, inside


def select_aligned_crown_points(
    ios_points: np.ndarray,
    transform: np.ndarray,
    cbct_image: nib.spatialimages.SpatialImage,
    fraction: float = 0.35,
) -> tuple[np.ndarray, dict]:
    """Choose the IOS PCA side most consistent with high-density CBCT crowns."""
    variants = ios_pca_side_variants(
        np.asarray(ios_points, dtype=np.float64),
        fractions=(float(fraction),),
        include_full=False,
    )
    scored: list[tuple[float, str, np.ndarray, dict]] = []
    for name, points in variants:
        moved = apply_transform(points, transform)
        values, inside = sample_hu_at_world(cbct_image, moved)
        visible = values[inside]
        if len(visible) == 0:
            score = -np.inf
            statistics = {
                "inside_fraction": 0.0,
                "mean_clipped_hu": -1000.0,
                "fraction_over_700hu": 0.0,
                "fraction_over_1400hu": 0.0,
            }
        else:
            clipped = np.clip(visible, -500.0, 3000.0)
            over_700 = float(np.mean(visible >= 700.0))
            over_1400 = float(np.mean(visible >= 1400.0))
            mean_hu = float(np.mean(clipped))
            score = mean_hu + 850.0 * over_700 + 350.0 * over_1400
            statistics = {
                "inside_fraction": float(np.mean(inside)),
                "mean_clipped_hu": mean_hu,
                "fraction_over_700hu": over_700,
                "fraction_over_1400hu": over_1400,
            }
        statistics["selection_score"] = float(score)
        scored.append((score, name, moved, statistics))
    if not scored or not np.isfinite(max(item[0] for item in scored)):
        raise ValueError("No transformed IOS crown-side points fall inside the CBCT")
    score, name, moved, statistics = max(scored, key=lambda item: item[0])
    statistics = dict(statistics)
    statistics.update(
        {
            "selected_variant": name,
            "selected_points": int(len(moved)),
            "alternative_scores": {item[1]: float(item[0]) for item in scored},
        }
    )
    return moved, statistics


def world_to_grid(points_world: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float64)
    homogeneous = np.column_stack((points_world, np.ones(len(points_world))))
    return (np.linalg.inv(np.asarray(affine, dtype=np.float64)) @ homogeneous.T).T[:, :3]


def crown_distance_voxels(
    points_world: np.ndarray,
    shape: tuple[int, int, int],
    affine: np.ndarray,
) -> tuple[np.ndarray, float]:
    ijk = world_to_grid(points_world, affine)
    shape_array = np.asarray(shape, dtype=np.int64)
    inside = np.all((ijk >= 0.0) & (ijk <= shape_array - 1.0), axis=1)
    seed = np.zeros(shape, dtype=bool)
    if np.any(inside):
        indices = np.rint(ijk[inside]).astype(np.int64)
        indices = np.minimum(np.maximum(indices, 0), shape_array - 1)
        seed[tuple(indices.T)] = True
    if not np.any(seed):
        raise ValueError("Aligned crown has no samples in the fixed localizer grid")
    return distance_transform_edt(~seed).astype(np.float32), float(np.mean(inside))


def build_crown_labels(
    image_hu: np.ndarray,
    crown_points: dict[str, np.ndarray],
    affine: np.ndarray,
    spacing_mm: float,
    radius_mm: float = 2.5,
    minimum_hu: float = 150.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Rasterize aligned IOS crown surfaces into upper/lower CBCT masks."""
    distances: dict[str, np.ndarray] = {}
    coverage: dict[str, float] = {}
    for jaw in ("upper", "lower"):
        distances[jaw], coverage[jaw] = crown_distance_voxels(
            crown_points[jaw], image_hu.shape, affine
        )
    radius_voxels = float(radius_mm) / float(spacing_mm)
    upper = (distances["upper"] <= radius_voxels) & (image_hu >= minimum_hu)
    lower = (distances["lower"] <= radius_voxels) & (image_hu >= minimum_hu)
    overlap = upper & lower
    if np.any(overlap):
        upper_wins = distances["upper"] <= distances["lower"]
        upper[overlap] = upper_wins[overlap]
        lower[overlap] = ~upper_wins[overlap]
    labels = np.zeros(image_hu.shape, dtype=np.uint8)
    labels[upper] = 1
    labels[lower] = 2
    if not np.any(labels == 1) or not np.any(labels == 2):
        raise ValueError("Pseudo crown rasterization produced an empty jaw mask")
    return labels, coverage


def labels_to_registration_ids(labels: np.ndarray) -> np.ndarray:
    output = np.zeros(np.asarray(labels).shape, dtype=np.uint8)
    output[np.asarray(labels) == 1] = 1
    output[np.asarray(labels) == 2] = 17
    return output


def save_registration_labels(
    labels: np.ndarray,
    affine: np.ndarray,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(labels_to_registration_ids(labels), affine)
    image.set_data_dtype(np.uint8)
    nib.save(image, str(output))
