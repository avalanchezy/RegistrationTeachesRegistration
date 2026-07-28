from __future__ import annotations

from itertools import combinations
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np


def component_score(
    extent_mm: np.ndarray,
    voxel_count: int,
    center_ijk: np.ndarray | None = None,
    image_shape: np.ndarray | None = None,
) -> float:
    expected = np.array([25.0, 48.0, 72.0])
    shape_cost = np.mean(
        np.abs(np.log(np.sort(np.maximum(extent_mm, 0.5)) / expected))
    )
    score = float(shape_cost - 0.04 * np.log1p(voxel_count))
    if center_ijk is not None and image_shape is not None:
        normalized = center_ijk / np.maximum(image_shape - 1, 1)
        score += float(abs(normalized[0] - 0.5))
        score += float(6.0 * max(normalized[2] - 0.72, 0.0))
        score += float(2.0 * max(0.08 - normalized[2], 0.0))
    return score


def component_pair_compatible(
    first: tuple, second: tuple, spacing: np.ndarray
) -> bool:
    first_center = 0.5 * (first[2] + first[3] - 1)
    second_center = 0.5 * (second[2] + second[3] - 1)
    delta_mm = np.abs(first_center - second_center) * spacing
    union_start = np.minimum(first[2], second[2])
    union_stop = np.maximum(first[3], second[3])
    union_extent_mm = (union_stop - union_start) * spacing
    return bool(
        np.linalg.norm(delta_mm) <= 75.0
        and np.linalg.norm(delta_mm[:2]) <= 55.0
        and delta_mm[2] <= 55.0
        and np.all(union_extent_mm <= np.array([115.0, 105.0, 110.0]))
    )


def cap_physical_bounds(
    start: np.ndarray,
    stop: np.ndarray,
    image_shape: np.ndarray,
    spacing: np.ndarray,
    max_extent_mm: np.ndarray = np.array([105.0, 95.0, 90.0]),
) -> tuple[np.ndarray, np.ndarray]:
    current_size = stop - start
    target_size = np.minimum(
        current_size, np.maximum(1, np.floor(max_extent_mm / spacing).astype(int))
    )
    center = 0.5 * (start + stop)
    capped_start = np.floor(center - 0.5 * target_size).astype(int)
    capped_start = np.minimum(np.maximum(capped_start, 0), image_shape - target_size)
    return capped_start, capped_start + target_size


def anterior_fallback_bounds(
    image_shape: np.ndarray,
    orientation: tuple[str, str, str],
    proposed_start: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if orientation != ("L", "P", "S") or proposed_start[1] <= 0.45 * image_shape[1]:
        return None
    start = np.floor(np.array([0.08, 0.00, 0.04]) * image_shape).astype(int)
    stop = np.ceil(np.array([0.92, 0.60, 0.72]) * image_shape).astype(int)
    return start, stop


def crop_dental_roi(
    source: Path,
    output: Path,
    threshold: float,
    margin_mm: float,
    max_crop_volume_cm3: float,
) -> dict[str, object]:
    image = nib.load(str(source))
    volume = np.asanyarray(image.dataobj)
    affine = np.asarray(image.affine, dtype=np.float64)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    voxel_volume = abs(float(np.linalg.det(affine[:3, :3])))
    threshold_trials = []
    for value in (
        threshold,
        threshold - 200,
        threshold - 400,
        threshold + 200,
        threshold - 600,
    ):
        if value not in threshold_trials and value > 0:
            threshold_trials.append(value)
    solutions = []
    single_component_solutions = []
    primary_threshold_singles = []
    for used_threshold in threshold_trials:
        labels = cc3d.connected_components(volume >= used_threshold, connectivity=26)
        stats = cc3d.statistics(labels)
        ranked = []
        for component_id in range(1, len(stats["voxel_counts"])):
            count = int(stats["voxel_counts"][component_id])
            if count * voxel_volume < 80.0:
                continue
            slices = stats["bounding_boxes"][component_id]
            starts = np.array([item.start for item in slices])
            stops = np.array([item.stop for item in slices])
            extent = (stops - starts - 1) * spacing
            if extent.max() < 25.0:
                continue
            center = 0.5 * (starts + stops - 1)
            ranked.append(
                (
                    component_score(extent, count, center, np.asarray(volume.shape)),
                    component_id,
                    starts,
                    stops,
                    extent,
                    count,
                )
            )
        ranked = sorted(ranked)
        compatible_pairs = [
            pair
            for pair in combinations(ranked[: min(8, len(ranked))], 2)
            if component_pair_compatible(pair[0], pair[1], spacing)
        ]
        if compatible_pairs:
            pair = min(compatible_pairs, key=lambda items: items[0][0] + items[1][0])
            penalty = abs(used_threshold - threshold) / 2000.0
            solutions.append((sum(item[0] for item in pair) + penalty, used_threshold, pair))
        elif ranked:
            best = min(ranked, key=lambda item: item[0])
            penalty = abs(used_threshold - threshold) / 2000.0
            single_component_solutions.append((best[0] + penalty, used_threshold, [best]))
        if used_threshold == threshold:
            for candidate in ranked:
                center = 0.5 * (candidate[2] + candidate[3] - 1)
                normalized = center / np.maximum(np.asarray(volume.shape) - 1, 1)
                if 0.05 <= normalized[2] <= 0.60:
                    primary_threshold_singles.append(
                        (candidate[0], used_threshold, [candidate])
                    )
    if not solutions:
        solutions = single_component_solutions
    if not solutions:
        raise RuntimeError(f"Could not find a dental-arch component in {source}")
    _, used_threshold, selected = min(solutions, key=lambda item: item[0])
    start = np.min([item[2] for item in selected], axis=0)
    stop = np.max([item[3] for item in selected], axis=0)
    margin_voxels = np.ceil(margin_mm / spacing).astype(int)
    start = np.maximum(start - margin_voxels, 0)
    stop = np.minimum(stop + margin_voxels, np.array(volume.shape))
    crop_volume_cm3 = float(np.prod((stop - start) * spacing) / 1000.0)
    used_volume_fallback = False
    used_anterior_fallback = False
    if crop_volume_cm3 > max_crop_volume_cm3 and primary_threshold_singles:
        _, used_threshold, selected = min(
            primary_threshold_singles, key=lambda item: item[0]
        )
        start = np.maximum(selected[0][2] - margin_voxels, 0)
        stop = np.minimum(selected[0][3] + margin_voxels, np.array(volume.shape))
        crop_volume_cm3 = float(np.prod((stop - start) * spacing) / 1000.0)
        used_volume_fallback = True
    fallback_bounds = anterior_fallback_bounds(
        np.asarray(volume.shape), nib.aff2axcodes(affine), start
    )
    if fallback_bounds is not None:
        start, stop = fallback_bounds
        crop_volume_cm3 = float(np.prod((stop - start) * spacing) / 1000.0)
        used_anterior_fallback = True
    used_hard_cap = False
    if crop_volume_cm3 > max_crop_volume_cm3:
        start, stop = cap_physical_bounds(
            start, stop, np.asarray(volume.shape), spacing
        )
        crop_volume_cm3 = float(np.prod((stop - start) * spacing) / 1000.0)
        used_hard_cap = True
    crop = volume[tuple(slice(int(lo), int(hi)) for lo, hi in zip(start, stop))]
    offset = np.eye(4)
    offset[:3, 3] = start
    crop_affine = affine @ offset
    output.parent.mkdir(parents=True, exist_ok=True)
    header = image.header.copy()
    header.set_data_shape(crop.shape)
    nib.save(nib.Nifti1Image(crop, crop_affine, header), str(output))
    return {
        "source": str(source),
        "output": str(output),
        "source_shape": "x".join(map(str, volume.shape)),
        "crop_shape": "x".join(map(str, crop.shape)),
        "start_ijk": " ".join(map(str, start.tolist())),
        "stop_ijk": " ".join(map(str, stop.tolist())),
        "threshold": used_threshold,
        "margin_mm": margin_mm,
        "crop_volume_cm3": crop_volume_cm3,
        "volume_fallback": int(used_volume_fallback),
        "anterior_fallback": int(used_anterior_fallback),
        "hard_cap": int(used_hard_cap),
        "component_ids": " ".join(str(item[1]) for item in selected),
        "component_scores": " ".join(f"{item[0]:.5f}" for item in selected),
    }
