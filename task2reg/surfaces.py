from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, find_objects
from scipy.spatial import cKDTree


@dataclass
class SurfaceCandidate:
    name: str
    points: np.ndarray
    metadata: dict[str, float | int | str | list[float]]


def _to_world(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return indices @ affine[:3, :3].T + affine[:3, 3]


def _sample(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= count:
        return points.astype(np.float64, copy=False)
    return points[rng.choice(len(points), size=count, replace=False)].astype(np.float64, copy=False)


def _physical_bbox(slices: tuple[slice, slice, slice], affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.array([item.start for item in slices], dtype=np.float64)
    stops = np.array([item.stop - 1 for item in slices], dtype=np.float64)
    corners = np.array(
        [[x, y, z] for x in (starts[0], stops[0]) for y in (starts[1], stops[1]) for z in (starts[2], stops[2])]
    )
    world = _to_world(corners, affine)
    return world.min(axis=0), world.max(axis=0)


def threshold_surface_candidates(
    volume_path: Path,
    jaw: str,
    source_extent: np.ndarray,
    thresholds: tuple[float, ...],
    points_per_candidate: int = 12000,
    components_per_threshold: int = 3,
    min_component_mm3: float = 80.0,
    seed: int = 2026,
) -> list[SurfaceCandidate]:
    image = nib.load(str(volume_path))
    volume = np.asanyarray(image.dataobj)
    affine = np.asarray(image.affine, dtype=np.float64)
    voxel_volume = abs(float(np.linalg.det(affine[:3, :3])))
    source_sorted = np.sort(np.maximum(np.asarray(source_extent, dtype=np.float64), 1.0))
    rng = np.random.default_rng(seed)
    candidates: list[SurfaceCandidate] = []

    # At 1600 HU the two dental arches are consistently separated even in
    # large-FOV scans. Select the two arch-shaped components once, then track
    # the requested jaw through the threshold sweep. This prevents a lower HU
    # candidate from silently jumping to a similarly sized skull component.
    anchor_solutions = []
    for anchor_threshold_trial in (1600.0, 1400.0, 1200.0, 1800.0, 1000.0):
        trial_mask = volume >= anchor_threshold_trial
        trial_labels = cc3d.connected_components(trial_mask, connectivity=26)
        trial_stats = cc3d.statistics(trial_labels)
        trial_ranked: list[tuple[float, int, np.ndarray, np.ndarray, float]] = []
        for component_id in range(1, len(trial_stats["voxel_counts"])):
            count = int(trial_stats["voxel_counts"][component_id])
            if count * voxel_volume < min_component_mm3:
                continue
            bbox_min, bbox_max = _physical_bbox(trial_stats["bounding_boxes"][component_id], affine)
            extent = np.maximum(bbox_max - bbox_min, 0.5)
            if extent.max() < 0.4 * source_sorted.max():
                continue
            extent_cost = float(np.mean(np.abs(np.log(np.sort(extent) / source_sorted))))
            score = extent_cost - 0.03 * np.log1p(count)
            trial_ranked.append(
                (score, component_id, bbox_min, bbox_max, float((bbox_min[2] + bbox_max[2]) * 0.5))
            )
        if len(trial_ranked) >= 2:
            pair = sorted(trial_ranked)[:2]
            threshold_penalty = abs(anchor_threshold_trial - 1600.0) / 2000.0
            anchor_solutions.append(
                (
                    sum(item[0] for item in pair) + threshold_penalty,
                    anchor_threshold_trial,
                    pair,
                )
            )
            del trial_labels, trial_mask
            break
        del trial_labels, trial_mask
    if not anchor_solutions:
        raise RuntimeError(f"Could not identify both dental arches in {volume_path}")
    _, anchor_threshold, arch_pair = min(
        anchor_solutions, key=lambda item: item[0]
    )
    anchor_mask = volume >= anchor_threshold
    anchor_labels = cc3d.connected_components(anchor_mask, connectivity=26)
    anchor_stats = cc3d.statistics(anchor_labels)
    selected_anchor = max(arch_pair, key=lambda item: item[4]) if jaw == "upper" else min(arch_pair, key=lambda item: item[4])
    anchor_score, anchor_id, anchor_bbox_min, anchor_bbox_max, anchor_z = selected_anchor
    anchor_component = anchor_labels == anchor_id

    for threshold in thresholds:
        mask = volume >= threshold
        if threshold <= anchor_threshold:
            labels = cc3d.connected_components(mask, connectivity=26)
            overlap_labels, overlap_counts = np.unique(labels[anchor_component], return_counts=True)
            valid = overlap_labels > 0
            if not np.any(valid):
                continue
            tracked_id = int(overlap_labels[valid][np.argmax(overlap_counts[valid])])
            component = labels == tracked_id
        else:
            tracked_id = anchor_id
            component = mask & anchor_component
        boundary = component & ~binary_erosion(component, structure=np.ones((3, 3, 3), dtype=bool))
        indices = np.argwhere(boundary)
        if len(indices) < 64:
            continue
        points_world = _to_world(indices, affine)
        bbox_min = points_world.min(axis=0)
        bbox_max = points_world.max(axis=0)
        threshold_penalty = abs(float(threshold) - anchor_threshold) / 4000.0
        candidates.append(
            SurfaceCandidate(
                name=f"threshold_{threshold:g}_tracked_{tracked_id}",
                points=_sample(points_world, points_per_candidate, rng),
                metadata={
                    "mode": "threshold",
                    "threshold": float(threshold),
                    "component_id": int(tracked_id),
                    "anchor_component_id": int(anchor_id),
                    "anchor_threshold": float(anchor_threshold),
                    "component_voxels": int(component.sum()),
                    "rank_score": float(anchor_score + threshold_penalty),
                    "anchor_z_mm": float(anchor_z),
                    "anchor_bbox_min": anchor_bbox_min.tolist(),
                    "anchor_bbox_max": anchor_bbox_max.tolist(),
                    "bbox_min": bbox_min.tolist(),
                    "bbox_max": bbox_max.tolist(),
                },
            )
        )
    return candidates


def threshold_roi_surface_candidates(
    volume_path: Path,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    thresholds: tuple[float, ...] = (800.0, 1000.0, 1200.0),
    margin_mm: float = 10.0,
    points_per_candidate: int = 30000,
    seed: int = 2026,
) -> list[SurfaceCandidate]:
    """Extract broad low-HU surfaces only inside a known dental-arch ROI."""
    image = nib.load(str(volume_path))
    volume = np.asanyarray(image.dataobj)
    affine = np.asarray(image.affine, dtype=np.float64)
    inverse = np.linalg.inv(affine)
    world_min = np.asarray(bbox_min, dtype=np.float64) - margin_mm
    world_max = np.asarray(bbox_max, dtype=np.float64) + margin_mm
    corners = np.array(
        [[x, y, z] for x in (world_min[0], world_max[0])
         for y in (world_min[1], world_max[1])
         for z in (world_min[2], world_max[2])],
        dtype=np.float64,
    )
    voxel_corners = _to_world(corners, inverse)
    start = np.maximum(np.floor(voxel_corners.min(axis=0)).astype(int), 0)
    stop = np.minimum(np.ceil(voxel_corners.max(axis=0)).astype(int) + 1, np.asarray(volume.shape))
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(start, stop))
    crop = volume[slices]
    rng = np.random.default_rng(seed)
    candidates = []
    for threshold in thresholds:
        mask = crop >= threshold
        boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool))
        indices = np.argwhere(boundary) + start
        if len(indices) < 128:
            continue
        points = _sample(_to_world(indices, affine), points_per_candidate, rng)
        candidates.append(
            SurfaceCandidate(
                name=f"adaptive_roi_{threshold:g}",
                points=points,
                metadata={
                    "mode": "adaptive_threshold_roi",
                    "threshold": float(threshold),
                    "margin_mm": float(margin_mm),
                    "roi_start_ijk": start.tolist(),
                    "roi_stop_ijk": stop.tolist(),
                    "surface_points_before_sampling": int(len(indices)),
                },
            )
        )
    return candidates


def threshold_aggregate_surface_candidates(
    volume_path: Path,
    source_extent: np.ndarray,
    thresholds: tuple[float, ...] = (1600.0, 1800.0, 2000.0, 2400.0),
    aggregate_counts: tuple[int, ...] = (2, 4),
    points_per_candidate: int = 16000,
    min_component_mm3: float = 5.0,
    seed: int = 2026,
) -> list[SurfaceCandidate]:
    """Aggregate disconnected high-HU tooth/enamel components.

    Some scans contain one connected mandibular cortex but split the actual
    dentition into left/right or per-tooth components. Keeping unions of the
    best component groups avoids forcing a premature upper/lower assignment.
    """
    image = nib.load(str(volume_path))
    volume = np.asanyarray(image.dataobj)
    affine = np.asarray(image.affine, dtype=np.float64)
    voxel_volume = abs(float(np.linalg.det(affine[:3, :3])))
    source_sorted = np.sort(np.maximum(np.asarray(source_extent, dtype=np.float64), 1.0))
    rng = np.random.default_rng(seed)
    candidates = []
    for threshold in thresholds:
        labels = cc3d.connected_components(volume >= threshold, connectivity=26)
        stats = cc3d.statistics(labels)
        ranked = []
        for component_id in range(1, len(stats["voxel_counts"])):
            count = int(stats["voxel_counts"][component_id])
            if count * voxel_volume < min_component_mm3:
                continue
            bbox_min, bbox_max = _physical_bbox(stats["bounding_boxes"][component_id], affine)
            extent = np.maximum(bbox_max - bbox_min, 0.5)
            extent_cost = float(np.mean(np.abs(np.log(np.sort(extent) / source_sorted))))
            score = extent_cost - 0.03 * np.log1p(count)
            ranked.append((score, component_id, count, bbox_min, bbox_max))
        ranked.sort(key=lambda item: item[0])
        emitted_counts: set[int] = set()
        for aggregate_count in aggregate_counts:
            selected = ranked[: min(aggregate_count, len(ranked))]
            if len(selected) < 2:
                continue
            if len(selected) in emitted_counts:
                continue
            emitted_counts.add(len(selected))
            selected_ids = np.asarray([item[1] for item in selected], dtype=labels.dtype)
            component = _chunked_membership_mask(labels, selected_ids)
            eroded = np.empty_like(component)
            binary_erosion(
                component,
                structure=np.ones((3, 3, 3), dtype=bool),
                output=eroded,
            )
            np.logical_not(eroded, out=eroded)
            np.logical_and(component, eroded, out=component)
            del eroded
            indices = np.argwhere(component)
            if len(indices) < 128:
                continue
            points = _to_world(indices, affine)
            combined_extent = np.maximum(points.max(axis=0) - points.min(axis=0), 0.5)
            combined_cost = float(np.mean(np.abs(np.log(np.sort(combined_extent) / source_sorted))))
            candidates.append(
                SurfaceCandidate(
                    name=f"threshold_{threshold:g}_aggregate_{len(selected)}",
                    points=_sample(points, points_per_candidate, rng),
                    metadata={
                        "mode": "threshold_aggregate",
                        "threshold": float(threshold),
                        "component_ids": selected_ids.astype(int).tolist(),
                        "component_voxels": int(sum(item[2] for item in selected)),
                        "aggregate_count": int(len(selected)),
                        "rank_score": combined_cost + abs(float(threshold) - 2000.0) / 4000.0,
                        "bbox_min": points.min(axis=0).tolist(),
                        "bbox_max": points.max(axis=0).tolist(),
                    },
                )
            )
        del labels
    return candidates


def _chunked_membership_mask(
    labels: np.ndarray,
    selected_ids: np.ndarray,
    max_scratch_bytes: int = 32 * 1024 * 1024,
) -> np.ndarray:
    """Build an ID-union mask without ``np.isin``'s volume-sized temporaries."""
    selected_ids = np.asarray(selected_ids, dtype=labels.dtype).reshape(-1)
    if len(selected_ids) == 0:
        return np.zeros(labels.shape, dtype=bool)
    mask = np.empty(labels.shape, dtype=bool)
    slice_voxels = int(np.prod(labels.shape[1:])) if labels.ndim > 1 else 1
    chunk_depth = max(1, max_scratch_bytes // max(slice_voxels, 1))
    for start in range(0, labels.shape[0], chunk_depth):
        stop = min(start + chunk_depth, labels.shape[0])
        source = labels[start:stop]
        target = mask[start:stop]
        np.equal(source, selected_ids[0], out=target)
        if len(selected_ids) > 1:
            scratch = np.empty_like(target)
            for component_id in selected_ids[1:]:
                np.equal(source, component_id, out=scratch)
                np.logical_or(target, scratch, out=target)
    return mask


def toothseg_surface_candidates(
    segmentation_path: Path,
    jaw: str,
    crown_fractions: tuple[float, ...] = (0.35, 0.45, 0.55),
    points_per_candidate: int = 16000,
    seed: int = 2026,
) -> list[SurfaceCandidate]:
    image = nib.load(str(segmentation_path))
    labels = np.asanyarray(image.dataobj)
    affine = np.asarray(image.affine, dtype=np.float64)
    class_ids = range(1, 17) if jaw == "upper" else range(17, 33)
    rng = np.random.default_rng(seed)
    per_tooth: list[np.ndarray] = []
    object_slices = find_objects(labels, max_label=32)

    for class_id in class_ids:
        tooth_slice = object_slices[class_id - 1]
        if tooth_slice is None:
            continue
        tooth = np.asarray(labels[tooth_slice]) == class_id
        boundary = tooth & ~binary_erosion(tooth, structure=np.ones((3, 3, 3), dtype=bool))
        indices = np.argwhere(boundary)
        if len(indices) < 32:
            continue
        indices += np.asarray([axis.start for axis in tooth_slice], dtype=indices.dtype)
        per_tooth.append(_to_world(indices, affine))

    if not per_tooth:
        raise ValueError(f"No {jaw} ToothSeg labels found in {segmentation_path}")

    full = np.concatenate(per_tooth, axis=0)
    candidates = [
        SurfaceCandidate(
            name="toothseg_full_teeth",
            points=_sample(full, points_per_candidate, rng),
            metadata={"mode": "toothseg", "crown_fraction": 1.0, "detected_teeth": len(per_tooth)},
        )
    ]
    for fraction in crown_fractions:
        crown_parts = []
        for points in per_tooth:
            z = points[:, 2]
            quantile = fraction if jaw == "upper" else 1.0 - fraction
            cut = np.quantile(z, quantile)
            crown_parts.append(points[z <= cut] if jaw == "upper" else points[z >= cut])
        crown = np.concatenate(crown_parts, axis=0)
        candidates.append(
            SurfaceCandidate(
                name=f"toothseg_crown_{fraction:.2f}",
                points=_sample(crown, points_per_candidate, rng),
                metadata={"mode": "toothseg", "crown_fraction": float(fraction), "detected_teeth": len(per_tooth)},
            )
        )
    return candidates


def crown_mask_surface_candidates(
    segmentation_path: Path,
    jaw: str,
    points_per_candidate: int = 16000,
    seed: int = 2026,
) -> list[SurfaceCandidate]:
    """Use a crown-localizer support region directly, without an offset boundary."""
    image = nib.load(str(segmentation_path))
    labels = np.asanyarray(image.dataobj)
    affine = np.asarray(image.affine, dtype=np.float64)
    class_id = 1 if jaw == "upper" else 17
    indices = np.argwhere(labels == class_id)
    if len(indices) < 32:
        raise ValueError(f"No usable {jaw} crown mask found in {segmentation_path}")
    points = _to_world(indices, affine)
    rng = np.random.default_rng(seed)
    components = cc3d.connected_components(labels == class_id, connectivity=26)
    component_sizes = np.bincount(components.reshape(-1))[1:]
    detected_components = int(np.sum(component_sizes >= 12))
    return [
        SurfaceCandidate(
            name="crown_mask_support",
            points=_sample(points, points_per_candidate, rng),
            metadata={
                "mode": "crown_localizer",
                "detected_components": detected_components,
                "support_voxels": int(len(indices)),
            },
        )
    ]


def crown_probability_surface_candidates(
    probability_path: Path,
    jaw: str,
    probability_thresholds: tuple[float, ...] = (0.25, 0.35, 0.50, 0.70),
    voxel_counts: tuple[int, ...] = (1500, 2500, 4000),
    minimum_component_voxels: int = 12,
    points_per_candidate: int = 16000,
    seed: int = 2026,
) -> list[SurfaceCandidate]:
    """Create several confidence-level crown supports from network probabilities."""
    with np.load(probability_path, allow_pickle=False) as payload:
        probabilities = payload["probabilities"].astype(np.float32)
        affine = payload["affine"].astype(np.float64)
    if probabilities.shape[0] != 3:
        raise ValueError(f"Expected 3 probability channels in {probability_path}")

    class_id = 1 if jaw == "upper" else 2
    class_probability = probabilities[class_id]
    winner = np.argmax(probabilities, axis=0) == class_id
    rng = np.random.default_rng(seed)
    candidates: list[SurfaceCandidate] = []

    def emit(name: str, mask: np.ndarray, metadata: dict[str, float | int | str]) -> None:
        components = cc3d.connected_components(mask, connectivity=26)
        sizes = np.bincount(components.reshape(-1))
        keep = np.flatnonzero(sizes >= minimum_component_voxels)
        keep = keep[keep != 0]
        filtered = np.isin(components, keep)
        indices = np.argwhere(filtered)
        if len(indices) < 32:
            return
        points = _to_world(indices, affine)
        expected_voxels = 2500.0
        rank_score = abs(float(np.log(max(len(indices), 1) / expected_voxels)))
        candidates.append(
            SurfaceCandidate(
                name=name,
                points=_sample(points, points_per_candidate, rng),
                metadata={
                    "mode": "crown_probability",
                    "support_voxels": int(len(indices)),
                    "detected_components": int(len(keep)),
                    "rank_score": rank_score,
                    **metadata,
                },
            )
        )

    for threshold in probability_thresholds:
        emit(
            f"crown_probability_p{threshold:.2f}",
            winner & (class_probability >= threshold),
            {"minimum_probability": float(threshold), "selection": "threshold"},
        )

    winner_indices = np.argwhere(winner)
    if len(winner_indices):
        winner_scores = class_probability[tuple(winner_indices.T)]
        order = np.argsort(winner_scores)[::-1]
        for requested_count in voxel_counts:
            selected_count = min(int(requested_count), len(order))
            mask = np.zeros(winner.shape, dtype=bool)
            selected = winner_indices[order[:selected_count]]
            mask[tuple(selected.T)] = True
            emit(
                f"crown_probability_top{requested_count}",
                mask,
                {
                    "requested_voxels": int(requested_count),
                    "selection": "top_voxels",
                },
            )
    if not candidates:
        raise ValueError(f"No usable {jaw} crown probabilities found in {probability_path}")
    return candidates


def crown_guided_cbct_surface_candidates(
    volume_path: Path,
    segmentation_path: Path,
    jaw: str,
    thresholds: tuple[float, ...] = (500.0, 800.0, 1100.0, 1400.0, 1700.0),
    guidance_radii_mm: tuple[float, ...] = (2.5, 4.0, 6.0),
    points_per_candidate: int = 16000,
    seed: int = 2026,
) -> list[SurfaceCandidate]:
    """Extract native-resolution CBCT isosurfaces near a localized crown support."""
    segmentation = nib.load(str(segmentation_path))
    labels = np.asanyarray(segmentation.dataobj)
    segmentation_affine = np.asarray(segmentation.affine, dtype=np.float64)
    class_id = 1 if jaw == "upper" else 17
    crown_indices = np.argwhere(labels == class_id)
    if len(crown_indices) < 32:
        raise ValueError(f"No usable {jaw} crown guidance found in {segmentation_path}")
    crown_points = _to_world(crown_indices, segmentation_affine)
    crown_tree = cKDTree(crown_points)

    image = nib.load(str(volume_path))
    volume = np.asanyarray(image.dataobj)
    affine = np.asarray(image.affine, dtype=np.float64)
    inverse = np.linalg.inv(affine)
    maximum_radius = max(guidance_radii_mm)
    world_min = crown_points.min(axis=0) - maximum_radius - 2.0
    world_max = crown_points.max(axis=0) + maximum_radius + 2.0
    corners = np.array(
        [
            [x, y, z]
            for x in (world_min[0], world_max[0])
            for y in (world_min[1], world_max[1])
            for z in (world_min[2], world_max[2])
        ],
        dtype=np.float64,
    )
    voxel_corners = _to_world(corners, inverse)
    start = np.maximum(np.floor(voxel_corners.min(axis=0)).astype(int), 0)
    stop = np.minimum(
        np.ceil(voxel_corners.max(axis=0)).astype(int) + 1,
        np.asarray(volume.shape),
    )
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(start, stop))
    crop = volume[slices]
    rng = np.random.default_rng(seed)
    candidates: list[SurfaceCandidate] = []

    for threshold in thresholds:
        threshold_mask = crop >= threshold
        boundary = threshold_mask & ~binary_erosion(
            threshold_mask, structure=np.ones((3, 3, 3), dtype=bool)
        )
        local_indices = np.argwhere(boundary)
        if len(local_indices) < 64:
            continue
        world_points = _to_world(local_indices + start, affine)
        distances = crown_tree.query(world_points, workers=1)[0]
        for radius_mm in guidance_radii_mm:
            selected = world_points[distances <= radius_mm]
            if len(selected) < 64:
                continue
            rank_score = (
                abs(float(threshold) - 500.0) / 4000.0
                + abs(float(radius_mm) - 2.0) / 10.0
                + abs(float(np.log(len(selected) / 8000.0))) * 0.05
            )
            candidates.append(
                SurfaceCandidate(
                    name=f"crown_guided_hu{threshold:g}_r{radius_mm:g}",
                    points=_sample(selected, points_per_candidate, rng),
                    metadata={
                        "mode": "crown_guided_cbct",
                        "threshold": float(threshold),
                        "guidance_radius_mm": float(radius_mm),
                        "support_voxels": int(len(crown_indices)),
                        "surface_points_before_sampling": int(len(selected)),
                        "rank_score": float(rank_score),
                        "roi_start_ijk": start.tolist(),
                        "roi_stop_ijk": stop.tolist(),
                    },
                )
            )
    if not candidates:
        raise ValueError(
            f"No native CBCT surface found near {jaw} crown guidance in {volume_path}"
        )
    return candidates
