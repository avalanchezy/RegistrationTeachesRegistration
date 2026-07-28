from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import open3d as o3d
import trimesh
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.data import apply_transform, load_manifest
from task2reg.surfaces import (
    threshold_aggregate_surface_candidates,
    threshold_surface_candidates,
    toothseg_surface_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add full-IOS fit and normal-consistency features to saved candidates."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--toothseg-dirs",
        type=Path,
        nargs="*",
        default=(),
        help="Directories containing STS2_<case>.nii.gz ToothSeg label volumes.",
    )
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--source-points", type=int, default=6000)
    parser.add_argument(
        "--max-candidates-per-jaw",
        type=int,
        default=0,
        help="Compute expensive geometry only for the best N selection-score candidates; 0 keeps all.",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse jaw outputs whose candidates.json was written successfully.",
    )
    return parser.parse_args()


def load_mesh_points_normals(path: Path, count: int, seed: int):
    mesh = trimesh.load(path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(vertices), size=min(count, len(vertices)), replace=False)
    normals = normals[indices]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    return vertices[indices], normals, np.asarray(mesh.bounds, dtype=np.float64)


def estimate_normals(points: np.ndarray) -> np.ndarray:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=2.5, max_nn=48)
    )
    normals = np.asarray(cloud.normals, dtype=np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    return normals


def target_cache_key(candidate: dict, default_volume: Path, default_jaw: str):
    metadata = candidate.get("target_metadata", {})
    volume_path = Path(metadata.get("volume_path", default_volume))
    tracked_jaw = str(metadata.get("axial_assignment", default_jaw))
    target_name = str(candidate.get("target", ""))
    target_name = target_name.removeprefix("axial_upper_").removeprefix("axial_lower_")
    return str(volume_path), tracked_jaw, target_name


def resolve_adaptive_volume_paths(candidates: list[dict], default_volume: Path) -> None:
    """Attach the ROI volume path omitted by legacy adaptive candidates."""
    target_volumes: dict[str, str] = {}
    available_volumes: set[str] = set()
    for candidate in candidates:
        metadata = candidate.get("target_metadata", {})
        volume = metadata.get("volume_path")
        if not volume:
            continue
        volume = str(Path(volume))
        target = str(candidate.get("target", ""))
        target = target.removeprefix("axial_upper_").removeprefix("axial_lower_")
        target_volumes[target] = volume
        available_volumes.add(volume)

    for candidate in candidates:
        metadata = candidate.get("target_metadata", {})
        if str(metadata.get("mode", "")) != "adaptive_threshold_roi":
            continue
        if metadata.get("volume_path"):
            continue
        coarse_target = str(metadata.get("coarse_target", ""))
        volume = target_volumes.get(coarse_target)
        if volume is None and len(available_volumes) == 1:
            volume = next(iter(available_volumes))
        metadata["volume_path"] = volume or str(default_volume)


def index_toothseg_volumes(directories: tuple[Path, ...] | list[Path]) -> dict[str, Path]:
    """Index label volumes once so geometry augmentation does not scan per jaw."""
    indexed: dict[str, Path] = {}
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.nii*")):
            name = path.name
            if name.endswith(".nii.gz"):
                name = name[:-7]
            elif name.endswith(".nii"):
                name = name[:-4]
            case_id = name.removeprefix("STS2_")
            indexed.setdefault(case_id, path.resolve())
    return indexed


def attach_toothseg_volume_path(
    candidates: list[dict], case_id: str, indexed_volumes: dict[str, Path]
) -> None:
    segmentation_path = indexed_volumes.get(case_id)
    for candidate in candidates:
        metadata = candidate.get("target_metadata", {})
        if str(metadata.get("mode", "")) != "toothseg":
            continue
        if segmentation_path is not None:
            metadata["volume_path"] = str(segmentation_path)


def adaptive_target_points(
    volume_path: Path,
    metadata: dict,
    source_points: np.ndarray,
    coarse_transform: np.ndarray,
    seed: int,
) -> np.ndarray | None:
    """Reconstruct the local low-HU target used by adaptive refinement."""
    start = np.asarray(metadata.get("roi_start_ijk", ()), dtype=np.int64)
    stop = np.asarray(metadata.get("roi_stop_ijk", ()), dtype=np.int64)
    if start.shape != (3,) or stop.shape != (3,) or np.any(stop <= start):
        return None

    image = nib.load(str(volume_path))
    volume = np.asanyarray(image.dataobj)
    start = np.maximum(start, 0)
    stop = np.minimum(stop, np.asarray(volume.shape, dtype=np.int64))
    crop = volume[tuple(slice(int(lo), int(hi)) for lo, hi in zip(start, stop))]
    threshold = float(metadata.get("threshold", 0.0))
    mask = crop >= threshold
    boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool))
    indices = np.argwhere(boundary) + start
    if len(indices) < 128:
        return None

    rng = np.random.default_rng(seed)
    if len(indices) > 30000:
        indices = indices[rng.choice(len(indices), size=30000, replace=False)]
    points = nib.affines.apply_affine(image.affine, indices.astype(np.float64))
    moved = apply_transform(source_points, coarse_transform)
    distances, _ = cKDTree(moved).query(points, workers=-1)
    radius = float(metadata.get("adaptive_radius_mm", 6.0))
    local_points = points[distances <= radius]
    return local_points if len(local_points) >= 128 else None


def build_target_cache(
    candidates: list[dict],
    default_volume: Path,
    default_jaw: str,
    extent: np.ndarray,
    source_points: np.ndarray,
    seed: int,
):
    resolve_adaptive_volume_paths(candidates, default_volume)
    requests: dict[tuple[str, str], list[dict]] = {}
    for candidate in candidates:
        metadata = candidate.get("target_metadata", {})
        mode = str(metadata.get("mode", ""))
        if mode not in {
            "threshold",
            "threshold_aggregate",
            "adaptive_threshold_roi",
            "toothseg",
        }:
            continue
        volume, tracked_jaw, _ = target_cache_key(candidate, default_volume, default_jaw)
        requests.setdefault((volume, tracked_jaw), []).append(candidate)

    cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for (volume, tracked_jaw), rows in requests.items():
        volume_path = Path(volume)
        tracked_thresholds = sorted(
            {
                float(row["target_metadata"].get("threshold", 0.0))
                for row in rows
                if str(row["target_metadata"].get("mode", "")) == "threshold"
            }
        )
        if tracked_thresholds:
            targets = threshold_surface_candidates(
                volume_path,
                tracked_jaw,
                extent,
                thresholds=tuple(tracked_thresholds),
                seed=seed,
            )
            for target in targets:
                cache[(volume, tracked_jaw, target.name)] = (
                    target.points,
                    estimate_normals(target.points),
                )
        aggregate_rows = [
            row
            for row in rows
            if str(row["target_metadata"].get("mode", "")) == "threshold_aggregate"
        ]
        if aggregate_rows:
            thresholds = sorted(
                {float(row["target_metadata"].get("threshold", 0.0)) for row in aggregate_rows}
            )
            counts = sorted(
                {int(row["target_metadata"].get("aggregate_count", 0)) for row in aggregate_rows}
            )
            targets = threshold_aggregate_surface_candidates(
                volume_path,
                extent,
                thresholds=tuple(thresholds),
                aggregate_counts=tuple(counts),
                seed=seed + 17,
            )
            for target in targets:
                cache[(volume, tracked_jaw, target.name)] = (
                    target.points,
                    estimate_normals(target.points),
                )
        toothseg_rows = [
            row
            for row in rows
            if str(row["target_metadata"].get("mode", "")) == "toothseg"
        ]
        if toothseg_rows:
            crown_fractions = sorted(
                {
                    float(row["target_metadata"].get("crown_fraction", 1.0))
                    for row in toothseg_rows
                    if float(row["target_metadata"].get("crown_fraction", 1.0)) < 1.0
                }
            )
            targets = toothseg_surface_candidates(
                volume_path,
                tracked_jaw,
                crown_fractions=tuple(crown_fractions),
                seed=seed + 31,
            )
            for target in targets:
                cache[(volume, tracked_jaw, target.name)] = (
                    target.points,
                    estimate_normals(target.points),
                )
        adaptive_rows = [
            row
            for row in rows
            if str(row["target_metadata"].get("mode", "")) == "adaptive_threshold_roi"
        ]
        for adaptive_index, row in enumerate(adaptive_rows):
            target = adaptive_target_points(
                volume_path,
                row["target_metadata"],
                source_points,
                np.asarray(row.get("transform_initial", row["transform"]), dtype=np.float64),
                seed + 101 + adaptive_index,
            )
            if target is None:
                continue
            cache[target_cache_key(row, default_volume, default_jaw)] = (
                target,
                estimate_normals(target),
            )
    return cache


def geometric_features(
    points: np.ndarray,
    normals: np.ndarray,
    target: np.ndarray,
    target_normals: np.ndarray,
    transform: np.ndarray,
) -> dict[str, float]:
    moved = apply_transform(points, transform)
    moved_normals = normals @ transform[:3, :3].T
    moved_normals /= np.maximum(np.linalg.norm(moved_normals, axis=1, keepdims=True), 1e-8)
    target_tree = cKDTree(target)
    distances, indices = target_tree.query(moved, workers=-1)
    ordered = np.sort(distances)

    def trimmed(fraction: float) -> float:
        keep = max(32, int(len(ordered) * fraction))
        return float(ordered[:keep].mean())

    source_tree = cKDTree(moved)
    target_distances, _ = source_tree.query(target, workers=-1)
    normal_keep = distances <= max(3.0, float(np.quantile(distances, 0.35)))
    cosine = np.abs(np.sum(moved_normals[normal_keep] * target_normals[indices[normal_keep]], axis=1))
    return {
        "full_trim_10_mm": trimmed(0.10),
        "full_trim_20_mm": trimmed(0.20),
        "full_trim_35_mm": trimmed(0.35),
        "full_distance_median_mm": float(np.median(distances)),
        "full_distance_p90_mm": float(np.quantile(distances, 0.9)),
        "full_overlap_1mm": float(np.mean(distances <= 1.0)),
        "full_overlap_2mm": float(np.mean(distances <= 2.0)),
        "full_overlap_3mm": float(np.mean(distances <= 3.0)),
        "full_target_coverage_2mm": float(np.mean(target_distances <= 2.0)),
        "full_normal_abs_cosine": float(cosine.mean()) if len(cosine) else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.case_ids or ())
    toothseg_volumes = index_toothseg_volumes(args.toothseg_dirs)
    records = {(row.case_id, row.jaw): row for row in load_manifest(args.manifest)}
    paths: dict[tuple[str, str], list[Path]] = {}
    for run in args.runs:
        for path in sorted(run.resolve().glob("*_*/candidates.json")):
            case_id, jaw = path.parent.name.rsplit("_", 1)
            if jaw in {"upper", "lower"} and (not wanted or case_id in wanted):
                paths.setdefault((case_id, jaw), []).append(path)
    summary = []
    for group_index, (key, candidate_paths) in enumerate(sorted(paths.items())):
        if key not in records:
            continue
        case_dir = args.output_dir / f"{key[0]}_{key[1]}"
        output_path = case_dir / "candidates.json"
        if args.resume and output_path.is_file():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            augmented = sum(
                int(row.get("full_geometry_available", 0)) for row in existing
            )
            summary.append(
                {
                    "case_id": key[0],
                    "jaw": key[1],
                    "rows": len(existing),
                    "augmented": augmented,
                    "resumed": 1,
                }
            )
            print(
                f"[{len(summary)}/{len(paths)}] {key[0]} {key[1]}: "
                f"reused {augmented}/{len(existing)} candidates",
                flush=True,
            )
            continue
        record = records[key]
        points, normals, bounds = load_mesh_points_normals(
            Path(record.ios_path), args.source_points, args.seed + group_index
        )
        extent = bounds[1] - bounds[0]
        merged = []
        for candidate_path in candidate_paths:
            run_name = candidate_path.parents[1].name
            for candidate in json.loads(candidate_path.read_text(encoding="utf-8")):
                row = dict(candidate)
                row["candidate_run"] = run_name
                merged.append(row)
        attach_toothseg_volume_path(merged, key[0], toothseg_volumes)
        augment_rows = merged
        if args.max_candidates_per_jaw > 0:
            augment_rows = sorted(
                merged, key=lambda row: float(row.get("selection_score_mm", np.inf))
            )[: args.max_candidates_per_jaw]
        augment_ids = {id(row) for row in augment_rows}
        target_cache = build_target_cache(
            augment_rows,
            Path(record.cbct_path),
            key[1],
            extent,
            points,
            args.seed + group_index,
        )
        skipped = 0
        for row in merged:
            if id(row) not in augment_ids:
                skipped += 1
                row["full_geometry_available"] = 0
                continue
            cache_key = target_cache_key(row, Path(record.cbct_path), key[1])
            target_bundle = target_cache.get(cache_key)
            if target_bundle is None:
                skipped += 1
                row["full_geometry_available"] = 0
            else:
                target, target_normals = target_bundle
                row.update(
                    geometric_features(
                        points,
                        normals,
                        target,
                        target_normals,
                        np.asarray(row["transform"], dtype=np.float64),
                    )
                )
                row["full_geometry_available"] = 1
        case_dir.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        temporary.replace(output_path)
        result_sources = [path.with_name("result.json") for path in candidate_paths]
        result_source = next((path for path in result_sources if path.exists()), None)
        if result_source is not None:
            shutil.copy2(result_source, case_dir / "result.json")
        augmented = len(merged) - skipped
        summary.append(
            {
                "case_id": key[0],
                "jaw": key[1],
                "rows": len(merged),
                "augmented": augmented,
                "resumed": 0,
            }
        )
        print(
            f"[{len(summary)}/{len(paths)}] {key[0]} {key[1]}: "
            f"augmented {augmented}/{len(merged)} candidates",
            flush=True,
        )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
