from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from threading import Lock

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.crown_network import probabilities_to_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep OOF crown-localizer mask postprocessing.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--probability-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85),
    )
    parser.add_argument("--minimum-component-voxels", type=int, nargs="+", default=(12, 50, 100))
    parser.add_argument("--maximum-components", type=int, nargs="+", default=(1, 2, 0))
    parser.add_argument("--minimum-hu", type=float, nargs="+", default=(-1000.0, 0.0, 150.0, 300.0))
    parser.add_argument("--jobs", type=int, default=1)
    return parser.parse_args()


def dice(prediction: np.ndarray, target: np.ndarray, class_id: int) -> float:
    predicted = prediction == class_id
    truth = target == class_id
    return float((2 * np.sum(predicted & truth) + 1) / (np.sum(predicted) + np.sum(truth) + 1))


def world_points(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    indices = np.argwhere(mask).astype(np.float64, copy=False)
    return indices @ affine[:3, :3].T + affine[:3, 3]


def surface_metrics(
    predicted_points: np.ndarray,
    target_points: np.ndarray,
    target_tree: cKDTree | None = None,
    target_centroid: np.ndarray | None = None,
    empty_distance_mm: float = 200.0,
) -> dict[str, float]:
    if len(predicted_points) == 0 or len(target_points) == 0:
        return {
            "symmetric_chamfer_mm": empty_distance_mm,
            "predicted_to_target_mm": empty_distance_mm,
            "target_to_predicted_mm": empty_distance_mm,
            "precision_at_2mm": 0.0,
            "coverage_at_2mm": 0.0,
            "centroid_error_mm": empty_distance_mm,
        }
    target_tree = target_tree or cKDTree(target_points)
    target_centroid = target_points.mean(axis=0) if target_centroid is None else target_centroid
    predicted_to_target = target_tree.query(predicted_points, workers=1)[0]
    target_to_predicted = cKDTree(predicted_points).query(target_points, workers=1)[0]
    return {
        "symmetric_chamfer_mm": float(
            0.5 * (np.mean(predicted_to_target) + np.mean(target_to_predicted))
        ),
        "predicted_to_target_mm": float(np.mean(predicted_to_target)),
        "target_to_predicted_mm": float(np.mean(target_to_predicted)),
        "precision_at_2mm": float(np.mean(predicted_to_target <= 2.0)),
        "coverage_at_2mm": float(np.mean(target_to_predicted <= 2.0)),
        "centroid_error_mm": float(
            np.linalg.norm(predicted_points.mean(axis=0) - target_centroid)
        ),
    }


def evaluate_configuration(
    configuration: tuple[float, int, int, float],
    payloads: list[tuple],
    metric_cache: dict[tuple[str, int, bytes], tuple[float, int, dict[str, float]]],
    cache_lock: Lock,
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    threshold, minimum_voxels, maximum_components, minimum_hu = configuration
    values = []
    volumes = []
    geometry = []
    case_rows = []
    for (
        case_id,
        probabilities,
        target,
        image,
        affine,
        target_points,
        target_trees,
        target_centroids,
    ) in payloads:
        prediction = probabilities_to_labels(
            probabilities,
            minimum_probability=threshold,
            minimum_component_voxels=minimum_voxels,
            maximum_components=maximum_components,
            image_hu=image,
            minimum_hu=minimum_hu,
        )
        for class_id, jaw in ((1, "upper"), (2, "lower")):
            predicted_mask = prediction == class_id
            digest = hashlib.blake2b(
                np.packbits(predicted_mask, axis=None).tobytes(), digest_size=16
            ).digest()
            cache_key = (case_id, class_id, digest)
            with cache_lock:
                cached = metric_cache.get(cache_key)
            if cached is None:
                value = dice(prediction, target, class_id)
                volume = int(np.sum(predicted_mask))
                jaw_geometry = surface_metrics(
                    world_points(predicted_mask, affine),
                    target_points[class_id],
                    target_tree=target_trees[class_id],
                    target_centroid=target_centroids[class_id],
                )
                cached = (value, volume, jaw_geometry)
                with cache_lock:
                    metric_cache.setdefault(cache_key, cached)
            else:
                value, volume, jaw_geometry = cached
            values.append(value)
            volumes.append(volume)
            geometry.append(jaw_geometry)
            case_rows.append(
                {
                    "threshold": threshold,
                    "minimum_component_voxels": minimum_voxels,
                    "maximum_components": maximum_components,
                    "minimum_hu": minimum_hu,
                    "case_id": case_id,
                    "jaw": jaw,
                    "dice": value,
                    "predicted_voxels": volume,
                    **jaw_geometry,
                }
            )
    row = {
        "threshold": threshold,
        "minimum_component_voxels": minimum_voxels,
        "maximum_components": maximum_components,
        "minimum_hu": minimum_hu,
        "jaws": len(values),
        "mean_dice": float(np.mean(values)),
        "median_dice": float(np.median(values)),
        "minimum_dice": float(np.min(values)),
        "mean_predicted_voxels": float(np.mean(volumes)),
        **{
            f"mean_{key}": float(np.mean([item[key] for item in geometry]))
            for key in geometry[0]
        },
    }
    return row, case_rows


def main() -> None:
    args = parse_args()
    payloads = []
    for path in sorted(args.probability_dir.glob("*.npz")):
        data_path = args.data_dir / path.name
        if not data_path.exists():
            continue
        with np.load(path, allow_pickle=False) as prediction_payload:
            probabilities = prediction_payload["probabilities"].astype(np.float32)
        with np.load(data_path, allow_pickle=False) as data_payload:
            target = data_payload["label"].astype(np.uint8)
            image = data_payload["image"].astype(np.int16)
            affine = data_payload["affine"].astype(np.float64)
        target_points = {
            class_id: world_points(target == class_id, affine)
            for class_id in (1, 2)
        }
        target_trees = {
            class_id: cKDTree(points) if len(points) else None
            for class_id, points in target_points.items()
        }
        target_centroids = {
            class_id: points.mean(axis=0) if len(points) else np.zeros(3, dtype=np.float64)
            for class_id, points in target_points.items()
        }
        payloads.append(
            (
                path.stem,
                probabilities,
                target,
                image,
                affine,
                target_points,
                target_trees,
                target_centroids,
            )
        )
    if not payloads:
        raise RuntimeError("No matching probability and pseudo-GT files were found")

    configurations = list(
        product(
            args.thresholds,
            args.minimum_component_voxels,
            args.maximum_components,
            args.minimum_hu,
        )
    )
    jobs = max(1, int(args.jobs))
    metric_cache: dict[
        tuple[str, int, bytes], tuple[float, int, dict[str, float]]
    ] = {}
    cache_lock = Lock()
    if jobs == 1:
        evaluated = [
            evaluate_configuration(item, payloads, metric_cache, cache_lock)
            for item in configurations
        ]
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            evaluated = list(
                executor.map(
                    lambda item: evaluate_configuration(
                        item, payloads, metric_cache, cache_lock
                    ),
                    configurations,
                )
            )
    rows = [item[0] for item in evaluated]
    per_case_rows = [case_row for item in evaluated for case_row in item[1]]
    rows.sort(
        key=lambda row: (
            float(row["mean_symmetric_chamfer_mm"]),
            -float(row["mean_coverage_at_2mm"]),
            -float(row["mean_dice"]),
            abs(float(row["mean_predicted_voxels"]) - 3700.0),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in (("grid.csv", rows), ("per_case.csv", per_case_rows)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    best_dice = max(rows, key=lambda row: (float(row["mean_dice"]), -float(row["mean_symmetric_chamfer_mm"])))
    summary = {
        "cases": len(payloads),
        "best_geometry": rows[0],
        "best_dice": best_dice,
        "grid_rows": len(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
