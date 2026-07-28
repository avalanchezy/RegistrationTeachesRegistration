from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.candidate_learning import load_candidate_group
from task2reg.data import apply_transform, ios_pca_side_variants, load_ios_points, load_manifest
from task2reg.geometry import robust_trimmed_icp
from task2reg.metrics import evaluate_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locally refine existing registration candidates against an OOF crown mask."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labeled-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--crown-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="Train-Labeled")
    parser.add_argument("--case-ids", nargs="*", default=[])
    parser.add_argument("--source-points", type=int, default=4000)
    parser.add_argument("--top-selection", type=int, default=12)
    parser.add_argument("--top-crown", type=int, default=12)
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.25, 0.50, 1.0))
    parser.add_argument("--maximum-center-motion-mm", type=float, default=4.0)
    parser.add_argument("--maximum-angle-deg", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def mask_points(path: Path, jaw: str) -> np.ndarray:
    image = nib.load(str(path))
    labels = np.asanyarray(image.dataobj)
    indices = np.argwhere(labels == (1 if jaw == "upper" else 17)).astype(np.float64)
    if len(indices) < 32:
        raise ValueError(f"No usable {jaw} mask in {path}")
    return indices @ np.asarray(image.affine)[:3, :3].T + np.asarray(image.affine)[:3, 3]


def trimmed_mean(values: np.ndarray, fraction: float = 0.20) -> float:
    keep = max(1, int(np.ceil(len(values) * fraction)))
    return float(np.mean(np.partition(values, keep - 1)[:keep]))


def crown_metrics(source: np.ndarray, target: np.ndarray, transform: np.ndarray) -> dict[str, float]:
    moved = apply_transform(source, transform)
    source_distances = cKDTree(target).query(moved, workers=1)[0]
    target_distances = cKDTree(moved).query(target, workers=1)[0]
    source_trim = trimmed_mean(source_distances)
    target_trim = trimmed_mean(target_distances)
    return {
        "crown_source_trim20_mm": source_trim,
        "crown_target_trim20_mm": target_trim,
        "crown_symmetric_trim20_mm": 0.5 * (source_trim + target_trim),
        "crown_source_median_mm": float(np.median(source_distances)),
        "crown_target_median_mm": float(np.median(target_distances)),
        "crown_source_overlap_2mm": float(np.mean(source_distances <= 2.0)),
        "crown_target_coverage_2mm": float(np.mean(target_distances <= 2.0)),
        "crown_centroid_error_mm": float(np.linalg.norm(moved.mean(axis=0) - target.mean(axis=0))),
    }


def interpolate_transform(
    initial: np.ndarray,
    refined: np.ndarray,
    source: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    initial_center = apply_transform(source, initial).mean(axis=0)
    refined_center = apply_transform(source, refined).mean(axis=0)
    delta_rotation = refined[:3, :3] @ initial[:3, :3].T
    rotation = Rotation.from_rotvec(
        Rotation.from_matrix(delta_rotation).as_rotvec() * alpha
    ).as_matrix()
    interpolated_center = initial_center + alpha * (refined_center - initial_center)
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = rotation
    delta[:3, 3] = interpolated_center - rotation @ initial_center
    return delta @ initial


def source_variants(points: np.ndarray, candidates: list[dict]) -> dict[str, np.ndarray]:
    fractions = set()
    for candidate in candidates:
        name = str(candidate.get("source_variant", ""))
        if name.startswith("pca_"):
            try:
                fractions.add(float(name.rsplit("_", 1)[-1]))
            except ValueError:
                pass
    return dict(ios_pca_side_variants(points, tuple(sorted(fractions)) or (0.25, 0.35), True))


def select_indices(candidates: list[dict], top_selection: int, top_crown: int) -> list[int]:
    selection = np.asarray([float(row.get("selection_score_mm", 100.0)) for row in candidates])
    crown = np.asarray([float(row["crown_symmetric_trim20_mm"]) for row in candidates])
    selected = set(np.argsort(selection)[:top_selection].tolist())
    selected.update(np.argsort(crown)[:top_crown].tolist())
    ordered = sorted(selected, key=lambda index: (min(selection[index], crown[index]), index))
    unique = []
    fingerprints = set()
    for index in ordered:
        fingerprint = tuple(np.round(np.asarray(candidates[index]["transform"]), 4).ravel())
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            unique.append(index)
    return unique


def main() -> None:
    args = parse_args()
    wanted = set(map(str, args.case_ids))
    records = {
        (record.case_id, record.jaw): record
        for record in load_manifest(args.manifest)
        if record.split == args.split
        and record.complete
        and (not wanted or record.case_id in wanted)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summaries = []

    profiles = {
        "tight": ((3.0, 2.0, 1.25), 0.35),
        "balanced": ((5.0, 3.0, 2.0, 1.25), 0.50),
    }
    for group_index, (key, record) in enumerate(sorted(records.items())):
        candidates = load_candidate_group(args.labeled_runs, key)
        mask_path = args.crown_mask_dir / f"STS2_{key[0]}.nii.gz"
        if not candidates or not mask_path.exists():
            continue
        full_source, _ = load_ios_points(
            record.ios_path,
            max(args.source_points * 4, 16000),
            args.seed + group_index,
        )
        variants = source_variants(full_source, candidates)
        for name, points in list(variants.items()):
            if len(points) > args.source_points:
                variants[name] = points[rng.choice(len(points), args.source_points, replace=False)]
        evaluation_source = full_source
        if len(evaluation_source) > args.source_points:
            evaluation_source = evaluation_source[
                rng.choice(len(evaluation_source), args.source_points, replace=False)
            ]
        target = mask_points(mask_path, key[1])
        ground_truth = (
            np.load(record.transform_path).astype(np.float64)
            if record.transform_path
            else None
        )

        usable = []
        for candidate in candidates:
            source = variants.get(str(candidate.get("source_variant", "")))
            if source is None:
                continue
            row = dict(candidate)
            row.update(crown_metrics(source, target, np.asarray(row["transform"], dtype=np.float64)))
            usable.append(row)
        selected_indices = select_indices(usable, args.top_selection, args.top_crown)
        refined_rows = []
        rejected = 0
        for candidate_index in selected_indices:
            candidate = usable[candidate_index]
            source = variants[str(candidate["source_variant"])]
            initial = np.asarray(candidate["transform"], dtype=np.float64)
            initial_score = float(candidate["crown_symmetric_trim20_mm"])
            initial_center = apply_transform(source, initial).mean(axis=0)
            for profile_name, (schedule, trim_fraction) in profiles.items():
                refined, correspondences = robust_trimmed_icp(
                    source,
                    target,
                    initial,
                    distance_schedule=schedule,
                    iterations_per_level=5,
                    trim_fraction=trim_fraction,
                )
                refined_center = apply_transform(source, refined).mean(axis=0)
                center_motion = float(np.linalg.norm(refined_center - initial_center))
                delta_rotation = refined[:3, :3] @ initial[:3, :3].T
                angle = float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(delta_rotation).as_rotvec())))
                if center_motion > args.maximum_center_motion_mm or angle > args.maximum_angle_deg:
                    rejected += 1
                    continue
                for alpha in args.alphas:
                    transform = interpolate_transform(initial, refined, source, alpha)
                    metrics = crown_metrics(source, target, transform)
                    improvement = initial_score - metrics["crown_symmetric_trim20_mm"]
                    if improvement <= 0.005:
                        continue
                    row = dict(candidate)
                    row.pop("candidate_run", None)
                    row.pop("candidate_jaw", None)
                    row["source_candidate_run"] = str(candidate.get("candidate_run", ""))
                    row["source_candidate_index"] = candidate_index
                    row["method"] = f"{candidate.get('method', 'candidate')}+crown-icp-{profile_name}"
                    row["transform_initial"] = initial.tolist()
                    row["transform"] = transform.tolist()
                    source_centroid = np.asarray(row.get("source_full_centroid", evaluation_source.mean(axis=0)))
                    row["predicted_full_centroid"] = apply_transform(source_centroid[None], transform)[0].tolist()
                    row["correspondences"] = int(correspondences)
                    row.update(metrics)
                    if ground_truth is not None:
                        row.update(
                            evaluate_transform(
                                evaluation_source, transform, ground_truth
                            )
                        )
                    row.update(
                        {
                            "crown_refinement_profile": profile_name,
                            "crown_refinement_initial_trim20_mm": initial_score,
                            "crown_refinement_improvement_mm": float(improvement),
                            "crown_refinement_center_motion_mm": center_motion * float(alpha),
                            "crown_refinement_angle_deg": angle * float(alpha),
                            "crown_refinement_alpha": float(alpha),
                        }
                    )
                    refined_rows.append(row)

        if not refined_rows:
            continue
        group_dir = args.output_dir / f"{key[0]}_{key[1]}"
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / "candidates.json").write_text(
            json.dumps(refined_rows, indent=2), encoding="utf-8"
        )
        selected_by_crown = min(refined_rows, key=lambda row: row["crown_symmetric_trim20_mm"])
        summary = {
            "case_id": key[0],
            "jaw": key[1],
            "input_candidates": len(usable),
            "selected_initializations": len(selected_indices),
            "refined_candidates": len(refined_rows),
            "rejected_refinements": rejected,
            "best_crown_symmetric_trim20_mm": float(
                selected_by_crown["crown_symmetric_trim20_mm"]
            ),
        }
        if ground_truth is not None:
            baseline_oracle = min(float(row["mean_tre_mm"]) for row in usable)
            refined_oracle = min(float(row["mean_tre_mm"]) for row in refined_rows)
            summary.update(
                {
                    "baseline_oracle_tre_mm": baseline_oracle,
                    "refined_oracle_tre_mm": refined_oracle,
                    "combined_oracle_tre_mm": min(baseline_oracle, refined_oracle),
                    "best_crown_refined_tre_mm": float(
                        selected_by_crown["mean_tre_mm"]
                    ),
                }
            )
        summaries.append(summary)
        (group_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if ground_truth is not None:
            message = (
                f"oracle {baseline_oracle:.3f} -> "
                f"{min(baseline_oracle, refined_oracle):.3f} mm"
            )
        else:
            message = (
                f"best crown fit={selected_by_crown['crown_symmetric_trim20_mm']:.3f} mm"
            )
        print(
            f"[{len(summaries)}/{len(records)}] {key[0]} {key[1]}: "
            f"{message}, refined={len(refined_rows)}",
            flush=True,
        )

    if not summaries:
        raise RuntimeError("No crown-refined candidate groups were generated")
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    report = {
        "split": args.split,
        "groups": len(summaries),
        "refined_candidates": int(sum(row["refined_candidates"] for row in summaries)),
        "mean_best_crown_symmetric_trim20_mm": float(
            np.mean([row["best_crown_symmetric_trim20_mm"] for row in summaries])
        ),
    }
    if "baseline_oracle_tre_mm" in summaries[0]:
        report.update(
            {
                "mean_baseline_oracle_tre_mm": float(
                    np.mean([row["baseline_oracle_tre_mm"] for row in summaries])
                ),
                "mean_refined_oracle_tre_mm": float(
                    np.mean([row["refined_oracle_tre_mm"] for row in summaries])
                ),
                "mean_combined_oracle_tre_mm": float(
                    np.mean([row["combined_oracle_tre_mm"] for row in summaries])
                ),
                "mean_best_crown_refined_tre_mm": float(
                    np.mean([row["best_crown_refined_tre_mm"] for row in summaries])
                ),
            }
        )
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
