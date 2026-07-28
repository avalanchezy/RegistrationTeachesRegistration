from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.candidate_learning import load_candidate_group as _load_candidate_group
from task2reg.data import (
    apply_transform,
    ios_pca_side_variants,
    load_ios_points,
    load_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure OOF crown-mask consistency of existing registration candidates."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labeled-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--crown-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="Train-Labeled")
    parser.add_argument(
        "--augmented-root",
        type=Path,
        help="Optionally write candidate runs augmented with the crown-consistency fields.",
    )
    parser.add_argument("--case-ids", nargs="*", default=[])
    parser.add_argument("--source-points", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def _fractional_rank(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / float(len(values) - 1)


def _trimmed_mean(values: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(len(values) * fraction)))
    return float(np.mean(np.partition(values, count - 1)[:count]))


def _mask_points(path: Path, jaw: str) -> np.ndarray:
    image = nib.load(str(path))
    labels = np.asanyarray(image.dataobj)
    class_id = 1 if jaw == "upper" else 17
    indices = np.argwhere(labels == class_id).astype(np.float64, copy=False)
    if len(indices) < 32:
        raise ValueError(f"No usable {jaw} crown mask in {path}")
    affine = np.asarray(image.affine, dtype=np.float64)
    return indices @ affine[:3, :3].T + affine[:3, 3]


def _source_variants(points: np.ndarray, candidates: list[dict]) -> dict[str, np.ndarray]:
    fractions = set()
    for candidate in candidates:
        name = str(candidate.get("source_variant", ""))
        if name.startswith("pca_"):
            try:
                fractions.add(float(name.rsplit("_", 1)[-1]))
            except ValueError:
                continue
    return dict(
        ios_pca_side_variants(
            points,
            tuple(sorted(fractions)) or (0.25, 0.35),
            include_full=True,
        )
    )


def _write_augmented_group(
    candidates: list[dict],
    key: tuple[str, str],
    run_dirs: list[Path],
    augmented_root: Path,
) -> dict[str, int]:
    case_id, jaw = key
    group_name = f"{case_id}_{jaw}"
    written: dict[str, int] = {}
    for source_run in run_dirs:
        selected = [
            candidate
            for candidate in candidates
            if str(candidate.get("candidate_run", "")) == str(source_run)
            and "crown_symmetric_trim20_mm" in candidate
        ]
        if not selected:
            continue
        destination_group = augmented_root / source_run.name / group_name
        destination_group.mkdir(parents=True, exist_ok=True)
        serialized = []
        for candidate in selected:
            payload = dict(candidate)
            payload.pop("candidate_run", None)
            payload.pop("candidate_jaw", None)
            serialized.append(payload)
        (destination_group / "candidates.json").write_text(
            json.dumps(serialized, indent=2), encoding="utf-8"
        )
        source_result = source_run / group_name / "result.json"
        if source_result.exists():
            shutil.copy2(source_result, destination_group / "result.json")
        written[str(source_run)] = 1
    return written


def main() -> None:
    args = parse_args()
    wanted = set(args.case_ids)
    records = {
        (record.case_id, record.jaw): record
        for record in load_manifest(args.manifest)
        if record.split == args.split
        and record.complete
        and (not wanted or record.case_id in wanted)
    }
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | str]] = []
    group_summaries = []
    augmented_group_counts = {str(run_dir): 0 for run_dir in args.labeled_runs}
    if args.augmented_root is not None:
        run_names = [run_dir.name for run_dir in args.labeled_runs]
        if len(run_names) != len(set(run_names)):
            raise ValueError("Augmented candidate runs require unique source directory names")

    for group_index, (key, record) in enumerate(sorted(records.items())):
        candidates = _load_candidate_group(args.labeled_runs, key)
        if not candidates:
            continue
        case_id, jaw = key
        mask_path = args.crown_mask_dir / f"STS2_{case_id}.nii.gz"
        if not mask_path.exists():
            continue
        target = _mask_points(mask_path, jaw)
        target_tree = cKDTree(target)
        source_points, _ = load_ios_points(
            record.ios_path,
            max(args.source_points * 6, 16000),
            args.seed + group_index,
        )
        variants = _source_variants(source_points, candidates)
        sampled_variants = {}
        for name, points in variants.items():
            if len(points) > args.source_points:
                indices = rng.choice(len(points), args.source_points, replace=False)
                points = points[indices]
            sampled_variants[name] = points

        group_rows = []
        for candidate_index, candidate in enumerate(candidates):
            source_name = str(candidate.get("source_variant", "full"))
            source = sampled_variants.get(source_name)
            if source is None:
                continue
            transformed = apply_transform(
                source, np.asarray(candidate["transform"], dtype=np.float64)
            )
            source_distances = target_tree.query(transformed, workers=1)[0]
            target_distances = cKDTree(transformed).query(target, workers=1)[0]
            source_trim20 = _trimmed_mean(source_distances, 0.20)
            target_trim20 = _trimmed_mean(target_distances, 0.20)
            row = {
                "case_id": case_id,
                "jaw": jaw,
                "candidate_index": candidate_index,
                "candidate_run": str(candidate.get("candidate_run", "")),
                "target": str(candidate.get("target", "")),
                "source_variant": source_name,
                "method": str(candidate.get("method", "")),
                "selection_score_mm": float(candidate.get("selection_score_mm", 100.0)),
                "rank_score_mm": float(candidate.get("rank_score_mm", 100.0)),
                "mean_tre_mm": float(candidate.get("mean_tre_mm", np.nan)),
                "crown_source_trim20_mm": source_trim20,
                "crown_target_trim20_mm": target_trim20,
                "crown_symmetric_trim20_mm": 0.5 * (source_trim20 + target_trim20),
                "crown_source_median_mm": float(np.median(source_distances)),
                "crown_target_median_mm": float(np.median(target_distances)),
                "crown_source_overlap_2mm": float(np.mean(source_distances <= 2.0)),
                "crown_target_coverage_2mm": float(np.mean(target_distances <= 2.0)),
                "crown_centroid_error_mm": float(
                    np.linalg.norm(transformed.mean(axis=0) - target.mean(axis=0))
                ),
            }
            candidate.update(
                {
                    name: value
                    for name, value in row.items()
                    if name.startswith("crown_")
                }
            )
            rows.append(row)
            group_rows.append(row)

        if not group_rows:
            continue
        selection = np.asarray([row["selection_score_mm"] for row in group_rows])
        crown = np.asarray([row["crown_symmetric_trim20_mm"] for row in group_rows])
        truth = np.asarray([row["mean_tre_mm"] for row in group_rows])
        selection_rank = _fractional_rank(selection)
        crown_rank = _fractional_rank(crown)
        if np.all(np.isfinite(truth)):
            policies = {}
            for alpha in np.linspace(0.0, 1.0, 21):
                selected = int(
                    np.argmin((1.0 - alpha) * selection_rank + alpha * crown_rank)
                )
                policies[f"alpha_{alpha:.2f}"] = float(truth[selected])
            group_summaries.append(
                {
                    "case_id": case_id,
                    "jaw": jaw,
                    "candidates": len(group_rows),
                    "oracle_tre_mm": float(np.min(truth)),
                    **policies,
                }
            )
            message = f"oracle={np.min(truth):.3f} mm"
        else:
            group_summaries.append(
                {
                    "case_id": case_id,
                    "jaw": jaw,
                    "candidates": len(group_rows),
                    "minimum_crown_symmetric_trim20_mm": float(np.min(crown)),
                }
            )
            message = f"best crown fit={np.min(crown):.3f} mm"
        print(
            f"[{group_index + 1}/{len(records)}] {case_id} {jaw}: "
            f"{len(group_rows)} candidates, {message}",
            flush=True,
        )
        if args.augmented_root is not None:
            written = _write_augmented_group(
                candidates,
                key,
                args.labeled_runs,
                args.augmented_root,
            )
            for source_run in written:
                augmented_group_counts[source_run] += 1

    if not group_summaries:
        raise RuntimeError("No candidate groups with matching crown masks were evaluated")
    policy_names = [name for name in group_summaries[0] if name.startswith("alpha_")]
    policy_summary = [
        {
            "policy": name,
            "mean_tre_mm": float(np.mean([row[name] for row in group_summaries])),
            "median_tre_mm": float(np.median([row[name] for row in group_summaries])),
            "p90_tre_mm": float(np.quantile([row[name] for row in group_summaries], 0.90)),
            "max_tre_mm": float(np.max([row[name] for row in group_summaries])),
        }
        for name in policy_names
    ]
    policy_summary.sort(key=lambda row: row["mean_tre_mm"])
    summary = {
        "split": args.split,
        "groups": len(group_summaries),
        "candidates": len(rows),
    }
    if policy_summary:
        summary.update(
            {
                "oracle_mean_tre_mm": float(
                    np.mean([row["oracle_tre_mm"] for row in group_summaries])
                ),
                "best_policy": policy_summary[0],
                "policies": policy_summary,
            }
        )
    augmented_runs = []
    if args.augmented_root is not None:
        for source_run in args.labeled_runs:
            destination_run = args.augmented_root / source_run.name
            augmented_runs.append(
                {
                    "source": str(source_run),
                    "destination": str(destination_run),
                    "groups": augmented_group_counts[str(source_run)],
                }
            )
        summary["augmented_runs"] = augmented_runs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in (
        ("candidate_scores.csv", rows),
        ("per_group.csv", group_summaries),
        ("policies.csv", policy_summary),
    ):
        if not table:
            continue
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
