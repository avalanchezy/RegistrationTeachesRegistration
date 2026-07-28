from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweep_multimodal_reranker import (
    exact_fallback_errors,
    load_exact_errors,
    summarize,
    write_csv,
)
from task2reg.candidate_learning import is_opposite_axial_target, load_candidate_groups
from task2reg.data import load_ios_points, load_manifest
from task2reg.metrics import evaluate_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep label-free threshold/ToothSeg transform consensus."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--threshold-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--toothseg-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exact-loo", type=Path)
    parser.add_argument("--top-k", type=int, nargs="+", default=(3, 5, 8, 10, 15, 20))
    parser.add_argument(
        "--rank-weights", type=float, nargs="+", default=(0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
    )
    parser.add_argument(
        "--selection", choices=("threshold", "toothseg", "midpoint"), nargs="+",
        default=("threshold", "toothseg", "midpoint")
    )
    parser.add_argument(
        "--agreement-thresholds", type=float, nargs="+", default=(0.5, 1.0, 1.5, 2.0, 3.0)
    )
    parser.add_argument("--anchor-radius-mm", type=float, default=20.0)
    parser.add_argument("--sample-points", type=int, default=2000)
    parser.add_argument("--min-gate-jaws", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--exclude-upper-opposite-axial", action="store_true")
    return parser.parse_args()


def project_orthogonal(linear: np.ndarray, determinant: int) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(linear, dtype=np.float64))
    projected = u @ vt
    if int(np.sign(np.linalg.det(projected))) != determinant:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def midpoint_transform(first: np.ndarray, second: np.ndarray, center: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    determinant = int(np.sign(np.linalg.det(first[:3, :3])))
    if determinant != int(np.sign(np.linalg.det(second[:3, :3]))):
        raise ValueError("Cannot interpolate transforms with different chirality")
    first_rotation = project_orthogonal(first[:3, :3], determinant)
    second_rotation = project_orthogonal(second[:3, :3], determinant)
    relative = project_orthogonal(second_rotation @ first_rotation.T, 1)
    half = Rotation.from_rotvec(0.5 * Rotation.from_matrix(relative).as_rotvec()).as_matrix()
    rotation = half @ first_rotation
    mapped_center = 0.5 * (
        first_rotation @ center + first[:3, 3]
        + second_rotation @ center + second[:3, 3]
    )
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = rotation
    output[:3, 3] = mapped_center - rotation @ center
    return output


def anchor_points(center: np.ndarray, radius: float) -> np.ndarray:
    offsets = np.vstack((np.zeros((1, 3)), radius * np.eye(3), -radius * np.eye(3)))
    return np.asarray(center, dtype=np.float64)[None, :] + offsets


def transformed(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def disagreement_mm(first: np.ndarray, second: np.ndarray, anchors: np.ndarray) -> float:
    return float(np.linalg.norm(transformed(anchors, first) - transformed(anchors, second), axis=1).mean())


def candidate_pool(rows: list[dict], jaw: str, top_k: int, exclude_opposite: bool) -> list[dict]:
    pool = rows
    if exclude_opposite and jaw == "upper":
        filtered = [row for row in pool if not is_opposite_axial_target(row, jaw)]
        if filtered:
            pool = filtered
    return sorted(pool, key=lambda row: float(row["selection_score_mm"]))[:top_k]


def select_pair(
    threshold_rows: list[dict],
    toothseg_rows: list[dict],
    jaw: str,
    top_k: int,
    rank_weight: float,
    anchor_radius: float,
    exclude_opposite: bool,
):
    threshold = candidate_pool(threshold_rows, jaw, top_k, exclude_opposite)
    toothseg = candidate_pool(toothseg_rows, jaw, top_k, False)
    center = np.asarray(threshold_rows[0]["source_full_centroid"], dtype=np.float64)
    anchors = anchor_points(center, anchor_radius)
    pairs = []
    for threshold_index, toothseg_index in itertools.product(
        range(len(threshold)), range(len(toothseg))
    ):
        first = np.asarray(threshold[threshold_index]["transform"], dtype=np.float64)
        second = np.asarray(toothseg[toothseg_index]["transform"], dtype=np.float64)
        if int(np.sign(np.linalg.det(first[:3, :3]))) != int(
            np.sign(np.linalg.det(second[:3, :3]))
        ):
            continue
        disagreement = disagreement_mm(first, second, anchors)
        threshold_rank = threshold_index / max(len(threshold) - 1, 1)
        toothseg_rank = toothseg_index / max(len(toothseg) - 1, 1)
        pairs.append(
            (
                disagreement + rank_weight * (threshold_rank + toothseg_rank),
                disagreement,
                threshold[threshold_index],
                toothseg[toothseg_index],
                center,
            )
        )
    if not pairs:
        raise RuntimeError("No chirality-consistent cross-modal candidate pair")
    return min(pairs, key=lambda item: (item[0], item[1]))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_groups = load_candidate_groups(args.threshold_runs)
    toothseg_groups = load_candidate_groups(args.toothseg_runs)
    keys = sorted(set(threshold_groups) & set(toothseg_groups))
    records = {
        (record.case_id, record.jaw): record
        for record in load_manifest(args.manifest)
        if record.complete and record.transform_path
    }
    exact = load_exact_errors(args.exact_loo)
    source_cache = {}
    ground_truth = {}
    for index, key in enumerate(keys):
        source_cache[key], _ = load_ios_points(
            Path(records[key].ios_path), args.sample_points, args.seed + index
        )
        ground_truth[key] = np.load(records[key].transform_path, allow_pickle=False)

    grid_rows = []
    agreement_grid = []
    rows_by_config = {}
    for top_k, rank_weight in itertools.product(args.top_k, args.rank_weights):
        base_rows = []
        for key in keys:
            _, disagreement, threshold, toothseg, center = select_pair(
                threshold_groups[key],
                toothseg_groups[key],
                key[1],
                top_k,
                rank_weight,
                args.anchor_radius_mm,
                args.exclude_upper_opposite_axial,
            )
            midpoint = midpoint_transform(
                np.asarray(threshold["transform"], dtype=np.float64),
                np.asarray(toothseg["transform"], dtype=np.float64),
                center,
            )
            midpoint_error = float(
                evaluate_transform(source_cache[key], midpoint, ground_truth[key])["mean_tre_mm"]
            )
            base_rows.append(
                {
                    "case_id": key[0],
                    "jaw": key[1],
                    "cross_modal_disagreement_mm": disagreement,
                    "threshold_mean_tre_mm": float(threshold["mean_tre_mm"]),
                    "toothseg_mean_tre_mm": float(toothseg["mean_tre_mm"]),
                    "midpoint_mean_tre_mm": midpoint_error,
                    "threshold_candidate_run": threshold.get("candidate_run", ""),
                    "toothseg_candidate_run": toothseg.get("candidate_run", ""),
                    "threshold_transform": np.asarray(threshold["transform"]).tolist(),
                    "toothseg_transform": np.asarray(toothseg["transform"]).tolist(),
                    "midpoint_transform": midpoint.tolist(),
                }
            )
        for selection in args.selection:
            rows = []
            for row in base_rows:
                selected = dict(row)
                selected["selection"] = selection
                selected["mean_tre_mm"] = float(row[f"{selection}_mean_tre_mm"])
                selected["transform"] = json.dumps(row[f"{selection}_transform"])
                rows.append(selected)
            raw = summarize([float(row["mean_tre_mm"]) for row in rows])
            combined = summarize(exact_fallback_errors(rows, exact))
            config = (top_k, rank_weight, selection)
            rows_by_config[config] = rows
            grid_rows.append(
                {
                    "top_k": top_k,
                    "rank_weight_mm": rank_weight,
                    "selection": selection,
                    **raw,
                    "exact_fallback_mean_tre_mm": combined["mean_tre_mm"],
                    "exact_fallback_median_tre_mm": combined["median_tre_mm"],
                    "exact_fallback_p90_tre_mm": combined["p90_tre_mm"],
                    "exact_fallback_max_tre_mm": combined["max_tre_mm"],
                }
            )
            for threshold in args.agreement_thresholds:
                accepted = [
                    row
                    for row in rows
                    if float(row["cross_modal_disagreement_mm"]) <= threshold
                ]
                accepted_summary = (
                    summarize([float(row["mean_tre_mm"]) for row in accepted])
                    if accepted
                    else {
                        "mean_tre_mm": None,
                        "median_tre_mm": None,
                        "p90_tre_mm": None,
                        "max_tre_mm": None,
                    }
                )
                agreement_grid.append(
                    {
                        "top_k": top_k,
                        "rank_weight_mm": rank_weight,
                        "selection": selection,
                        "threshold_mm": threshold,
                        "accepted_jaws": len(accepted),
                        "coverage": len(accepted) / max(len(rows), 1),
                        **accepted_summary,
                    }
                )

    grid_rows.sort(
        key=lambda row: (
            float(row["exact_fallback_mean_tre_mm"]),
            float(row["exact_fallback_p90_tre_mm"]),
            float(row["exact_fallback_max_tre_mm"]),
        )
    )
    best = grid_rows[0]
    best_config = (
        int(best["top_k"]), float(best["rank_weight_mm"]), str(best["selection"])
    )
    best_rows = rows_by_config[best_config]
    agreement = []
    for threshold in args.agreement_thresholds:
        accepted = [
            row
            for row in best_rows
            if float(row["cross_modal_disagreement_mm"]) <= threshold
        ]
        agreement.append(
            {
                "threshold_mm": threshold,
                "accepted_jaws": len(accepted),
                "coverage": len(accepted) / max(len(best_rows), 1),
                "mean_tre_mm": (
                    float(np.mean([float(row["mean_tre_mm"]) for row in accepted]))
                    if accepted
                    else None
                ),
            }
        )
    write_csv(args.output_dir / "grid.csv", grid_rows)
    write_csv(args.output_dir / "best_oof.csv", best_rows)
    write_csv(args.output_dir / "agreement_gate.csv", agreement)
    write_csv(args.output_dir / "agreement_grid.csv", agreement_grid)
    eligible_gates = [
        row
        for row in agreement_grid
        if int(row["accepted_jaws"]) >= args.min_gate_jaws
        and row["mean_tre_mm"] is not None
    ]
    best_gate = (
        min(
            eligible_gates,
            key=lambda row: (
                float(row["mean_tre_mm"]),
                float(row["p90_tre_mm"]),
                -int(row["accepted_jaws"]),
            ),
        )
        if eligible_gates
        else None
    )
    summary = {
        "jaws": len(keys),
        "best": best,
        "agreement_gate": agreement,
        "best_pseudo_gate": best_gate,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
