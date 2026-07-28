from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.data import apply_transform, load_ios_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select registration pseudo-labels that agree across independent geometry runs."
    )
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-disagreement-mm", type=float, default=2.0)
    parser.add_argument("--max-rank-score-mm", type=float, default=2.0)
    parser.add_argument("--max-source-score-mm", type=float, default=0.6)
    parser.add_argument("--max-source-p90-mm", type=float, default=2.0)
    parser.add_argument("--min-target-coverage-2mm", type=float, default=0.15)
    parser.add_argument("--min-consensus-runs", type=int, default=2)
    parser.add_argument(
        "--required-consensus-runs",
        type=int,
        nargs="*",
        default=(),
        help="Zero-based teacher indices that must belong to an accepted consensus cluster.",
    )
    parser.add_argument("--sample-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _load_results(run_dir: Path) -> dict[tuple[str, str], dict]:
    results = {}
    for path in sorted(run_dir.glob("*_*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = payload["record"]
        key = (str(record["case_id"]), str(record["jaw"]))
        results[key] = payload
    return results


def _transform(payload: dict) -> np.ndarray:
    return np.asarray(payload["registration"]["transform"], dtype=np.float64)


def _consensus_transform(transforms: list[np.ndarray], points: np.ndarray) -> tuple[np.ndarray, int]:
    moved = [apply_transform(points, transform) for transform in transforms]
    costs = []
    for index, candidate in enumerate(moved):
        costs.append(sum(float(np.mean(np.linalg.norm(candidate - other, axis=1))) for other in moved))
    selected = int(np.argmin(costs))
    return transforms[selected], selected


def _largest_consistent_cluster(
    transforms: list[np.ndarray],
    points: np.ndarray,
    max_disagreement_mm: float,
    min_size: int,
) -> tuple[list[int], np.ndarray]:
    if min_size < 2 or min_size > len(transforms):
        raise ValueError("min_size must be between 2 and the number of transforms")
    moved = [apply_transform(points, transform) for transform in transforms]
    distances = np.zeros((len(transforms), len(transforms)), dtype=np.float64)
    for first, second in combinations(range(len(transforms)), 2):
        distance = float(np.mean(np.linalg.norm(moved[first] - moved[second], axis=1)))
        distances[first, second] = distances[second, first] = distance
    determinants = [int(np.sign(np.linalg.det(transform[:3, :3]))) for transform in transforms]
    candidates: list[tuple[int, float, tuple[int, ...]]] = []
    for size in range(len(transforms), min_size - 1, -1):
        for indices in combinations(range(len(transforms)), size):
            if len({determinants[index] for index in indices}) != 1:
                continue
            pairwise = [distances[a, b] for a, b in combinations(indices, 2)]
            if pairwise and max(pairwise) <= max_disagreement_mm:
                candidates.append((size, float(np.mean(pairwise)), indices))
        if candidates:
            break
    if not candidates:
        return [], distances
    _, _, best = min(candidates, key=lambda item: (-item[0], item[1], item[2]))
    return list(best), distances


def main() -> None:
    args = parse_args()
    run_results = [_load_results(path.resolve()) for path in args.runs]
    if len(run_results) < 2:
        raise ValueError("At least two independent runs are required")
    if args.min_consensus_runs > len(run_results):
        raise ValueError("--min-consensus-runs exceeds the number of supplied runs")
    if any(index < 0 or index >= len(run_results) for index in args.required_consensus_runs):
        raise ValueError("--required-consensus-runs contains an invalid teacher index")
    common_keys = set.intersection(*(set(results) for results in run_results))
    if not common_keys:
        raise RuntimeError("The supplied runs have no common case/jaw results")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    accepted_payloads = []
    consensus_payloads = []
    for key in sorted(common_keys):
        payloads = [results[key] for results in run_results]
        record = payloads[0]["record"]
        points, _ = load_ios_points(Path(record["ios_path"]), args.sample_points, args.seed)
        transforms = [_transform(payload) for payload in payloads]
        rank_scores = [
            float(payload["selection"].get("rank_score", payload["selection"]["score"]))
            for payload in payloads
        ]
        source_scores = [float(payload["registration"]["score"]) for payload in payloads]
        source_p90 = [float(payload["registration"]["p90_distance"]) for payload in payloads]
        coverage = [float(payload["selection"].get("target_coverage_2mm", 0.0)) for payload in payloads]
        quality_indices = [
            index
            for index in range(len(payloads))
            if rank_scores[index] <= args.max_rank_score_mm
            and source_scores[index] <= args.max_source_score_mm
            and source_p90[index] <= args.max_source_p90_mm
            and coverage[index] >= args.min_target_coverage_2mm
        ]
        local_cluster: list[int] = []
        local_distances = np.zeros((len(quality_indices), len(quality_indices)), dtype=np.float64)
        if len(quality_indices) >= args.min_consensus_runs:
            local_cluster, local_distances = _largest_consistent_cluster(
                [transforms[index] for index in quality_indices],
                points,
                args.max_disagreement_mm,
                args.min_consensus_runs,
            )
        cluster = [quality_indices[index] for index in local_cluster]
        local_lookup = {original: local for local, original in enumerate(quality_indices)}
        disagreements = [
            float(local_distances[local_lookup[first], local_lookup[second]])
            for first, second in combinations(cluster, 2)
        ]
        determinants = [int(np.sign(np.linalg.det(transform[:3, :3]))) for transform in transforms]
        if cluster:
            consensus, local_selected_index = _consensus_transform(
                [transforms[index] for index in cluster], points
            )
            selected_index = cluster[local_selected_index]
            cluster_ranks = [rank_scores[index] for index in cluster]
            cluster_source_scores = [source_scores[index] for index in cluster]
            cluster_source_p90 = [source_p90[index] for index in cluster]
            cluster_coverage = [coverage[index] for index in cluster]
        else:
            consensus = transforms[0]
            selected_index = 0
            cluster_ranks = rank_scores
            cluster_source_scores = source_scores
            cluster_source_p90 = source_p90
            cluster_coverage = coverage
        mean_disagreement = float(np.mean(disagreements)) if disagreements else float("inf")
        max_disagreement = float(np.max(disagreements)) if disagreements else float("inf")
        determinant_agreement = bool(cluster)
        required_runs_present = all(index in cluster for index in args.required_consensus_runs)
        accepted = (
            bool(cluster)
            and required_runs_present
            and max(cluster_ranks) <= args.max_rank_score_mm
            and max(cluster_source_scores) <= args.max_source_score_mm
            and max(cluster_source_p90) <= args.max_source_p90_mm
            and min(cluster_coverage) >= args.min_target_coverage_2mm
        )
        if accepted:
            repeatability_confidence = np.exp(-max_disagreement / args.max_disagreement_mm)
            rank_confidence = np.exp(-max(cluster_ranks) / args.max_rank_score_mm)
            source_confidence = np.exp(-max(cluster_source_scores) / args.max_source_score_mm)
            p90_confidence = np.exp(-max(cluster_source_p90) / args.max_source_p90_mm)
            coverage_confidence = min(1.0, min(cluster_coverage) / 0.6)
            confidence = float(
                (
                    repeatability_confidence
                    * rank_confidence
                    * source_confidence
                    * p90_confidence
                    * coverage_confidence
                )
                ** 0.2
            )
        else:
            confidence = 0.0
        row = {
            "case_id": key[0],
            "jaw": key[1],
            "accepted": int(accepted),
            "repeat_count": len(payloads),
            "quality_repeat_count": len(quality_indices),
            "quality_repeats": " ".join(str(index) for index in quality_indices),
            "consensus_count": len(cluster),
            "consensus_repeats": " ".join(str(index) for index in cluster),
            "outlier_count": len(payloads) - len(cluster),
            "determinant": determinants[selected_index],
            "determinant_agreement": int(determinant_agreement),
            "required_runs_present": int(required_runs_present),
            "mean_disagreement_mm": mean_disagreement,
            "max_disagreement_mm": max_disagreement,
            "mean_rank_score_mm": float(np.mean(cluster_ranks)),
            "max_rank_score_mm": max(cluster_ranks),
            "max_source_score_mm": max(cluster_source_scores),
            "max_source_p90_mm": max(cluster_source_p90),
            "min_target_coverage_2mm": min(cluster_coverage),
            "confidence": confidence,
            "selected_repeat": selected_index,
        }
        rows.append(row)
        if cluster:
            consensus_payloads.append(
                {
                    **row,
                    "ios_path": record["ios_path"],
                    "cbct_path": record["cbct_path"],
                    "transform": consensus.tolist(),
                    "consensus_repeat_indices": cluster,
                }
            )
        if accepted:
            accepted_payloads.append(
                {
                    **row,
                    "ios_path": record["ios_path"],
                    "cbct_path": record["cbct_path"],
                    "transform": consensus.tolist(),
                    "repeat_transforms": [transforms[index].tolist() for index in cluster],
                    "all_repeat_transforms": [transform.tolist() for transform in transforms],
                    "consensus_repeat_indices": cluster,
                }
            )
        print(
            f"{key[0]} {key[1]}: {'ACCEPT' if accepted else 'reject'} | "
            f"cluster={len(cluster)}/{len(transforms)} max={max_disagreement:.3f} mm | "
            f"rank={max(cluster_ranks):.3f} source={max(cluster_source_scores):.3f}/"
            f"p90={max(cluster_source_p90):.3f} mm | coverage={min(cluster_coverage):.3f}"
        )

    with (args.output_dir / "consistency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "pseudo_labels.json").write_text(
        json.dumps(accepted_payloads, indent=2), encoding="utf-8"
    )
    (args.output_dir / "all_consensus.json").write_text(
        json.dumps(consensus_payloads, indent=2), encoding="utf-8"
    )
    print(f"Accepted {len(accepted_payloads)} / {len(rows)} case-jaws")


if __name__ == "__main__":
    main()
