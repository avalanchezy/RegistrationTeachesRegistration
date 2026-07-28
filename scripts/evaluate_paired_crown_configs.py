from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired case-level audit of two crown postprocess configurations."
    )
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--left-threshold", type=float, required=True)
    parser.add_argument("--right-threshold", type=float, required=True)
    parser.add_argument("--left-minimum-component-voxels", type=int, default=4)
    parser.add_argument("--right-minimum-component-voxels", type=int, default=4)
    parser.add_argument("--left-maximum-components", type=int, default=0)
    parser.add_argument("--right-maximum-components", type=int, default=0)
    parser.add_argument("--left-minimum-hu", type=float, default=-1000.0)
    parser.add_argument("--right-minimum-hu", type=float, default=-1000.0)
    parser.add_argument("--metric", default="symmetric_chamfer_mm")
    parser.add_argument("--bootstrap-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_case_values(
    path: Path,
    metric: str,
    threshold: float,
    minimum_component_voxels: int,
    maximum_components: int,
    minimum_hu: float,
) -> dict[str, float]:
    grouped: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not np.isclose(float(row["threshold"]), threshold):
                continue
            if int(row["minimum_component_voxels"]) != minimum_component_voxels:
                continue
            if int(row["maximum_components"]) != maximum_components:
                continue
            if not np.isclose(float(row["minimum_hu"]), minimum_hu):
                continue
            grouped.setdefault(str(row["case_id"]), {})[str(row["jaw"])] = float(
                row[metric]
            )
    invalid = {case_id: sorted(values) for case_id, values in grouped.items() if set(values) != {"upper", "lower"}}
    if invalid:
        raise RuntimeError(f"Incomplete jaw pairs under {path}: {invalid}")
    if not grouped:
        raise RuntimeError(f"No matching rows under {path}")
    return {
        case_id: float(np.mean([values["upper"], values["lower"]]))
        for case_id, values in grouped.items()
    }


def robust_selection_diagnostics(
    delta: np.ndarray,
    case_ids: list[str],
    bootstrap_95_ci: tuple[float, float],
    candidate_label: str,
    reference_label: str,
) -> dict[str, object]:
    """Require a candidate gain to survive distribution and influence checks."""
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 1 or delta.size != len(case_ids) or delta.size < 2:
        raise ValueError("Robust paired selection requires at least two case deltas")

    mean_delta = float(np.mean(delta))
    median_delta = float(np.median(delta))
    left_wins = int(np.sum(delta < 0.0))
    right_wins = int(np.sum(delta > 0.0))
    leave_one_out = (float(np.sum(delta)) - delta) / float(delta.size - 1)
    removal_effect = np.abs(leave_one_out - mean_delta)
    influential_index = int(np.argmax(removal_effect))

    criteria = {
        "candidate_mean_is_better": mean_delta < 0.0,
        "bootstrap_interval_excludes_zero": bootstrap_95_ci[1] < 0.0,
        "candidate_median_is_better": median_delta < 0.0,
        "candidate_wins_more_cases": left_wins > right_wins,
        "candidate_gain_survives_every_leave_one_case_out_check": bool(
            np.max(leave_one_out) < 0.0
        ),
    }
    candidate_selected = all(criteria.values())
    net_gain = -float(np.sum(delta))
    largest_favorable_delta = max(0.0, -float(np.min(delta)))
    favorable_share = (
        largest_favorable_delta / net_gain if net_gain > 0.0 else None
    )
    return {
        "policy": (
            "candidate replaces the reference only when its paired gain has a "
            "negative mean and median, a bootstrap interval below zero, a case "
            "win majority, and remains negative after removing any one case"
        ),
        "candidate_label": candidate_label,
        "reference_label": reference_label,
        "criteria": criteria,
        "recommended_label": candidate_label if candidate_selected else reference_label,
        "candidate_selected": candidate_selected,
        "leave_one_case_out_mean_delta_range": [
            float(np.min(leave_one_out)),
            float(np.max(leave_one_out)),
        ],
        "most_influential_case": {
            "case_id": case_ids[influential_index],
            "delta_candidate_minus_reference": float(delta[influential_index]),
            "mean_delta_without_case": float(leave_one_out[influential_index]),
            "absolute_change_in_mean_when_removed": float(
                removal_effect[influential_index]
            ),
        },
        "largest_favorable_case_share_of_net_gain": favorable_share,
    }


def main() -> None:
    args = parse_args()
    left = load_case_values(
        args.left,
        args.metric,
        args.left_threshold,
        args.left_minimum_component_voxels,
        args.left_maximum_components,
        args.left_minimum_hu,
    )
    right = load_case_values(
        args.right,
        args.metric,
        args.right_threshold,
        args.right_minimum_component_voxels,
        args.right_maximum_components,
        args.right_minimum_hu,
    )
    if set(left) != set(right):
        raise RuntimeError(
            f"Case sets differ: left-only={sorted(set(left) - set(right))}, "
            f"right-only={sorted(set(right) - set(left))}"
        )
    case_ids = sorted(left)
    left_values = np.asarray([left[case_id] for case_id in case_ids], dtype=np.float64)
    right_values = np.asarray([right[case_id] for case_id in case_ids], dtype=np.float64)
    delta = left_values - right_values
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(
        0, len(case_ids), size=(args.bootstrap_samples, len(case_ids))
    )
    bootstrap = np.mean(delta[indices], axis=1)
    bootstrap_95_ci = tuple(
        float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
    )
    payload = {
        "metric": args.metric,
        "cases": len(case_ids),
        "left": {
            "label": args.left_label,
            "mean": float(np.mean(left_values)),
        },
        "right": {
            "label": args.right_label,
            "mean": float(np.mean(right_values)),
        },
        "delta_left_minus_right": {
            "mean": float(np.mean(delta)),
            "median": float(np.median(delta)),
            "bootstrap_95_ci": list(bootstrap_95_ci),
            "wilcoxon_p": float(wilcoxon(delta).pvalue),
            "left_better_cases": int(np.sum(delta < 0.0)),
            "right_better_cases": int(np.sum(delta > 0.0)),
            "ties": int(np.sum(delta == 0.0)),
        },
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "robust_selection": robust_selection_diagnostics(
            delta,
            case_ids,
            bootstrap_95_ci,
            args.left_label,
            args.right_label,
        ),
        "per_case": [
            {
                "case_id": case_id,
                "left": float(left[case_id]),
                "right": float(right[case_id]),
                "delta_left_minus_right": float(left[case_id] - right[case_id]),
            }
            for case_id in case_ids
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
