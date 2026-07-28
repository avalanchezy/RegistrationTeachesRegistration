from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from evaluate_nested_crown_fusion_selection import cbct_groups
except ModuleNotFoundError:  # Imported as scripts.evaluate_nested_global_mode_selection.
    from scripts.evaluate_nested_crown_fusion_selection import cbct_groups


MODE_COMPLEXITY = {
    "fixed": 0,
    "raw": 0,
    "tuned": 1,
    "incumbent": 0,
    "crown": 1,
    "direct": 1,
    "direct_probability": 2,
    "direct_guided": 2,
    "direct_guided_fine": 2,
    "direct_guided_high": 2,
    "direct_guided_all": 4,
    "crown_toothseg": 2,
    "all_unrefined": 5,
    "all_refined": 6,
}


def mode_complexity(name: str) -> int:
    if name in MODE_COMPLEXITY:
        return MODE_COMPLEXITY[name]
    if name.startswith("tuned_"):
        return 1
    raise ValueError(f"Unknown global mode: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recommend a global crown mode by leaving CBCT payload groups out "
            "of the cross-fitted OOF comparison."
        )
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        required=True,
        help="Mode/path pairs formatted as name=best_joint_oof.csv.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cbct-hash-cache", type=Path, required=True)
    parser.add_argument("--exact-loo", type=Path)
    parser.add_argument("--baseline-mode", default="direct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric", default="mean_tre_mm")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def load_mode(specification: str, metric: str) -> tuple[str, dict[tuple[str, str], float]]:
    if "=" not in specification:
        raise ValueError(f"Invalid mode specification: {specification}")
    name, raw_path = specification.split("=", 1)
    path = Path(raw_path)
    values: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["case_id"]), str(row["jaw"]))
            if key in values:
                raise ValueError(f"Duplicate row for {name} {key}")
            values[key] = float(row[metric])
    if not values:
        raise RuntimeError(f"No rows were loaded for mode {name} from {path}")
    return name, values


def load_exact_errors(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            (str(row["case_id"]), str(row["jaw"])): float(row["mean_tre_mm"])
            for row in csv.DictReader(handle)
        }


def apply_exact_fallback(
    modes: dict[str, dict[tuple[str, str], float]],
    exact: dict[tuple[str, str], float],
) -> int:
    replaced = 0
    for key, value in exact.items():
        present = [key in values for values in modes.values()]
        if any(present) and not all(present):
            raise RuntimeError(f"Exact-fallback key is incomplete across modes: {key}")
        if all(present):
            for values in modes.values():
                values[key] = value
            replaced += 1
    return replaced


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_mm": float(np.mean(array)),
        "median_mm": float(np.median(array)),
        "p90_mm": float(np.quantile(array, 0.9)),
        "maximum_mm": float(np.max(array)),
    }


def robust_mode_decision(
    group_deltas: np.ndarray,
    bootstrap_ci95: tuple[float, float],
    candidate_mode: str,
    baseline_mode: str = "direct",
) -> dict[str, object]:
    deltas = np.asarray(group_deltas, dtype=np.float64)
    if deltas.ndim != 1 or deltas.size < 2:
        raise ValueError("Robust global mode selection requires at least two groups")
    leave_one_out = (float(np.sum(deltas)) - deltas) / float(deltas.size - 1)
    candidate_wins = int(np.sum(deltas < 0.0))
    baseline_wins = int(np.sum(deltas > 0.0))
    criteria = {
        "candidate_mean_is_better": float(np.mean(deltas)) < 0.0,
        "bootstrap_interval_excludes_zero": bootstrap_ci95[1] < 0.0,
        "candidate_median_is_better": float(np.median(deltas)) < 0.0,
        "candidate_wins_more_cbct_groups": candidate_wins > baseline_wins,
        "candidate_gain_survives_every_leave_one_group_out_check": bool(
            np.max(leave_one_out) < 0.0
        ),
    }
    candidate_selected = candidate_mode == baseline_mode or all(criteria.values())
    influential_index = int(
        np.argmax(np.abs(leave_one_out - float(np.mean(deltas))))
    )
    return {
        "policy": (
            "an additional global mode replaces the baseline only when its CBCT-group "
            "delta has a negative mean and median, a bootstrap interval below "
            "zero, a group win majority, and remains negative after removing "
            "any one CBCT group"
        ),
        "candidate_mode": candidate_mode,
        "baseline_mode": baseline_mode,
        "criteria": criteria,
        "candidate_selected": candidate_selected,
        "recommended_mode": candidate_mode if candidate_selected else baseline_mode,
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "ties": int(np.sum(deltas == 0.0)),
        "leave_one_group_out_mean_delta_range_mm": [
            float(np.min(leave_one_out)),
            float(np.max(leave_one_out)),
        ],
        "most_influential_group_index": influential_index,
        "most_influential_group_delta_mm": float(deltas[influential_index]),
    }


def main() -> None:
    args = parse_args()
    loaded = dict(load_mode(specification, args.metric) for specification in args.modes)
    unknown = sorted(
        name
        for name in loaded
        if name not in MODE_COMPLEXITY and not name.startswith("tuned_")
    )
    if unknown:
        raise ValueError(f"Unknown global modes: {unknown}")
    if args.baseline_mode not in loaded:
        raise ValueError(f"Baseline mode is unavailable: {args.baseline_mode}")
    exact = load_exact_errors(args.exact_loo)
    exact_fallback_rows = apply_exact_fallback(loaded, exact)
    key_sets = [set(values) for values in loaded.values()]
    common_keys = set.intersection(*key_sets)
    if any(keys != common_keys for keys in key_sets):
        raise RuntimeError("Global modes do not contain identical case/jaw rows")
    cases = sorted({case_id for case_id, _ in common_keys})
    groups = cbct_groups(args.manifest, args.cbct_hash_cache, cases)

    selection_counts: Counter[str] = Counter()
    records: list[dict[str, object]] = []
    heldout_values: list[float] = []
    for payload_hash, heldout_cases in groups.items():
        training_keys = [key for key in common_keys if key[0] not in heldout_cases]
        training_means = {
            name: float(np.mean([values[key] for key in training_keys]))
            for name, values in loaded.items()
        }
        selected = min(
            loaded,
            key=lambda name: (
                training_means[name],
                mode_complexity(name),
                name,
            ),
        )
        selection_counts[selected] += 1
        test_keys = sorted(key for key in common_keys if key[0] in heldout_cases)
        selected_values = [loaded[selected][key] for key in test_keys]
        heldout_values.extend(selected_values)
        records.append(
            {
                "cbct_sha256": payload_hash,
                "heldout_cases": heldout_cases,
                "heldout_jaws": len(test_keys),
                "selected_mode": selected,
                "training_mean_tre_mm": training_means[selected],
                "heldout_mean_tre_mm": float(np.mean(selected_values)),
                "training_mode_means_mm": training_means,
            }
        )

    fixed = {
        name: summarize([values[key] for key in sorted(common_keys)])
        for name, values in loaded.items()
    }
    recommended = min(
        loaded,
        key=lambda name: (
            -selection_counts[name],
            fixed[name]["mean_mm"],
            mode_complexity(name),
            name,
        ),
    )
    full_oof_best = min(
        loaded,
        key=lambda name: (
            fixed[name]["mean_mm"],
            mode_complexity(name),
            name,
        ),
    )

    baseline = args.baseline_mode
    group_deltas = []
    for heldout_cases in groups.values():
        keys = [key for key in common_keys if key[0] in heldout_cases]
        group_deltas.append(
            float(
                np.mean(
                    [loaded[recommended][key] - loaded[baseline][key] for key in keys]
                )
            )
        )
    rng = np.random.default_rng(args.seed)
    delta_array = np.asarray(group_deltas, dtype=np.float64)
    indices = rng.integers(
        0, len(delta_array), size=(args.bootstrap_samples, len(delta_array))
    )
    bootstrap = np.mean(delta_array[indices], axis=1)
    bootstrap_ci95 = (
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    )
    comparison_to_baseline = {
        "baseline_mode": baseline,
        "mean_delta_mm": float(np.mean(delta_array)),
        "median_delta_mm": float(np.median(delta_array)),
        "bootstrap_ci95_mm": list(bootstrap_ci95),
        "bootstrap_probability_better": float(np.mean(bootstrap < 0.0)),
    }
    robust_selection = robust_mode_decision(
        delta_array,
        bootstrap_ci95,
        recommended,
        baseline,
    )
    deployed_mode = str(robust_selection["recommended_mode"])

    summary = {
        "protocol": (
            "leave-one-CBCT-payload-group-out recommendation over "
            "cross-fitted per-mode OOF predictions"
        ),
        "metric": args.metric,
        "cases": len(cases),
        "jaws": len(common_keys),
        "cbct_groups": len(groups),
        "best": {
            "mode": deployed_mode,
            "nested_candidate_mode": recommended,
            "nested_selection_count": selection_counts[recommended],
            "nested_selection_fraction": selection_counts[recommended] / len(groups),
            "fixed_oof": fixed[deployed_mode],
            "comparison_to_baseline": comparison_to_baseline,
            "robust_selection": robust_selection,
        },
        "full_oof_best_mode": full_oof_best,
        "nested_mixture": summarize(heldout_values),
        "selection_counts": dict(selection_counts.most_common()),
        "fixed_modes": fixed,
        "exact_fallback": {
            "path": str(args.exact_loo) if args.exact_loo is not None else None,
            "replaced_jaw_rows": exact_fallback_rows,
        },
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
