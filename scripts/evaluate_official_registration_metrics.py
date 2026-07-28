from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_nested_crown_fusion_selection import cbct_groups
from scripts.evaluate_nested_global_mode_selection import mode_complexity
from task2reg.data import load_manifest
from task2reg.metrics import rotation_error_deg


MetricPair = tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cached OOF transforms with the official STSR Task 2 "
            "translation and rotation metrics."
        )
    )
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cbct-hash-cache", type=Path, required=True)
    parser.add_argument("--exact-loo", type=Path)
    parser.add_argument("--baseline-mode", default="direct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def parse_mode(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(f"Invalid mode specification: {specification}")
    name, raw_path = specification.split("=", 1)
    mode_complexity(name)
    return name, Path(raw_path)


def parse_transform_key(value: str) -> np.ndarray:
    values = np.fromstring(value, sep=",", dtype=np.float64)
    if values.shape != (16,) or not np.isfinite(values).all():
        raise ValueError("transform_key must contain 16 finite comma-separated values")
    return values.reshape(4, 4)


def official_metrics(transform: np.ndarray, ground_truth: np.ndarray) -> MetricPair:
    translation = float(np.linalg.norm(transform[:3, 3] - ground_truth[:3, 3]))
    rotation = rotation_error_deg(transform, ground_truth)
    return translation, rotation


def load_ground_truth(path: Path) -> dict[tuple[str, str], np.ndarray]:
    records = load_manifest(path)
    output = {}
    for record in records:
        if record.split != "Train-Labeled" or not record.transform_path:
            continue
        output[(record.case_id, record.jaw)] = np.load(
            record.transform_path, allow_pickle=False
        ).astype(np.float64)
    return output


def load_mode_metrics(
    path: Path,
    ground_truth: dict[tuple[str, str], np.ndarray],
) -> dict[tuple[str, str], MetricPair]:
    output = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["case_id"]), str(row["jaw"]))
            if key in output:
                raise ValueError(f"Duplicate OOF transform for {key} in {path}")
            if key not in ground_truth:
                raise ValueError(f"Missing ground truth for {key}")
            output[key] = official_metrics(
                parse_transform_key(row["transform_key"]), ground_truth[key]
            )
    if not output:
        raise RuntimeError(f"No OOF transforms found in {path}")
    return output


def load_exact_metrics(path: Path | None) -> dict[tuple[str, str], MetricPair]:
    if path is None:
        return {}
    output = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("translation_error_mm") or not row.get("rotation_error_deg"):
                raise ValueError(
                    "Exact LOO CSV must contain translation_error_mm and rotation_error_deg"
                )
            key = (str(row["case_id"]), str(row["jaw"]))
            output[key] = (
                float(row["translation_error_mm"]),
                float(row["rotation_error_deg"]),
            )
    return output


def summarize(values: dict[tuple[str, str], MetricPair]) -> dict[str, float]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    return {
        "mean_translation_error_mm": float(np.mean(array[:, 0])),
        "median_translation_error_mm": float(np.median(array[:, 0])),
        "p90_translation_error_mm": float(np.quantile(array[:, 0], 0.9)),
        "max_translation_error_mm": float(np.max(array[:, 0])),
        "mean_rotation_error_deg": float(np.mean(array[:, 1])),
        "median_rotation_error_deg": float(np.median(array[:, 1])),
        "p90_rotation_error_deg": float(np.quantile(array[:, 1], 0.9)),
        "max_rotation_error_deg": float(np.max(array[:, 1])),
    }


def rank_modes(summaries: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    names = sorted(summaries)
    translation = np.asarray(
        [summaries[name]["mean_translation_error_mm"] for name in names]
    )
    rotation = np.asarray(
        [summaries[name]["mean_rotation_error_deg"] for name in names]
    )
    translation_ranks = rankdata(translation, method="min")
    rotation_ranks = rankdata(rotation, method="min")
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "mode": name,
                **summaries[name],
                "translation_rank": int(translation_ranks[index]),
                "rotation_rank": int(rotation_ranks[index]),
                "official_average_rank": float(
                    (translation_ranks[index] + rotation_ranks[index]) / 2.0
                ),
                "mode_complexity": mode_complexity(name),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            float(row["official_average_rank"]),
            int(row["rotation_rank"]),
            int(row["translation_rank"]),
            float(row["mean_rotation_error_deg"]),
            float(row["mean_translation_error_mm"]),
            int(row["mode_complexity"]),
            str(row["mode"]),
        ),
    )


def subset_summary(
    values: dict[tuple[str, str], MetricPair], keys: list[tuple[str, str]]
) -> dict[str, float]:
    return summarize({key: values[key] for key in keys})


def group_means(
    values: dict[tuple[str, str], MetricPair],
    groups: dict[str, list[str]],
) -> np.ndarray:
    rows = []
    for cases in groups.values():
        keys = [key for key in values if key[0] in cases]
        rows.append(np.mean(np.asarray([values[key] for key in keys]), axis=0))
    return np.asarray(rows, dtype=np.float64)


def bootstrap_rank_comparison(
    values: dict[str, dict[tuple[str, str], MetricPair]],
    groups: dict[str, list[str]],
    candidate: str,
    baseline: str,
    samples: int,
    seed: int,
) -> dict[str, object]:
    names = sorted(values)
    grouped = np.stack([group_means(values[name], groups) for name in names])
    candidate_index = names.index(candidate)
    baseline_index = names.index(baseline)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, grouped.shape[1], size=(samples, grouped.shape[1]))
    rank_delta = np.empty(samples, dtype=np.float64)
    translation_delta = np.empty(samples, dtype=np.float64)
    rotation_delta = np.empty(samples, dtype=np.float64)
    for index, group_indices in enumerate(sampled):
        means = np.mean(grouped[:, group_indices, :], axis=1)
        translation_ranks = rankdata(means[:, 0], method="min")
        rotation_ranks = rankdata(means[:, 1], method="min")
        average_ranks = (translation_ranks + rotation_ranks) / 2.0
        rank_delta[index] = average_ranks[candidate_index] - average_ranks[baseline_index]
        translation_delta[index] = means[candidate_index, 0] - means[baseline_index, 0]
        rotation_delta[index] = means[candidate_index, 1] - means[baseline_index, 1]
    return {
        "candidate_mode": candidate,
        "baseline_mode": baseline,
        "probability_better_official_rank": float(np.mean(rank_delta < 0.0)),
        "probability_tied_official_rank": float(np.mean(rank_delta == 0.0)),
        "probability_worse_official_rank": float(np.mean(rank_delta > 0.0)),
        "translation_delta_ci95_mm": [
            float(np.quantile(translation_delta, 0.025)),
            float(np.quantile(translation_delta, 0.975)),
        ],
        "rotation_delta_ci95_deg": [
            float(np.quantile(rotation_delta, 0.025)),
            float(np.quantile(rotation_delta, 0.975)),
        ],
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    mode_paths = dict(parse_mode(specification) for specification in args.modes)
    if args.baseline_mode not in mode_paths:
        raise ValueError(f"Baseline mode is unavailable: {args.baseline_mode}")
    ground_truth = load_ground_truth(args.manifest)
    raw = {
        name: load_mode_metrics(path, ground_truth) for name, path in mode_paths.items()
    }
    key_sets = [set(values) for values in raw.values()]
    common_keys = set.intersection(*key_sets)
    if any(keys != common_keys for keys in key_sets):
        raise RuntimeError("Official metric modes do not contain identical jaw rows")
    exact = load_exact_metrics(args.exact_loo)
    unknown_exact = set(exact) - common_keys
    if unknown_exact:
        raise RuntimeError(f"Exact LOO contains unknown rows: {sorted(unknown_exact)}")
    deployed = {
        name: {key: exact.get(key, metric) for key, metric in values.items()}
        for name, values in raw.items()
    }
    raw_summaries = {name: summarize(values) for name, values in raw.items()}
    deployed_summaries = {name: summarize(values) for name, values in deployed.items()}
    fixed_ranking = rank_modes(deployed_summaries)

    cases = sorted({case_id for case_id, _ in common_keys})
    groups = cbct_groups(args.manifest, args.cbct_hash_cache, cases)
    selection_counts: Counter[str] = Counter()
    nested_rows = []
    for payload_hash, heldout_cases in groups.items():
        training_keys = sorted(key for key in common_keys if key[0] not in heldout_cases)
        heldout_keys = sorted(key for key in common_keys if key[0] in heldout_cases)
        training_summaries = {
            name: subset_summary(values, training_keys) for name, values in deployed.items()
        }
        training_ranking = rank_modes(training_summaries)
        selected = str(training_ranking[0]["mode"])
        selection_counts[selected] += 1
        heldout = subset_summary(deployed[selected], heldout_keys)
        nested_rows.append(
            {
                "cbct_sha256": payload_hash,
                "heldout_cases": ",".join(heldout_cases),
                "heldout_jaws": len(heldout_keys),
                "selected_mode": selected,
                "training_official_average_rank": training_ranking[0][
                    "official_average_rank"
                ],
                **heldout,
            }
        )

    fixed_by_name = {str(row["mode"]): row for row in fixed_ranking}
    nested_candidate = min(
        deployed,
        key=lambda name: (
            -selection_counts[name],
            float(fixed_by_name[name]["official_average_rank"]),
            int(fixed_by_name[name]["rotation_rank"]),
            int(fixed_by_name[name]["translation_rank"]),
            mode_complexity(name),
            name,
        ),
    )
    baseline = args.baseline_mode
    bootstrap = bootstrap_rank_comparison(
        deployed,
        groups,
        nested_candidate,
        baseline,
        args.bootstrap_samples,
        args.seed,
    )

    leave_one_group_out = []
    for payload_hash, heldout_cases in groups.items():
        retained_keys = sorted(key for key in common_keys if key[0] not in heldout_cases)
        ranking = rank_modes(
            {name: subset_summary(values, retained_keys) for name, values in deployed.items()}
        )
        by_name = {str(row["mode"]): row for row in ranking}
        leave_one_group_out.append(
            {
                "cbct_sha256": payload_hash,
                "candidate_rank_delta": float(
                    by_name[nested_candidate]["official_average_rank"]
                    - by_name[baseline]["official_average_rank"]
                ),
            }
        )
    candidate_row = fixed_by_name[nested_candidate]
    baseline_row = fixed_by_name[baseline]
    robust_criteria = {
        "candidate_fixed_official_rank_is_better": float(
            candidate_row["official_average_rank"]
        )
        < float(baseline_row["official_average_rank"]),
        "candidate_is_not_pareto_dominated_by_baseline": not (
            float(candidate_row["mean_translation_error_mm"])
            >= float(baseline_row["mean_translation_error_mm"])
            and float(candidate_row["mean_rotation_error_deg"])
            >= float(baseline_row["mean_rotation_error_deg"])
        ),
        "bootstrap_probability_better_is_at_least_0p80": float(
            bootstrap["probability_better_official_rank"]
        )
        >= 0.80,
        "candidate_never_loses_leave_one_group_out_rank": max(
            row["candidate_rank_delta"] for row in leave_one_group_out
        )
        <= 0.0,
        "candidate_nested_selection_count_is_not_lower": selection_counts[
            nested_candidate
        ]
        >= selection_counts[baseline],
    }
    candidate_selected = nested_candidate == baseline or all(robust_criteria.values())
    recommended_mode = nested_candidate if candidate_selected else baseline

    metric_rows = []
    for name in sorted(deployed):
        for key in sorted(common_keys):
            raw_metric = raw[name][key]
            deployed_metric = deployed[name][key]
            metric_rows.append(
                {
                    "mode": name,
                    "case_id": key[0],
                    "jaw": key[1],
                    "exact_fallback": int(key in exact),
                    "raw_translation_error_mm": raw_metric[0],
                    "raw_rotation_error_deg": raw_metric[1],
                    "translation_error_mm": deployed_metric[0],
                    "rotation_error_deg": deployed_metric[1],
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "official_metrics.csv", metric_rows)
    write_csv(args.output_dir / "fixed_mode_ranking.csv", fixed_ranking)
    write_csv(args.output_dir / "nested_selection.csv", nested_rows)
    summary = {
        "protocol": (
            "official mean translation and trace-geodesic rotation metrics with "
            "separate ranks averaged equally; mode selection is grouped by CBCT SHA256"
        ),
        "cases": len(cases),
        "jaws": len(common_keys),
        "cbct_groups": len(groups),
        "exact_fallback_jaws": len(exact),
        "raw_summaries": raw_summaries,
        "fixed_mode_ranking": fixed_ranking,
        "nested_selection_counts": dict(selection_counts.most_common()),
        "nested_candidate_mode": nested_candidate,
        "baseline_mode": baseline,
        "bootstrap_comparison": bootstrap,
        "leave_one_group_out": leave_one_group_out,
        "robust_criteria": robust_criteria,
        "candidate_selected": candidate_selected,
        "recommended_mode": recommended_mode,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "raw_summaries"}, indent=2))


if __name__ == "__main__":
    main()
