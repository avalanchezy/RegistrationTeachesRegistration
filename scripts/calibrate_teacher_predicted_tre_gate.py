from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.candidate_learning import load_candidate_groups
from task2reg.data import load_manifest
from scripts.train_semisupervised_candidate_reranker import cbct_groups_by_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate deployment predicted-TRE gates from strictly OOF candidate "
            "selections and cached per-seed OOF regression predictions."
        )
    )
    parser.add_argument("--candidate-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--selected-oof", type=Path, required=True)
    parser.add_argument("--regression-oof", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cbct-hash-cache", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0),
    )
    return parser.parse_args()


def transform_key(row: dict) -> str:
    return ",".join(
        f"{value:.6f}"
        for value in np.asarray(row["transform"], dtype=np.float64).reshape(-1)
    )


def read_selected(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def summarize(values: np.ndarray) -> dict[str, float | int]:
    return {
        "accepted": int(len(values)),
        "mean_true_tre_mm": float(np.mean(values)),
        "median_true_tre_mm": float(np.median(values)),
        "p90_true_tre_mm": float(np.quantile(values, 0.9)),
        "maximum_true_tre_mm": float(np.max(values)),
        "success_le_2mm": float(np.mean(values <= 2.0)),
        "success_le_3mm": float(np.mean(values <= 3.0)),
    }


def grouped_bootstrap_diagnostics(
    rows: list[dict[str, object]],
    accepted_mask: np.ndarray,
    case_groups: dict[str, str],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    group_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        case_id = str(row["case_id"])
        group_indices.setdefault(case_groups.get(case_id, case_id), []).append(index)
    groups = sorted(group_indices)
    accepted_groups = {
        group
        for group, indices in group_indices.items()
        if any(bool(accepted_mask[index]) for index in indices)
    }
    result: dict[str, object] = {
        "cbct_groups": len(groups),
        "accepted_cbct_groups": len(accepted_groups),
    }
    if samples < 1 or not accepted_groups:
        return result
    rng = np.random.default_rng(seed)
    p90_values: list[float] = []
    mean_values: list[float] = []
    success_values: list[float] = []
    for _ in range(samples):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_indices = [
            index
            for group in sampled_groups
            for index in group_indices[str(group)]
            if bool(accepted_mask[index])
        ]
        if not sampled_indices:
            continue
        values = np.asarray(
            [float(rows[index]["true_tre_mm"]) for index in sampled_indices],
            dtype=np.float64,
        )
        mean_values.append(float(np.mean(values)))
        p90_values.append(float(np.quantile(values, 0.9)))
        success_values.append(float(np.mean(values <= 3.0)))
    result.update(
        {
            "bootstrap_samples": len(p90_values),
            "bootstrap_mean_true_tre_ci95_mm": [
                float(np.quantile(mean_values, 0.025)),
                float(np.quantile(mean_values, 0.975)),
            ],
            "bootstrap_p90_true_tre_ci95_mm": [
                float(np.quantile(p90_values, 0.025)),
                float(np.quantile(p90_values, 0.975)),
            ],
            "bootstrap_success_le_3mm_ci95": [
                float(np.quantile(success_values, 0.025)),
                float(np.quantile(success_values, 0.975)),
            ],
            "bootstrap_probability_p90_le_2p5": float(
                np.mean(np.asarray(p90_values) <= 2.5)
            ),
            "bootstrap_probability_success_le_3mm_ge_0p95": float(
                np.mean(np.asarray(success_values) >= 0.95)
            ),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    groups = load_candidate_groups(args.candidate_runs)
    selected = read_selected(args.selected_oof)
    seed_predictions = [joblib.load(path)["predictions"] for path in args.regression_oof]
    rows: list[dict[str, object]] = []

    for payload in selected:
        key = (str(payload["case_id"]), str(payload["jaw"]))
        candidates = groups[key]
        wanted = str(payload["transform_key"])
        matches = [
            index for index, candidate in enumerate(candidates) if transform_key(candidate) == wanted
        ]
        if not matches:
            raise RuntimeError(f"Selected transform is absent from candidate group {key}")
        if len(matches) > 1:
            source = str(payload.get("candidate_run", "")).lower()
            source_matches = [
                index
                for index in matches
                if str(candidates[index].get("candidate_run", "")).lower() == source
            ]
            if source_matches:
                matches = source_matches
        index = matches[0]
        predictions = np.asarray(
            [np.asarray(seed[key], dtype=np.float64)[index] for seed in seed_predictions],
            dtype=np.float64,
        )
        predictions = predictions[np.isfinite(predictions)]
        if not len(predictions):
            raise RuntimeError(f"No finite OOF regression prediction for {key} index {index}")
        candidate = candidates[index]
        candidate_run = str(candidate.get("candidate_run", ""))
        full_p90 = candidate.get(
            "full_distance_p90_mm", candidate.get("full_p90_mm", float("nan"))
        )
        rows.append(
            {
                "case_id": key[0],
                "jaw": key[1],
                "predicted_tre_mm": float(np.median(predictions)),
                "prediction_seed_mean_mm": float(np.mean(predictions)),
                "prediction_seed_std_mm": float(np.std(predictions)),
                "prediction_seeds": int(len(predictions)),
                "true_tre_mm": float(payload["mean_tre_mm"]),
                "absolute_calibration_error_mm": float(
                    abs(np.median(predictions) - float(payload["mean_tre_mm"]))
                ),
                "roi_used": "roi" in candidate_run.lower(),
                "full_p90_mm": float(full_p90),
                "candidate_run": candidate_run,
                "target": str(candidate.get("target", "")),
                "method": str(candidate.get("method", "")),
            }
        )

    predicted = np.asarray([float(row["predicted_tre_mm"]) for row in rows])
    truth = np.asarray([float(row["true_tre_mm"]) for row in rows])
    if args.manifest is not None:
        case_groups = cbct_groups_by_case(
            load_manifest(args.manifest),
            {str(row["case_id"]) for row in rows},
            args.cbct_hash_cache,
        )
    else:
        case_groups = {str(row["case_id"]): str(row["case_id"]) for row in rows}
    grid = []
    for threshold_index, threshold in enumerate(sorted(set(args.thresholds))):
        accepted_mask = predicted <= threshold
        accepted = truth[accepted_mask]
        record: dict[str, object] = {
            "max_predicted_tre_mm": float(threshold),
            "coverage": float(len(accepted) / len(truth)),
        }
        if len(accepted):
            record.update(summarize(accepted))
            record.update(
                grouped_bootstrap_diagnostics(
                    rows,
                    accepted_mask,
                    case_groups,
                    samples=args.bootstrap_samples,
                    seed=args.seed + threshold_index,
                )
            )
        grid.append(record)

    eligible = [
        row
        for row in grid
        if int(row.get("accepted", 0)) >= 8
        and float(row.get("p90_true_tre_mm", np.inf)) <= 2.5
        and float(row.get("success_le_3mm", 0.0)) >= 0.95
    ]
    recommended = max(eligible, key=lambda row: float(row["coverage"])) if eligible else None
    correlation = (
        float(np.corrcoef(predicted, truth)[0, 1])
        if len(rows) > 1 and np.std(predicted) > 0 and np.std(truth) > 0
        else None
    )
    summary = {
        "protocol": "strict grouped OOF selected-candidate calibration",
        "jaws": len(rows),
        "cbct_groups": len(set(case_groups.values())),
        "seeds": len(seed_predictions),
        "prediction_true_tre_pearson": correlation,
        "mean_absolute_calibration_error_mm": float(
            np.mean([float(row["absolute_calibration_error_mm"]) for row in rows])
        ),
        "recommended_gate": recommended,
        "gate_grid": grid,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({name for row in rows for name in row})
    with (args.output_dir / "per_jaw.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
