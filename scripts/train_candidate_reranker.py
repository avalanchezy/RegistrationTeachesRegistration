from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.candidate_learning import FEATURE_NAMES, candidate_features, load_candidate_groups
from task2reg.data import load_manifest
from task2reg.priors import fit_rotation_prior


def training_subset(rows: list[dict], top_unsupervised: int, top_oracle: int) -> list[dict]:
    by_fit = sorted(rows, key=lambda row: float(row["selection_score_mm"]))[:top_unsupervised]
    by_truth = sorted(rows, key=lambda row: float(row["mean_tre_mm"]))[:top_oracle]
    selected = {}
    for row in (*by_fit, *by_truth):
        key = tuple(np.asarray(row["transform"]).round(7).reshape(-1))
        selected[key] = row
    return list(selected.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-case-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-unsupervised", type=int, default=40)
    parser.add_argument("--top-oracle", type=int, default=12)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", type=float, default=0.8)
    parser.add_argument("--eval-top-candidates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    eval_cases = set(args.eval_case_ids)
    records = load_manifest(args.manifest)
    priors = {
        jaw: fit_rotation_prior(records, jaw, excluded_cases=eval_cases)
        for jaw in ("upper", "lower")
    }
    train_groups = load_candidate_groups(args.train_runs)
    eval_groups = load_candidate_groups(args.eval_runs)
    x_train, y_train, weights = [], [], []
    for (case_id, jaw), rows in train_groups.items():
        if case_id in eval_cases:
            continue
        selected = training_subset(rows, args.top_unsupervised, args.top_oracle)
        group_weight = 1.0 / max(len(selected), 1)
        for row in selected:
            x_train.append(candidate_features(row, priors[jaw]))
            y_train.append(np.log1p(float(row["mean_tre_mm"])))
            weights.append(group_weight)
    if not x_train:
        raise RuntimeError("No labeled training candidates were found")

    model = ExtraTreesRegressor(
        n_estimators=args.trees,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(np.stack(x_train), np.asarray(y_train), sample_weight=np.asarray(weights))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "eval_cases": sorted(eval_cases),
            "top_unsupervised": args.top_unsupervised,
            "top_oracle": args.top_oracle,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "eval_top_candidates": args.eval_top_candidates,
        },
        args.output_dir / "candidate_reranker.joblib",
    )

    output_rows = []
    for (case_id, jaw), rows in sorted(eval_groups.items()):
        if case_id not in eval_cases:
            continue
        candidate_rows = sorted(
            rows, key=lambda row: float(row["selection_score_mm"])
        )[: args.eval_top_candidates]
        features = np.stack([candidate_features(row, priors[jaw]) for row in candidate_rows])
        predictions = np.expm1(model.predict(features))
        selected_index = int(np.argmin(predictions))
        selected = candidate_rows[selected_index]
        oracle = min(rows, key=lambda row: float(row["mean_tre_mm"]))
        output_rows.append(
            {
                "case_id": case_id,
                "jaw": jaw,
                "predicted_tre_mm": float(predictions[selected_index]),
                "mean_tre_mm": float(selected["mean_tre_mm"]),
                "original_unsupervised_rank": int(selected["unsupervised_rank"]),
                "source_variant": selected["source_variant"],
                "target": selected["target"],
                "method": selected["method"],
                "candidate_run": selected.get("candidate_run", ""),
                "oracle_mean_tre_mm": float(oracle["mean_tre_mm"]),
                "oracle_original_rank": int(oracle["unsupervised_rank"]),
            }
        )
        print(
            f"{case_id} {jaw}: predicted={predictions[selected_index]:.3f} mm "
            f"TRE={float(selected['mean_tre_mm']):.3f} mm rank={selected['unsupervised_rank']}"
        )
    if not output_rows:
        raise RuntimeError("No evaluation candidates matched --eval-case-ids")
    output_csv = args.output_dir / "evaluation.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    errors = np.asarray([row["mean_tre_mm"] for row in output_rows])
    print(f"Saved: {output_csv}")
    print(f"Mean TRE: {errors.mean():.3f} mm; median TRE: {np.median(errors):.3f} mm")


if __name__ == "__main__":
    main()
