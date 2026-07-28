from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_multiseed_reranker_ensemble import (
    aggregate_predictions,
    selected_row,
)
from scripts.sweep_multimodal_reranker import (
    exact_fallback_errors,
    load_exact_errors,
    load_exact_official,
    summarize,
    summarize_official,
    write_csv,
)
from scripts.sweep_pairwise_multimodal_reranker import (
    fit_pairwise_model,
    score_group,
)
from scripts.train_semisupervised_candidate_reranker import (
    cbct_groups_by_case,
    feature_groups,
    stratified_folds,
)
from task2reg.candidate_learning import (
    REGISTRATION_TARGET_NAMES,
    enrich_candidate_registration_metrics,
    load_candidate_groups,
)
from task2reg.data import load_manifest
from task2reg.priors import fit_rotation_prior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multiseed strict CBCT-grouped pairwise candidate ensemble."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labeled-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exact-loo", type=Path)
    parser.add_argument("--cbct-hash-cache", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--top-unsupervised", type=int, required=True)
    parser.add_argument("--top-oracle", type=int, required=True)
    parser.add_argument("--min-samples-leaf", type=int, required=True)
    parser.add_argument("--max-features", type=float, required=True)
    parser.add_argument("--eval-top-candidates", type=int, required=True)
    parser.add_argument("--model-scope", choices=("jaw", "shared"), default="jaw")
    parser.add_argument(
        "--model-type", choices=("extra_trees", "random_forest"), default="extra_trees"
    )
    parser.add_argument(
        "--criterion", choices=("gini", "entropy", "log_loss"), default="gini"
    )
    parser.add_argument(
        "--optimization-target",
        choices=REGISTRATION_TARGET_NAMES,
        default="mean_tre_mm",
    )
    parser.add_argument("--min-log-tre-gap", type=float, default=0.05)
    parser.add_argument("--max-pairs-per-group", type=int, default=600)
    parser.add_argument("--eval-opponents", type=int, default=30)
    parser.add_argument("--cv-trees", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--group-context-features", action="store_true")
    parser.add_argument("--roi-view-feature", action="store_true")
    parser.add_argument("--modality-features", action="store_true")
    parser.add_argument("--balance-candidate-runs", action="store_true")
    parser.add_argument("--exclude-upper-opposite-axial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.roi_view_feature and not args.group_context_features:
        raise ValueError("--roi-view-feature requires --group-context-features")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(args.manifest)
    groups = load_candidate_groups(args.labeled_runs)
    enrich_candidate_registration_metrics(groups, records)
    train_cases = {key[0] for key in groups}
    case_group = cbct_groups_by_case(records, train_cases, args.cbct_hash_cache)
    exact = load_exact_errors(args.exact_loo)
    exact_official = load_exact_official(args.exact_loo)
    config: dict[str, object] = {
        "top_unsupervised": args.top_unsupervised,
        "top_oracle": args.top_oracle,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "model_scope": args.model_scope,
        "model_type": args.model_type,
        "criterion": args.criterion,
        "optimization_target": args.optimization_target,
        "min_log_tre_gap": args.min_log_tre_gap,
    }

    seed_predictions: list[dict[tuple[str, str], np.ndarray]] = []
    for seed_index, seed in enumerate(args.seeds, start=1):
        folds = stratified_folds(groups, args.folds, seed, case_group)
        predictions = {
            key: np.full(len(rows), np.nan, dtype=np.float64) for key, rows in groups.items()
        }
        for fold_index, validation_cases in enumerate(folds):
            priors = {
                jaw: fit_rotation_prior(records, jaw, excluded_cases=validation_cases)
                for jaw in ("upper", "lower")
            }
            features = feature_groups(
                groups,
                priors,
                args.group_context_features,
                args.roi_view_feature,
                args.modality_features,
            )
            model = fit_pairwise_model(
                groups,
                features,
                train_cases - validation_cases,
                config,
                args,
                seed + fold_index,
                args.cv_trees,
            )
            for key, rows in groups.items():
                if key[0] not in validation_cases:
                    continue
                estimator = model[key[1]] if isinstance(model, dict) else model
                indices, scores = score_group(
                    estimator,
                    rows,
                    features[key],
                    key[1],
                    args.eval_top_candidates,
                    args,
                )
                predictions[key][indices] = -scores
        if any(np.all(np.isnan(values)) for values in predictions.values()):
            raise RuntimeError(f"Seed {seed} did not predict every candidate group")
        seed_predictions.append(predictions)
        print(
            f"[{seed_index}/{len(args.seeds)}] completed strict OOF seed {seed}",
            flush=True,
        )

    methods = ("mean", "median", "rank_mean", "vote")
    policy_rows: list[dict[str, object]] = []
    candidate_score_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "cases": len(train_cases),
        "jaws": len(groups),
        "cbct_groups": len(set(case_group.values())),
        "seeds": args.seeds,
        "configuration": {
            **config,
            "eval_top_candidates": args.eval_top_candidates,
            "eval_opponents": args.eval_opponents,
            "max_pairs_per_group": args.max_pairs_per_group,
            "cv_trees": args.cv_trees,
        },
        "policies": {},
    }
    for method in methods:
        rows_out = []
        for key, rows in sorted(groups.items()):
            available = np.flatnonzero(~np.isnan(seed_predictions[0][key]))
            prediction_matrix = np.stack(
                [predictions[key][available] for predictions in seed_predictions]
            )
            if np.isnan(prediction_matrix).any():
                raise RuntimeError(f"Inconsistent candidate budget for {key}")
            scores = aggregate_predictions(prediction_matrix, method)
            for index, score in zip(available, scores):
                candidate_score_rows.append(
                    {
                        "case_id": key[0],
                        "jaw": key[1],
                        "ensemble_method": method,
                        "candidate_index": int(index),
                        "ensemble_score": float(score),
                        "mean_tre_mm": float(rows[int(index)]["mean_tre_mm"]),
                        "translation_error_mm": float(
                            rows[int(index)]["translation_error_mm"]
                        ),
                        "rotation_error_deg": float(
                            rows[int(index)]["rotation_error_deg"]
                        ),
                        "official_balanced_error": float(
                            rows[int(index)]["official_balanced_error"]
                        ),
                        "optimization_target": args.optimization_target,
                        "candidate_run": rows[int(index)].get("candidate_run", ""),
                    }
                )
            local_index = int(np.argmin(scores))
            index = int(available[local_index])
            row = selected_row(
                key[0], key[1], rows, index, float(scores[local_index]), method
            )
            row["optimization_target"] = args.optimization_target
            rows_out.append(row)
        raw = summarize([float(row["mean_tre_mm"]) for row in rows_out])
        combined = summarize(exact_fallback_errors(rows_out, exact))
        official = summarize_official(rows_out, exact_official)
        summary["policies"][method] = {
            "raw": raw,
            "exact_fallback": combined,
            "official": official,
        }
        write_csv(args.output_dir / f"oof_{method}.csv", rows_out)
        policy_rows.extend(rows_out)

    write_csv(args.output_dir / "all_policy_selections.csv", policy_rows)
    write_csv(args.output_dir / "candidate_ensemble_scores.csv", candidate_score_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
