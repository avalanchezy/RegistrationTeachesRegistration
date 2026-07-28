from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_semisupervised_candidate_reranker import (
    cbct_groups_by_case,
    evaluate,
    feature_groups,
    fit_model,
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
        description="Strict CBCT-grouped hyperparameter sweep for mixed registration candidates."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labeled-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exact-loo", type=Path)
    parser.add_argument("--cbct-hash-cache", type=Path)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--top-unsupervised", type=int, nargs="+", default=(20, 40, 60))
    parser.add_argument("--top-oracle", type=int, nargs="+", default=(2, 4, 8))
    parser.add_argument("--min-samples-leaf", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--max-features", type=float, nargs="+", default=(0.4, 0.6, 0.8))
    parser.add_argument("--eval-top-candidates", type=int, nargs="+", default=(30, 60, 90))
    parser.add_argument(
        "--model-scopes", choices=("jaw", "shared"), nargs="+", default=("jaw", "shared")
    )
    parser.add_argument(
        "--model-types",
        choices=("extra_trees", "random_forest", "hist_gradient_boosting"),
        nargs="+",
        default=("extra_trees",),
    )
    parser.add_argument(
        "--target-transforms",
        choices=("log1p", "sqrt", "identity"),
        nargs="+",
        default=("log1p",),
    )
    parser.add_argument(
        "--optimization-targets",
        choices=REGISTRATION_TARGET_NAMES,
        nargs="+",
        default=("mean_tre_mm",),
    )
    parser.add_argument(
        "--selection-metric",
        choices=("tre", "official"),
        default="tre",
    )
    parser.add_argument(
        "--tree-criteria",
        choices=("squared_error", "absolute_error", "friedman_mse", "poisson"),
        nargs="+",
        default=("squared_error",),
    )
    parser.add_argument("--tree-max-depths", type=int, nargs="+", default=(0,))
    parser.add_argument("--cv-trees", type=int, default=150)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgb-l2", type=float, default=1.0)
    parser.add_argument("--hgb-ensemble-leaf-nodes", type=int, nargs="*", default=())
    parser.add_argument("--hgb-early-stopping", action="store_true")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--group-context-features", action="store_true")
    parser.add_argument("--roi-view-feature", action="store_true")
    parser.add_argument("--modality-features", action="store_true")
    parser.add_argument("--balance-candidate-runs", action="store_true")
    parser.add_argument("--exclude-upper-opposite-axial", action="store_true")
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_tre_mm": float(array.mean()),
        "median_tre_mm": float(np.median(array)),
        "p90_tre_mm": float(np.quantile(array, 0.9)),
        "max_tre_mm": float(array.max()),
    }


def load_exact_errors(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["case_id"], row["jaw"]): float(row["mean_tre_mm"])
            for row in csv.DictReader(handle)
        }


def load_exact_official(path: Path | None) -> dict[tuple[str, str], tuple[float, float]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            (row["case_id"], row["jaw"]): (
                float(row["translation_error_mm"]),
                float(row["rotation_error_deg"]),
            )
            for row in csv.DictReader(handle)
        }


def summarize_official(
    rows: list[dict[str, object]],
    exact: dict[tuple[str, str], tuple[float, float]],
) -> dict[str, float]:
    values = np.asarray(
        [
            exact.get(
                (str(row["case_id"]), str(row["jaw"])),
                (
                    float(row["translation_error_mm"]),
                    float(row["rotation_error_deg"]),
                ),
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    return {
        "exact_fallback_mean_translation_error_mm": float(np.mean(values[:, 0])),
        "exact_fallback_mean_rotation_error_deg": float(np.mean(values[:, 1])),
        "exact_fallback_official_balanced_error": float(
            0.5 * (np.mean(values[:, 0]) / 10.0 + np.mean(values[:, 1]) / 5.0)
        ),
    }


def exact_fallback_errors(
    rows: list[dict[str, object]], exact: dict[tuple[str, str], float]
) -> list[float]:
    return [
        exact.get((str(row["case_id"]), str(row["jaw"])), float(row["mean_tre_mm"]))
        for row in rows
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def estimator_args(
    args: argparse.Namespace,
    top_unsupervised: int,
    top_oracle: int,
    min_samples_leaf: int,
    max_features: float,
    model_scope: str,
    model_type: str,
    target_transform: str,
    optimization_target: str,
    tree_criterion: str,
    tree_max_depth: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_type=model_type,
        target_transform=target_transform,
        optimization_target=optimization_target,
        tree_criterion=tree_criterion,
        tree_max_depth=tree_max_depth,
        cv_trees=args.cv_trees,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=args.n_jobs,
        hgb_learning_rate=args.hgb_learning_rate,
        hgb_max_leaf_nodes=args.hgb_max_leaf_nodes,
        hgb_l2=args.hgb_l2,
        hgb_ensemble_leaf_nodes=args.hgb_ensemble_leaf_nodes,
        hgb_early_stopping=args.hgb_early_stopping,
        jaw_specific_models=model_scope == "jaw",
        top_unsupervised=top_unsupervised,
        top_oracle=top_oracle,
        pseudo_top_unsupervised=0,
        pseudo_top_consensus=0,
        balance_candidate_runs=args.balance_candidate_runs,
        exact_template_weight_multiplier=1.0,
        geometry_pseudo_weight_multiplier=0.0,
        threshold_pseudo_weight_multiplier=0.0,
        exclude_upper_opposite_axial=args.exclude_upper_opposite_axial,
        eval_top_candidates=0,
    )


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
    folds = stratified_folds(groups, args.folds, args.seed, case_group)
    exact = load_exact_errors(args.exact_loo)
    exact_official = load_exact_official(args.exact_loo)

    fold_data = []
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
        fold_data.append((fold_index, validation_cases, features))

    grid_rows: list[dict[str, object]] = []
    best_key = None
    best_oof: list[dict[str, object]] = []
    base_grid = itertools.product(
        args.top_unsupervised,
        args.top_oracle,
        args.min_samples_leaf,
        args.max_features,
        args.model_scopes,
        args.model_types,
        args.target_transforms,
        args.optimization_targets,
        args.tree_criteria,
        args.tree_max_depths,
    )
    for config_index, config in enumerate(base_grid, start=1):
        (
            top_unsupervised,
            top_oracle,
            min_leaf,
            max_features,
            model_scope,
            model_type,
            target_transform,
            optimization_target,
            tree_criterion,
            tree_max_depth,
        ) = config
        model_args = estimator_args(
            args,
            top_unsupervised,
            top_oracle,
            min_leaf,
            max_features,
            model_scope,
            model_type,
            target_transform,
            optimization_target,
            tree_criterion,
            tree_max_depth,
        )
        predictions_by_budget = {budget: [] for budget in args.eval_top_candidates}
        for fold_index, validation_cases, features in fold_data:
            model = fit_model(
                groups,
                features,
                train_cases - validation_cases,
                {},
                {},
                {},
                0.0,
                model_args,
                args.seed + fold_index,
            )
            for budget in args.eval_top_candidates:
                model_args.eval_top_candidates = budget
                _, rows = evaluate(model, groups, features, validation_cases, model_args)
                for row in rows:
                    row["fold"] = fold_index
                predictions_by_budget[budget].extend(rows)

        for budget, rows in predictions_by_budget.items():
            raw = summarize([float(row["mean_tre_mm"]) for row in rows])
            combined = summarize(exact_fallback_errors(rows, exact))
            official = summarize_official(rows, exact_official)
            grid_row = {
                "top_unsupervised": top_unsupervised,
                "top_oracle": top_oracle,
                "min_samples_leaf": min_leaf,
                "max_features": max_features,
                "model_scope": model_scope,
                "model_type": model_type,
                "target_transform": target_transform,
                "optimization_target": optimization_target,
                "tree_criterion": tree_criterion,
                "tree_max_depth": tree_max_depth,
                "eval_top_candidates": budget,
                **raw,
                "exact_fallback_mean_tre_mm": combined["mean_tre_mm"],
                "exact_fallback_median_tre_mm": combined["median_tre_mm"],
                "exact_fallback_p90_tre_mm": combined["p90_tre_mm"],
                "exact_fallback_max_tre_mm": combined["max_tre_mm"],
                "exact_fallback_jaws": len(set(exact) & {(str(r['case_id']), str(r['jaw'])) for r in rows}),
                **official,
            }
            grid_rows.append(grid_row)
            if args.selection_metric == "official":
                selection_key = (
                    official["exact_fallback_official_balanced_error"],
                    official["exact_fallback_mean_rotation_error_deg"],
                    official["exact_fallback_mean_translation_error_mm"],
                    combined["mean_tre_mm"],
                )
            else:
                selection_key = (
                    combined["mean_tre_mm"],
                    combined["p90_tre_mm"],
                    combined["max_tre_mm"],
                    raw["mean_tre_mm"],
                )
            if best_key is None or selection_key < best_key:
                best_key = selection_key
                best_oof = [
                    dict(
                        row,
                        sweep_top_unsupervised=top_unsupervised,
                        sweep_top_oracle=top_oracle,
                        sweep_min_samples_leaf=min_leaf,
                        sweep_max_features=max_features,
                        sweep_model_scope=model_scope,
                        sweep_model_type=model_type,
                        sweep_target_transform=target_transform,
                        sweep_optimization_target=optimization_target,
                        sweep_tree_criterion=tree_criterion,
                        sweep_tree_max_depth=tree_max_depth,
                        sweep_eval_top_candidates=budget,
                    )
                    for row in rows
                ]
        write_csv(args.output_dir / "grid.csv", grid_rows)
        current = min(
            grid_rows,
            key=(
                (lambda row: (
                    float(row["exact_fallback_official_balanced_error"]),
                    float(row["exact_fallback_mean_rotation_error_deg"]),
                    float(row["exact_fallback_mean_translation_error_mm"]),
                ))
                if args.selection_metric == "official"
                else (lambda row: (
                    float(row["exact_fallback_mean_tre_mm"]),
                    float(row["exact_fallback_p90_tre_mm"]),
                    float(row["exact_fallback_max_tre_mm"]),
                ))
            ),
        )
        print(
            f"[{config_index}] best exact-fallback mean="
            f"{float(current['exact_fallback_mean_tre_mm']):.4f} mm | {current}",
            flush=True,
        )

    best = min(
        grid_rows,
        key=(
            (lambda row: (
                float(row["exact_fallback_official_balanced_error"]),
                float(row["exact_fallback_mean_rotation_error_deg"]),
                float(row["exact_fallback_mean_translation_error_mm"]),
            ))
            if args.selection_metric == "official"
            else (lambda row: (
                float(row["exact_fallback_mean_tre_mm"]),
                float(row["exact_fallback_p90_tre_mm"]),
                float(row["exact_fallback_max_tre_mm"]),
            ))
        ),
    )
    write_csv(args.output_dir / "best_oof.csv", best_oof)
    summary = {
        "cases": len(train_cases),
        "jaws": len(groups),
        "cbct_groups": len(set(case_group.values())),
        "grid_rows": len(grid_rows),
        "exact_fallback_jaws": len(exact),
        "best": best,
        "seed": args.seed,
        "cv_trees": args.cv_trees,
        "model_types": list(args.model_types),
        "target_transforms": list(args.target_transforms),
        "optimization_targets": list(args.optimization_targets),
        "selection_metric": args.selection_metric,
        "tree_criteria": list(args.tree_criteria),
        "tree_max_depths": list(args.tree_max_depths),
        "feature_configuration": {
            "group_context_features": args.group_context_features,
            "roi_view_feature": args.roi_view_feature,
            "modality_features": args.modality_features,
            "balance_candidate_runs": args.balance_candidate_runs,
            "exclude_upper_opposite_axial": args.exclude_upper_opposite_axial,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
