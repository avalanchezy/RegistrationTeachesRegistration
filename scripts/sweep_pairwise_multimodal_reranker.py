from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_semisupervised_candidate_reranker import (
    cbct_groups_by_case,
    feature_groups,
    stratified_folds,
    subset_by_scores,
    top_indices,
)
from task2reg.candidate_learning import (
    REGISTRATION_TARGET_NAMES,
    enrich_candidate_registration_metrics,
    is_opposite_axial_target,
    load_candidate_groups,
    registration_target_value,
)
from task2reg.data import load_manifest
from task2reg.priors import fit_rotation_prior
from scripts.sweep_multimodal_reranker import load_exact_official, summarize_official


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict CBCT-grouped pairwise learning-to-rank sweep."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labeled-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exact-loo", type=Path)
    parser.add_argument("--cbct-hash-cache", type=Path)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--top-unsupervised", type=int, nargs="+", default=(20,))
    parser.add_argument("--top-oracle", type=int, nargs="+", default=(8,))
    parser.add_argument("--min-samples-leaf", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--max-features", type=float, nargs="+", default=(0.2, 0.4, 0.8))
    parser.add_argument("--eval-top-candidates", type=int, nargs="+", default=(10, 20, 30))
    parser.add_argument("--model-scopes", choices=("jaw", "shared"), nargs="+", default=("jaw", "shared"))
    parser.add_argument("--model-types", choices=("extra_trees", "random_forest"), nargs="+", default=("extra_trees",))
    parser.add_argument("--criteria", choices=("gini", "entropy", "log_loss"), nargs="+", default=("gini",))
    parser.add_argument(
        "--optimization-targets",
        choices=REGISTRATION_TARGET_NAMES,
        nargs="+",
        default=("mean_tre_mm",),
    )
    parser.add_argument(
        "--selection-metric", choices=("tre", "official"), default="tre"
    )
    parser.add_argument("--min-log-tre-gaps", type=float, nargs="+", default=(0.05, 0.1, 0.2))
    parser.add_argument("--max-pairs-per-group", type=int, default=600)
    parser.add_argument("--eval-opponents", type=int, default=30)
    parser.add_argument("--cv-trees", type=int, default=200)
    parser.add_argument("--final-trees", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=6)
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_classifier(config: dict[str, object], trees: int, n_jobs: int, seed: int):
    common = dict(
        n_estimators=trees,
        criterion=str(config["criterion"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=float(config["max_features"]),
        class_weight="balanced",
        n_jobs=n_jobs,
        random_state=seed,
    )
    if config["model_type"] == "extra_trees":
        return ExtraTreesClassifier(**common)
    return RandomForestClassifier(**common)


def pairwise_target_values(
    rows: list[dict], indices: list[int], optimization_target: str
) -> np.ndarray:
    return np.log1p(
        [registration_target_value(rows[index], optimization_target) for index in indices]
    )


def fit_pairwise_model(
    groups,
    features,
    training_cases: set[str],
    config: dict[str, object],
    args: argparse.Namespace,
    seed: int,
    trees: int,
):
    rng = np.random.default_rng(seed)
    x_by_jaw: dict[str, list[np.ndarray]] = {"upper": [], "lower": []}
    y_by_jaw: dict[str, list[int]] = {"upper": [], "lower": []}
    w_by_jaw: dict[str, list[float]] = {"upper": [], "lower": []}
    for key, rows in sorted(groups.items()):
        if key[0] not in training_cases:
            continue
        optimization_target = str(config.get("optimization_target", "mean_tre_mm"))
        selected = subset_by_scores(
            rows,
            optimization_target,
            int(config["top_unsupervised"]),
            int(config["top_oracle"]),
            args.balance_candidate_runs,
        )
        index_by_id = {id(row): index for index, row in enumerate(rows)}
        indices = [index_by_id[id(row)] for row in selected]
        target_values = pairwise_target_values(rows, indices, optimization_target)
        pairs = [
            (indices[first], indices[second])
            for first, second in itertools.combinations(range(len(indices)), 2)
            if abs(target_values[first] - target_values[second]) >= float(config["min_log_tre_gap"])
        ]
        if len(pairs) > args.max_pairs_per_group:
            chosen = rng.choice(len(pairs), args.max_pairs_per_group, replace=False)
            pairs = [pairs[int(index)] for index in chosen]
        if not pairs:
            continue
        group_weight = 1.0 / (2.0 * len(pairs))
        jaw = key[1]
        for first, second in pairs:
            difference = features[key][first] - features[key][second]
            first_better = int(
                registration_target_value(rows[first], optimization_target)
                < registration_target_value(rows[second], optimization_target)
            )
            x_by_jaw[jaw].extend((difference, -difference))
            y_by_jaw[jaw].extend((first_better, 1 - first_better))
            w_by_jaw[jaw].extend((group_weight, group_weight))

    if config["model_scope"] == "jaw":
        models = {}
        for jaw_index, jaw in enumerate(("upper", "lower")):
            model = build_classifier(config, trees, args.n_jobs, seed + 1009 * jaw_index)
            model.fit(
                np.stack(x_by_jaw[jaw]),
                np.asarray(y_by_jaw[jaw]),
                sample_weight=np.asarray(w_by_jaw[jaw]),
            )
            models[jaw] = model
        return models

    model = build_classifier(config, trees, args.n_jobs, seed)
    model.fit(
        np.stack(x_by_jaw["upper"] + x_by_jaw["lower"]),
        np.asarray(y_by_jaw["upper"] + y_by_jaw["lower"]),
        sample_weight=np.asarray(w_by_jaw["upper"] + w_by_jaw["lower"]),
    )
    return model


def evaluation_indices(rows: list[dict], jaw: str, budget: int, args: argparse.Namespace):
    pool = list(range(len(rows)))
    if args.exclude_upper_opposite_axial and jaw == "upper":
        filtered = [index for index in pool if not is_opposite_axial_target(rows[index], jaw)]
        if filtered:
            pool = filtered
    subset = [rows[index] for index in pool]
    local = top_indices(subset, "selection_score_mm", budget, args.balance_candidate_runs)
    return [pool[index] for index in local]


def score_group(model, rows, features: np.ndarray, jaw: str, budget: int, args):
    candidates = evaluation_indices(rows, jaw, budget, args)
    opponents = candidates[: min(args.eval_opponents, len(candidates))]
    differences = []
    comparison_counts = []
    for candidate in candidates:
        comparison = [index for index in opponents if index != candidate]
        if not comparison:
            comparison_counts.append(0)
            continue
        differences.append(features[candidate][None, :] - features[comparison])
        comparison_counts.append(len(comparison))
    probabilities = (
        model.predict_proba(np.concatenate(differences, axis=0))[:, 1]
        if differences
        else np.empty(0, dtype=np.float64)
    )
    scores = []
    offset = 0
    for count in comparison_counts:
        if count == 0:
            scores.append(0.5)
            continue
        scores.append(float(probabilities[offset : offset + count].mean()))
        offset += count
    return candidates, np.asarray(scores, dtype=np.float64)


def rank_group(model, rows, features: np.ndarray, jaw: str, budget: int, args):
    candidates, scores = score_group(model, rows, features, jaw, budget, args)
    local_index = int(np.argmax(scores))
    return candidates[local_index], float(scores[local_index])


def evaluate(model, groups, features, cases: set[str], budget: int, args):
    output = []
    for key, rows in sorted(groups.items()):
        if key[0] not in cases:
            continue
        estimator = model[key[1]] if isinstance(model, dict) else model
        index, score = rank_group(estimator, rows, features[key], key[1], budget, args)
        selected = rows[index]
        optimization_target = str(
            getattr(args, "optimization_target", "mean_tre_mm")
        )
        oracle = min(
            rows, key=lambda row: registration_target_value(row, optimization_target)
        )
        output.append(
            {
                "case_id": key[0],
                "jaw": key[1],
                "pairwise_win_score": score,
                "mean_tre_mm": float(selected["mean_tre_mm"]),
                "translation_error_mm": float(selected["translation_error_mm"]),
                "rotation_error_deg": float(selected["rotation_error_deg"]),
                "official_balanced_error": float(selected["official_balanced_error"]),
                "optimization_target": optimization_target,
                "original_unsupervised_rank": int(selected["unsupervised_rank"]),
                "source_variant": selected["source_variant"],
                "target": selected["target"],
                "method": selected["method"],
                "candidate_run": selected.get("candidate_run", ""),
                "transform_key": ",".join(
                    f"{value:.6f}"
                    for value in np.asarray(selected["transform"], dtype=np.float64).reshape(-1)
                ),
                "oracle_mean_tre_mm": float(oracle["mean_tre_mm"]),
                "oracle_optimization_target": registration_target_value(
                    oracle, optimization_target
                ),
            }
        )
    return output


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

    grid_rows = []
    best_key = None
    best_oof = []
    configurations = itertools.product(
        args.top_unsupervised,
        args.top_oracle,
        args.min_samples_leaf,
        args.max_features,
        args.model_scopes,
        args.model_types,
        args.criteria,
        args.optimization_targets,
        args.min_log_tre_gaps,
    )
    for config_index, values in enumerate(configurations, start=1):
        config = dict(
            top_unsupervised=int(values[0]),
            top_oracle=int(values[1]),
            min_samples_leaf=int(values[2]),
            max_features=float(values[3]),
            model_scope=str(values[4]),
            model_type=str(values[5]),
            criterion=str(values[6]),
            optimization_target=str(values[7]),
            min_log_tre_gap=float(values[8]),
        )
        args.optimization_target = str(config["optimization_target"])
        rows_by_budget = {budget: [] for budget in args.eval_top_candidates}
        for fold_index, validation_cases, features in fold_data:
            model = fit_pairwise_model(
                groups,
                features,
                train_cases - validation_cases,
                config,
                args,
                args.seed + fold_index,
                args.cv_trees,
            )
            for budget in args.eval_top_candidates:
                rows = evaluate(model, groups, features, validation_cases, budget, args)
                for row in rows:
                    row["fold"] = fold_index
                rows_by_budget[budget].extend(rows)

        for budget, rows in rows_by_budget.items():
            raw = summarize([float(row["mean_tre_mm"]) for row in rows])
            combined = summarize(
                [
                    exact.get((str(row["case_id"]), str(row["jaw"])), float(row["mean_tre_mm"]))
                    for row in rows
                ]
            )
            official = summarize_official(rows, exact_official)
            grid_row = {
                **config,
                "eval_top_candidates": budget,
                **raw,
                "exact_fallback_mean_tre_mm": combined["mean_tre_mm"],
                "exact_fallback_median_tre_mm": combined["median_tre_mm"],
                "exact_fallback_p90_tre_mm": combined["p90_tre_mm"],
                "exact_fallback_max_tre_mm": combined["max_tre_mm"],
                **official,
            }
            grid_rows.append(grid_row)
            key = (
                (
                    official["exact_fallback_official_balanced_error"],
                    official["exact_fallback_mean_rotation_error_deg"],
                    official["exact_fallback_mean_translation_error_mm"],
                )
                if args.selection_metric == "official"
                else (
                    combined["mean_tre_mm"],
                    combined["p90_tre_mm"],
                    combined["max_tre_mm"],
                    raw["mean_tre_mm"],
                )
            )
            if best_key is None or key < best_key:
                best_key = key
                best_oof = [dict(row, **config, eval_top_candidates=budget) for row in rows]
        write_csv(args.output_dir / "grid.csv", grid_rows)
        current = min(
            grid_rows,
            key=lambda row: (
                float(row["exact_fallback_official_balanced_error"]),
                float(row["exact_fallback_mean_rotation_error_deg"]),
                float(row["exact_fallback_mean_translation_error_mm"]),
            )
            if args.selection_metric == "official"
            else (
                float(row["exact_fallback_mean_tre_mm"]),
                float(row["exact_fallback_p90_tre_mm"]),
                float(row["exact_fallback_max_tre_mm"]),
            ),
        )
        print(
            f"[{config_index}] best exact-fallback mean="
            f"{float(current['exact_fallback_mean_tre_mm']):.4f} mm",
            flush=True,
        )

    best = min(
        grid_rows,
        key=lambda row: (
            float(row["exact_fallback_official_balanced_error"]),
            float(row["exact_fallback_mean_rotation_error_deg"]),
            float(row["exact_fallback_mean_translation_error_mm"]),
        )
        if args.selection_metric == "official"
        else (
            float(row["exact_fallback_mean_tre_mm"]),
            float(row["exact_fallback_p90_tre_mm"]),
            float(row["exact_fallback_max_tre_mm"]),
        ),
    )
    write_csv(args.output_dir / "best_oof.csv", best_oof)
    final_priors = {jaw: fit_rotation_prior(records, jaw, excluded_cases=set()) for jaw in ("upper", "lower")}
    final_features = feature_groups(
        groups,
        final_priors,
        args.group_context_features,
        args.roi_view_feature,
        args.modality_features,
    )
    final_model = fit_pairwise_model(
        groups, final_features, train_cases, best, args, args.seed, args.final_trees
    )
    joblib.dump(
        {
            "model": final_model,
            "best_config": best,
            "group_context_features": args.group_context_features,
            "roi_view_feature": args.roi_view_feature,
            "modality_features": args.modality_features,
            "balance_candidate_runs": args.balance_candidate_runs,
            "exclude_upper_opposite_axial": args.exclude_upper_opposite_axial,
            "eval_opponents": args.eval_opponents,
        },
        args.output_dir / "pairwise_reranker.joblib",
    )
    summary = {
        "cases": len(train_cases),
        "jaws": len(groups),
        "cbct_groups": len(set(case_group.values())),
        "grid_rows": len(grid_rows),
        "exact_fallback_jaws": len(exact),
        "selection_metric": args.selection_metric,
        "best": best,
        "seed": args.seed,
        "cv_trees": args.cv_trees,
        "final_trees": args.final_trees,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
