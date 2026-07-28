from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.candidate_learning import FEATURE_NAMES, candidate_features, load_candidate_groups
from task2reg.data import load_manifest
from task2reg.priors import fit_rotation_prior
from task2reg.template_transfer import sha256_nifti_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune a candidate reranker with case-grouped, chirality-stratified CV."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-runs", type=Path, nargs="+", default=())
    parser.add_argument("--eval-case-ids", nargs="*", default=())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--top-unsupervised", type=int, nargs="+", default=(20, 40, 80))
    parser.add_argument("--top-oracle", type=int, nargs="+", default=(4, 8, 12))
    parser.add_argument("--min-samples-leaf", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--max-features", type=float, nargs="+", default=(0.6, 0.8, 1.0))
    parser.add_argument(
        "--eval-top-candidates",
        type=int,
        nargs="+",
        default=(20, 40, 80),
        help="Restrict inference to candidates inside the trained unsupervised-score support.",
    )
    parser.add_argument("--cv-trees", type=int, default=250)
    parser.add_argument("--final-trees", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--cv-grouping",
        choices=("cbct", "case"),
        default="cbct",
        help="Keep cases sharing the same decompressed CBCT payload in one fold.",
    )
    return parser.parse_args()


def training_subset(rows: list[dict], top_unsupervised: int, top_oracle: int) -> list[dict]:
    selected: dict[tuple[float, ...], dict] = {}
    for row in sorted(rows, key=lambda item: float(item["selection_score_mm"]))[:top_unsupervised]:
        key = tuple(np.asarray(row["transform"]).round(7).reshape(-1))
        selected[key] = row
    for row in sorted(rows, key=lambda item: float(item["mean_tre_mm"]))[:top_oracle]:
        key = tuple(np.asarray(row["transform"]).round(7).reshape(-1))
        selected[key] = row
    return list(selected.values())


def stratified_case_folds(
    groups: dict[tuple[str, str], list[dict]],
    folds: int,
    seed: int,
    group_by_case: dict[str, str] | None = None,
) -> list[set[str]]:
    chirality: dict[str, int] = {}
    for (case_id, _), rows in groups.items():
        if rows:
            chirality[case_id] = int(rows[0].get("ground_truth_chirality", 1))
    members_by_group: dict[str, set[str]] = defaultdict(set)
    chiralities_by_group: dict[str, list[int]] = defaultdict(list)
    for case_id, sign in chirality.items():
        group_id = group_by_case.get(case_id, case_id) if group_by_case else case_id
        members_by_group[group_id].add(case_id)
        chiralities_by_group[group_id].append(sign)
    if folds < 2 or folds > len(members_by_group):
        raise ValueError(f"folds must be in [2, {len(members_by_group)}]")

    group_chirality = {
        group_id: Counter(signs).most_common(1)[0][0]
        for group_id, signs in chiralities_by_group.items()
    }
    rng = np.random.default_rng(seed)
    split = [set() for _ in range(folds)]
    for sign in (-1, 1):
        group_ids = sorted(
            group_id for group_id, value in group_chirality.items() if value == sign
        )
        rng.shuffle(group_ids)
        for group_id in group_ids:
            fold_index = min(range(folds), key=lambda index: len(split[index]))
            split[fold_index].update(members_by_group[group_id])
    return split


def cbct_groups_by_case(records, case_ids: set[str]) -> dict[str, str]:
    """Map cases to decompressed CBCT hashes, including gzip-repacked duplicates."""
    path_by_case: dict[str, Path] = {}
    for record in records:
        if record.case_id in case_ids and record.cbct_path:
            path_by_case.setdefault(record.case_id, Path(record.cbct_path))
    hash_by_path = {
        path: sha256_nifti_payload(path) for path in sorted(set(path_by_case.values()))
    }
    return {case_id: hash_by_path[path] for case_id, path in path_by_case.items()}


def feature_groups(groups: dict[tuple[str, str], list[dict]], priors) -> dict[tuple[str, str], np.ndarray]:
    return {
        key: np.stack([candidate_features(row, priors[key[1]]) for row in rows])
        for key, rows in groups.items()
    }


def fit_model(
    groups: dict[tuple[str, str], list[dict]],
    features: dict[tuple[str, str], np.ndarray],
    training_cases: set[str],
    top_unsupervised: int,
    top_oracle: int,
    min_samples_leaf: int,
    max_features: float,
    trees: int,
    seed: int,
) -> ExtraTreesRegressor:
    x_train: list[np.ndarray] = []
    y_train: list[float] = []
    weights: list[float] = []
    for key, rows in groups.items():
        if key[0] not in training_cases:
            continue
        selected = training_subset(rows, top_unsupervised, top_oracle)
        index_by_id = {id(row): index for index, row in enumerate(rows)}
        group_weight = 1.0 / max(len(selected), 1)
        for row in selected:
            x_train.append(features[key][index_by_id[id(row)]])
            y_train.append(np.log1p(float(row["mean_tre_mm"])))
            weights.append(group_weight)
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(np.stack(x_train), np.asarray(y_train), sample_weight=np.asarray(weights))
    return model


def evaluate_groups(
    model: ExtraTreesRegressor,
    groups: dict[tuple[str, str], list[dict]],
    features: dict[tuple[str, str], np.ndarray],
    cases: set[str],
    eval_top_candidates: int,
) -> tuple[list[float], list[dict[str, object]]]:
    errors: list[float] = []
    selections: list[dict[str, object]] = []
    for (case_id, jaw), rows in sorted(groups.items()):
        if case_id not in cases:
            continue
        candidate_indices = sorted(
            range(len(rows)), key=lambda index: float(rows[index]["selection_score_mm"])
        )[:eval_top_candidates]
        predicted = np.expm1(model.predict(features[(case_id, jaw)][candidate_indices]))
        local_index = int(np.argmin(predicted))
        index = candidate_indices[local_index]
        selected = rows[index]
        oracle = min(rows, key=lambda row: float(row["mean_tre_mm"]))
        error = float(selected["mean_tre_mm"])
        errors.append(error)
        selections.append(
            {
                "case_id": case_id,
                "jaw": jaw,
                "predicted_tre_mm": float(predicted[local_index]),
                "mean_tre_mm": error,
                "original_unsupervised_rank": int(selected["unsupervised_rank"]),
                "source_variant": selected["source_variant"],
                "target": selected["target"],
                "method": selected["method"],
                "candidate_run": selected.get("candidate_run", ""),
                "oracle_mean_tre_mm": float(oracle["mean_tre_mm"]),
                "oracle_original_rank": int(oracle["unsupervised_rank"]),
            }
        )
    return errors, selections


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_cases = set(args.eval_case_ids)
    records = load_manifest(args.manifest)
    loaded_train_groups = load_candidate_groups(args.train_runs)
    loaded_eval_groups = load_candidate_groups(args.eval_runs) if args.eval_runs else {}
    candidate_cases = (
        {key[0] for key in loaded_train_groups}
        | {key[0] for key in loaded_eval_groups}
        | eval_cases
    )
    if args.cv_grouping == "cbct":
        group_by_case = cbct_groups_by_case(records, candidate_cases)
    else:
        group_by_case = {case_id: case_id for case_id in candidate_cases}

    eval_group_ids = {group_by_case.get(case_id, case_id) for case_id in eval_cases}
    excluded_train_cases = {
        case_id
        for case_id in candidate_cases
        if group_by_case.get(case_id, case_id) in eval_group_ids
    }
    train_groups = {
        key: rows
        for key, rows in loaded_train_groups.items()
        if key[0] not in excluded_train_cases
    }
    eval_groups = loaded_eval_groups
    all_train_cases = {key[0] for key in train_groups}
    folds = stratified_case_folds(train_groups, args.folds, args.seed, group_by_case)

    fold_features = []
    for validation_cases in folds:
        excluded = eval_cases | validation_cases
        priors = {jaw: fit_rotation_prior(records, jaw, excluded) for jaw in ("upper", "lower")}
        fold_features.append(feature_groups(train_groups, priors))

    grid_rows: list[dict[str, object]] = []
    configurations = itertools.product(
        args.top_unsupervised,
        args.top_oracle,
        args.min_samples_leaf,
        args.max_features,
    )
    for config_index, (top_fit, top_truth, leaf, max_features) in enumerate(configurations, 1):
        errors_by_eval_top = {value: [] for value in args.eval_top_candidates}
        for fold_index, validation_cases in enumerate(folds):
            model = fit_model(
                train_groups,
                fold_features[fold_index],
                all_train_cases - validation_cases,
                top_fit,
                top_truth,
                leaf,
                max_features,
                args.cv_trees,
                args.seed + fold_index,
            )
            for eval_top in args.eval_top_candidates:
                fold_errors, _ = evaluate_groups(
                    model, train_groups, fold_features[fold_index], validation_cases, eval_top
                )
                errors_by_eval_top[eval_top].extend(fold_errors)
        summaries = []
        for eval_top, errors in errors_by_eval_top.items():
            values = np.asarray(errors)
            row = {
                "top_unsupervised": top_fit,
                "top_oracle": top_truth,
                "min_samples_leaf": leaf,
                "max_features": max_features,
                "eval_top_candidates": eval_top,
                "mean_tre_mm": float(values.mean()),
                "median_tre_mm": float(np.median(values)),
                "p90_tre_mm": float(np.quantile(values, 0.9)),
            }
            grid_rows.append(row)
            summaries.append(f"eval={eval_top}:{values.mean():.3f}")
        print(
            f"[{config_index}] fit={top_fit} truth={top_truth} leaf={leaf} "
            f"features={max_features:.2f} " + " ".join(summaries)
        )
    grid_rows.sort(key=lambda row: (row["mean_tre_mm"], row["p90_tre_mm"]))
    write_csv(args.output_dir / "cv_grid.csv", grid_rows)
    best = grid_rows[0]
    (args.output_dir / "best_config.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )

    oof_errors: list[float] = []
    oof_selections: list[dict[str, object]] = []
    for fold_index, validation_cases in enumerate(folds):
        model = fit_model(
            train_groups,
            fold_features[fold_index],
            all_train_cases - validation_cases,
            int(best["top_unsupervised"]),
            int(best["top_oracle"]),
            int(best["min_samples_leaf"]),
            float(best["max_features"]),
            args.cv_trees,
            args.seed + fold_index,
        )
        errors, selections = evaluate_groups(
            model,
            train_groups,
            fold_features[fold_index],
            validation_cases,
            int(best["eval_top_candidates"]),
        )
        for row in selections:
            row["fold"] = fold_index
            row["cbct_group"] = group_by_case.get(str(row["case_id"]), str(row["case_id"]))
        oof_errors.extend(errors)
        oof_selections.extend(selections)
    write_csv(args.output_dir / "oof_evaluation.csv", oof_selections)

    final_priors = {
        jaw: fit_rotation_prior(records, jaw, eval_cases) for jaw in ("upper", "lower")
    }
    final_train_features = feature_groups(train_groups, final_priors)
    final_eval_features = feature_groups(eval_groups, final_priors)
    final_model = fit_model(
        train_groups,
        final_train_features,
        all_train_cases,
        int(best["top_unsupervised"]),
        int(best["top_oracle"]),
        int(best["min_samples_leaf"]),
        float(best["max_features"]),
        args.final_trees,
        args.seed,
    )
    errors, selections = evaluate_groups(
        final_model,
        eval_groups,
        final_eval_features,
        eval_cases,
        int(best["eval_top_candidates"]),
    )
    write_csv(args.output_dir / "evaluation.csv", selections)
    importances = sorted(
        (
            {"feature": name, "importance": float(value)}
            for name, value in zip(FEATURE_NAMES, final_model.feature_importances_)
        ),
        key=lambda row: row["importance"],
        reverse=True,
    )
    write_csv(args.output_dir / "feature_importance.csv", importances)
    joblib.dump(
        {
            "model": final_model,
            "feature_names": FEATURE_NAMES,
            "eval_cases": sorted(eval_cases),
            "cv_grouping": args.cv_grouping,
            "case_to_group": group_by_case,
            "best_config": best,
        },
        args.output_dir / "candidate_reranker.joblib",
    )
    oof_values = np.asarray(oof_errors)
    print(f"Best grouped-CV configuration: {best}")
    print(
        f"Leakage-safe OOF: mean={oof_values.mean():.3f} mm; "
        f"median={np.median(oof_values):.3f} mm; "
        f"p90={np.quantile(oof_values, 0.9):.3f} mm"
    )
    if not errors:
        return
    values = np.asarray(errors)
    print(
        f"Held-out dev: mean={values.mean():.3f} mm; "
        f"median={np.median(values):.3f} mm; p90={np.quantile(values, 0.9):.3f} mm"
    )


if __name__ == "__main__":
    main()
