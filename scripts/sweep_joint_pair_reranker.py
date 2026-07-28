from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweep_multimodal_reranker import load_exact_errors, summarize
from scripts.train_semisupervised_candidate_reranker import (
    cbct_groups_by_case,
    feature_groups,
    fit_model,
    inverse_tre_target,
    stratified_folds,
    top_indices,
)
from task2reg.candidate_learning import is_opposite_axial_target, load_candidate_groups
from task2reg.data import CaseRecord, load_manifest
from task2reg.priors import fit_rotation_prior, proper_protocol_rotation


@dataclass(frozen=True)
class PairPrior:
    relative_rotation: np.ndarray
    relative_translation: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict CBCT-grouped OOF sweep for joint upper/lower candidate selection."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labeled-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exact-loo", type=Path)
    parser.add_argument("--cbct-hash-cache", type=Path)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--top-unsupervised", type=int, default=20)
    parser.add_argument("--top-oracle", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument("--max-features", type=float, default=0.4)
    parser.add_argument("--eval-top-candidates", type=int, default=30)
    parser.add_argument("--pair-top-k", type=int, nargs="+", default=(1, 3, 5, 10))
    parser.add_argument(
        "--angle-weights", type=float, nargs="+", default=(0.0, 0.01, 0.025, 0.05, 0.1)
    )
    parser.add_argument(
        "--translation-weights", type=float, nargs="+", default=(0.0,)
    )
    parser.add_argument(
        "--model-type",
        choices=("extra_trees", "random_forest", "hist_gradient_boosting"),
        default="extra_trees",
    )
    parser.add_argument(
        "--target-transform",
        choices=("log1p", "sqrt", "identity"),
        default="log1p",
    )
    parser.add_argument(
        "--tree-criterion",
        choices=("squared_error", "absolute_error", "friedman_mse", "poisson"),
        default="squared_error",
    )
    parser.add_argument("--tree-max-depth", type=int, default=0)
    parser.add_argument("--model-scope", choices=("jaw", "shared"), default="jaw")
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
    parser.add_argument(
        "--allow-chirality-mismatch",
        action="store_true",
        help="Permit physically inconsistent upper/lower reflection signs.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_pair_prior(records: list[CaseRecord], excluded_cases: set[str]) -> PairPrior:
    by_case: dict[str, dict[str, np.ndarray]] = {}
    for record in records:
        if (
            record.split != "Train-Labeled"
            or record.case_id in excluded_cases
            or not record.transform_path
        ):
            continue
        by_case.setdefault(record.case_id, {})[record.jaw] = np.load(record.transform_path)
    rotations = []
    translations = []
    for jaws in by_case.values():
        if set(jaws) != {"upper", "lower"}:
            continue
        upper = jaws["upper"]
        lower = jaws["lower"]
        upper_rotation = proper_protocol_rotation(upper[:3, :3])
        lower_rotation = proper_protocol_rotation(lower[:3, :3])
        rotations.append(upper_rotation.T @ lower_rotation)
        translations.append(upper[:3, 3] - lower[:3, 3])
    if len(rotations) < 3:
        raise ValueError("Need at least three paired labeled cases for a pair prior")
    return PairPrior(
        relative_rotation=Rotation.from_matrix(np.stack(rotations)).mean().as_matrix(),
        relative_translation=np.mean(np.stack(translations), axis=0),
    )


def model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_type=args.model_type,
        target_transform=args.target_transform,
        tree_criterion=args.tree_criterion,
        tree_max_depth=args.tree_max_depth,
        cv_trees=args.cv_trees,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
        hgb_learning_rate=args.hgb_learning_rate,
        hgb_max_leaf_nodes=args.hgb_max_leaf_nodes,
        hgb_l2=args.hgb_l2,
        hgb_ensemble_leaf_nodes=args.hgb_ensemble_leaf_nodes,
        hgb_early_stopping=args.hgb_early_stopping,
        jaw_specific_models=args.model_scope == "jaw",
        top_unsupervised=args.top_unsupervised,
        top_oracle=args.top_oracle,
        pseudo_top_unsupervised=0,
        pseudo_top_consensus=0,
        balance_candidate_runs=args.balance_candidate_runs,
        exact_template_weight_multiplier=1.0,
        geometry_pseudo_weight_multiplier=0.0,
        threshold_pseudo_weight_multiplier=0.0,
        exclude_upper_opposite_axial=args.exclude_upper_opposite_axial,
        eval_top_candidates=args.eval_top_candidates,
    )


def candidate_predictions(
    model: object,
    rows: list[dict],
    features: np.ndarray,
    jaw: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    candidate_pool = list(range(len(rows)))
    if args.exclude_upper_opposite_axial and jaw == "upper":
        filtered = [
            index
            for index in candidate_pool
            if not is_opposite_axial_target(rows[index], jaw)
        ]
        if filtered:
            candidate_pool = filtered
    if args.balance_candidate_runs:
        subset = [rows[index] for index in candidate_pool]
        local_indices = top_indices(
            subset, "selection_score_mm", args.eval_top_candidates, True
        )
        indices = [candidate_pool[index] for index in local_indices]
    else:
        indices = sorted(
            candidate_pool,
            key=lambda index: float(rows[index]["selection_score_mm"]),
        )[: args.eval_top_candidates]
    estimator = model[jaw] if isinstance(model, dict) else model
    predictions = inverse_tre_target(
        estimator.predict(features[indices]), args.target_transform
    )
    return sorted(
        (
            {
                "row": rows[index],
                "prediction_mm": float(prediction),
                "candidate_index": index,
            }
            for index, prediction in zip(indices, predictions)
        ),
        key=lambda item: float(item["prediction_mm"]),
    )


def pair_geometry(
    upper: dict[str, object], lower: dict[str, object], prior: PairPrior
) -> tuple[float, float]:
    upper_transform = np.asarray(upper["row"]["transform"], dtype=np.float64)
    lower_transform = np.asarray(lower["row"]["transform"], dtype=np.float64)
    relative = (
        proper_protocol_rotation(upper_transform[:3, :3]).T
        @ proper_protocol_rotation(lower_transform[:3, :3])
    )
    angle = float(
        np.degrees(
            Rotation.from_matrix(prior.relative_rotation.T @ relative).magnitude()
        )
    )
    translation = (
        upper_transform[:3, 3]
        - lower_transform[:3, 3]
        - prior.relative_translation
    )
    return angle, float(np.linalg.norm(translation))


def select_pair(
    upper: list[dict[str, object]],
    lower: list[dict[str, object]],
    prior: PairPrior,
    top_k: int,
    angle_weight: float,
    translation_weight: float,
    allow_chirality_mismatch: bool,
) -> dict[str, object]:
    pairs = []
    for upper_item, lower_item in itertools.product(upper[:top_k], lower[:top_k]):
        upper_row = upper_item["row"]
        lower_row = lower_item["row"]
        if (
            not allow_chirality_mismatch
            and int(upper_row.get("chirality", 1))
            != int(lower_row.get("chirality", 1))
        ):
            continue
        angle, translation = pair_geometry(upper_item, lower_item, prior)
        objective = (
            float(upper_item["prediction_mm"])
            + float(lower_item["prediction_mm"])
            + angle_weight * angle
            + translation_weight * translation
        )
        pairs.append(
            {
                "upper": upper_item,
                "lower": lower_item,
                "objective": objective,
                "relative_angle_deg": angle,
                "relative_translation_deviation_mm": translation,
            }
        )
    if not pairs:
        # A model can put opposite chiralities first; widen to the full evaluated pool.
        if top_k < max(len(upper), len(lower)):
            return select_pair(
                upper,
                lower,
                prior,
                max(len(upper), len(lower)),
                angle_weight,
                translation_weight,
                allow_chirality_mismatch,
            )
        raise RuntimeError("No chirality-consistent upper/lower candidate pair")
    return min(pairs, key=lambda item: float(item["objective"]))


def metrics_with_exact(
    selections: list[dict[str, object]],
    exact: dict[tuple[str, str], float],
) -> dict[str, float]:
    values = [
        exact.get(
            (str(row["case_id"]), str(row["jaw"])), float(row["mean_tre_mm"])
        )
        for row in selections
    ]
    return summarize(values)


def selection_row(
    case_id: str,
    jaw: str,
    item: dict[str, object],
    pair: dict[str, object],
) -> dict[str, object]:
    row = item["row"]
    return {
        "case_id": case_id,
        "jaw": jaw,
        "predicted_tre_mm": item["prediction_mm"],
        "mean_tre_mm": float(row["mean_tre_mm"]),
        "oracle_mean_tre_mm": "",
        "original_unsupervised_rank": int(row["unsupervised_rank"]),
        "source_variant": row["source_variant"],
        "target": row["target"],
        "method": row["method"],
        "candidate_run": row.get("candidate_run", ""),
        "transform_key": ",".join(
            f"{value:.6f}"
            for value in np.asarray(row["transform"], dtype=np.float64).reshape(-1)
        ),
        "pair_objective": pair["objective"],
        "pair_relative_angle_deg": pair["relative_angle_deg"],
        "pair_translation_deviation_mm": pair[
            "relative_translation_deviation_mm"
        ],
    }


def main() -> None:
    args = parse_args()
    if args.roi_view_feature and not args.group_context_features:
        raise ValueError("--roi-view-feature requires --group-context-features")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(args.manifest)
    groups = load_candidate_groups(args.labeled_runs)
    train_cases = {key[0] for key in groups}
    case_group = cbct_groups_by_case(records, train_cases, args.cbct_hash_cache)
    folds = stratified_folds(groups, args.folds, args.seed, case_group)
    exact = load_exact_errors(args.exact_loo)
    estimator_config = model_args(args)

    predictions: dict[str, dict[str, list[dict[str, object]]]] = {}
    priors_by_case: dict[str, PairPrior] = {}
    fold_by_case: dict[str, int] = {}
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
        model = fit_model(
            groups,
            features,
            train_cases - validation_cases,
            {},
            {},
            {},
            0.0,
            estimator_config,
            args.seed + fold_index,
        )
        pair_prior = fit_pair_prior(records, validation_cases)
        for case_id in validation_cases:
            priors_by_case[case_id] = pair_prior
            fold_by_case[case_id] = fold_index
            for jaw in ("upper", "lower"):
                key = (case_id, jaw)
                if key not in groups:
                    continue
                predictions.setdefault(case_id, {})[jaw] = candidate_predictions(
                    model, groups[key], features[key], jaw, args
                )

    independent_rows = []
    for case_id, jaws in sorted(predictions.items()):
        if set(jaws) != {"upper", "lower"}:
            continue
        pair = select_pair(
            jaws["upper"],
            jaws["lower"],
            priors_by_case[case_id],
            1,
            0.0,
            0.0,
            True,
        )
        for jaw in ("upper", "lower"):
            item = pair[jaw]
            out = selection_row(case_id, jaw, item, pair)
            out["fold"] = fold_by_case[case_id]
            out["oracle_mean_tre_mm"] = min(
                float(row["mean_tre_mm"]) for row in groups[(case_id, jaw)]
            )
            independent_rows.append(out)
    write_csv(args.output_dir / "independent_oof.csv", independent_rows)
    independent_raw = summarize(
        [float(row["mean_tre_mm"]) for row in independent_rows]
    )
    independent_exact = metrics_with_exact(independent_rows, exact)

    grid = []
    best_key = None
    best_rows: list[dict[str, object]] = []
    for top_k, angle_weight, translation_weight in itertools.product(
        args.pair_top_k, args.angle_weights, args.translation_weights
    ):
        rows_out = []
        for case_id, jaws in sorted(predictions.items()):
            if set(jaws) != {"upper", "lower"}:
                continue
            pair = select_pair(
                jaws["upper"],
                jaws["lower"],
                priors_by_case[case_id],
                top_k,
                angle_weight,
                translation_weight,
                args.allow_chirality_mismatch,
            )
            for jaw in ("upper", "lower"):
                out = selection_row(case_id, jaw, pair[jaw], pair)
                out["fold"] = fold_by_case[case_id]
                out["oracle_mean_tre_mm"] = min(
                    float(row["mean_tre_mm"]) for row in groups[(case_id, jaw)]
                )
                rows_out.append(out)
        raw = summarize([float(row["mean_tre_mm"]) for row in rows_out])
        combined = metrics_with_exact(rows_out, exact)
        grid_row = {
            "pair_top_k": top_k,
            "angle_weight_mm_per_deg": angle_weight,
            "translation_weight": translation_weight,
            **raw,
            "exact_fallback_mean_tre_mm": combined["mean_tre_mm"],
            "exact_fallback_median_tre_mm": combined["median_tre_mm"],
            "exact_fallback_p90_tre_mm": combined["p90_tre_mm"],
            "exact_fallback_max_tre_mm": combined["max_tre_mm"],
        }
        grid.append(grid_row)
        key = (
            combined["mean_tre_mm"],
            combined["p90_tre_mm"],
            combined["max_tre_mm"],
            raw["mean_tre_mm"],
        )
        if best_key is None or key < best_key:
            best_key = key
            best_rows = rows_out
        write_csv(args.output_dir / "grid.csv", grid)

    grid.sort(
        key=lambda row: (
            float(row["exact_fallback_mean_tre_mm"]),
            float(row["exact_fallback_p90_tre_mm"]),
            float(row["exact_fallback_max_tre_mm"]),
        )
    )
    write_csv(args.output_dir / "grid.csv", grid)
    write_csv(args.output_dir / "best_joint_oof.csv", best_rows)
    summary_payload = {
        "cases": len(predictions),
        "jaws": len(independent_rows),
        "cbct_groups": len(set(case_group.values())),
        "exact_fallback_jaws": len(exact),
        "independent_raw": independent_raw,
        "independent_exact_fallback": independent_exact,
        "best_joint": grid[0],
        "model": {
            "type": args.model_type,
            "target_transform": args.target_transform,
            "tree_criterion": args.tree_criterion,
            "tree_max_depth": args.tree_max_depth,
            "scope": args.model_scope,
            "top_unsupervised": args.top_unsupervised,
            "top_oracle": args.top_oracle,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "eval_top_candidates": args.eval_top_candidates,
            "cv_trees": args.cv_trees,
            "seed": args.seed,
        },
        "feature_configuration": {
            "group_context_features": args.group_context_features,
            "roi_view_feature": args.roi_view_feature,
            "modality_features": args.modality_features,
            "balance_candidate_runs": args.balance_candidate_runs,
            "exclude_upper_opposite_axial": args.exclude_upper_opposite_axial,
            "allow_chirality_mismatch": args.allow_chirality_mismatch,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
