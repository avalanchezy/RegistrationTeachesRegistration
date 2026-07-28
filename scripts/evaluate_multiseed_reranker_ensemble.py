from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweep_multimodal_reranker import (
    estimator_args,
    exact_fallback_errors,
    load_exact_errors,
    summarize,
    write_csv,
)
from scripts.sweep_joint_pair_reranker import fit_pair_prior, pair_geometry
from scripts.train_semisupervised_candidate_reranker import (
    add_pseudo_targets,
    cbct_groups_by_case,
    feature_groups,
    fit_model,
    inverse_tre_target,
    leakage_safe_pseudo_keys,
    stratified_folds,
    top_indices,
)
from task2reg.candidate_learning import (
    REGISTRATION_TARGET_NAMES,
    enrich_candidate_registration_metrics,
    is_opposite_axial_target,
    load_candidate_groups,
)
from task2reg.data import load_manifest
from task2reg.priors import fit_rotation_prior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Candidate-level ensemble of strict CBCT-grouped OOF rerankers."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labeled-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--pseudo-runs", type=Path, nargs="*", default=())
    parser.add_argument("--pseudo-labels", type=Path)
    parser.add_argument("--pseudo-weight", type=float, default=0.0)
    parser.add_argument("--pseudo-top-unsupervised", type=int, default=20)
    parser.add_argument("--pseudo-top-consensus", type=int, default=8)
    parser.add_argument(
        "--pseudo-target-mode",
        choices=("distance", "additive", "quadrature"),
        default="distance",
    )
    parser.add_argument("--min-pseudo-confidence", type=float, default=0.0)
    parser.add_argument("--max-pseudo-full-median-mm", type=float, default=float("inf"))
    parser.add_argument("--max-pseudo-full-p90-mm", type=float, default=float("inf"))
    parser.add_argument("--require-pseudo-full-geometry", action="store_true")
    parser.add_argument("--sample-points", type=int, default=5000)
    parser.add_argument("--pseudo-target-seed", type=int, default=20260715)
    parser.add_argument("--exact-template-pseudo-weight-multiplier", type=float, default=0.0)
    parser.add_argument("--geometry-pseudo-weight-multiplier", type=float, default=0.0)
    parser.add_argument("--threshold-pseudo-weight-multiplier", type=float, default=0.0)
    parser.add_argument("--cross-modal-pseudo-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--include-threshold-pseudo-in-cv", action="store_true")
    parser.add_argument("--include-geometry-pseudo-in-cv", action="store_true")
    parser.add_argument(
        "--exclude-cross-modal-pseudo-in-cv",
        action="store_true",
        help="Disable the historical default that admits label-free cross-modal teachers.",
    )
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
        "--optimization-target",
        choices=REGISTRATION_TARGET_NAMES,
        default="mean_tre_mm",
    )
    parser.add_argument(
        "--tree-criterion",
        choices=("squared_error", "absolute_error", "friedman_mse", "poisson"),
        default="squared_error",
    )
    parser.add_argument("--tree-max-depth", type=int, default=0)
    parser.add_argument("--cv-trees", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--resume-seeds",
        action="store_true",
        help="Reuse atomically saved per-seed OOF predictions from the output directory.",
    )
    parser.add_argument("--hgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgb-l2", type=float, default=1.0)
    parser.add_argument("--hgb-ensemble-leaf-nodes", type=int, nargs="*", default=())
    parser.add_argument("--hgb-early-stopping", action="store_true")
    parser.add_argument("--group-context-features", action="store_true")
    parser.add_argument("--roi-view-feature", action="store_true")
    parser.add_argument("--modality-features", action="store_true")
    parser.add_argument("--balance-candidate-runs", action="store_true")
    parser.add_argument("--exclude-upper-opposite-axial", action="store_true")
    parser.add_argument("--joint-pair-selection", action="store_true")
    parser.add_argument("--pair-top-k", type=int, nargs="+", default=(2, 3, 5, 8, 10))
    parser.add_argument(
        "--angle-weights", type=float, nargs="+", default=(0.0, 0.01, 0.025, 0.05, 0.075, 0.1)
    )
    parser.add_argument(
        "--translation-weights", type=float, nargs="+", default=(0.0, 0.025, 0.05, 0.075)
    )
    parser.add_argument("--allow-chirality-mismatch", action="store_true")
    return parser.parse_args()


def candidate_pool(rows: list[dict], jaw: str, args: argparse.Namespace) -> list[int]:
    pool = list(range(len(rows)))
    if args.exclude_upper_opposite_axial and jaw == "upper":
        filtered = [index for index in pool if not is_opposite_axial_target(rows[index], jaw)]
        if filtered:
            pool = filtered
    subset = [rows[index] for index in pool]
    local = top_indices(
        subset,
        "selection_score_mm",
        args.eval_top_candidates,
        args.balance_candidate_runs,
    )
    return [pool[index] for index in local]


def fractional_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / float(max(len(values) - 1, 1))


def aggregate_predictions(predictions: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return np.mean(predictions, axis=0)
    if method == "median":
        return np.median(predictions, axis=0)
    if method == "rank_mean":
        return np.mean(np.stack([fractional_ranks(row) for row in predictions]), axis=0)
    if method == "vote":
        votes = np.zeros(predictions.shape[1], dtype=np.float64)
        for row in predictions:
            votes[int(np.argmin(row))] += 1.0
        return -votes + 1e-6 * np.mean(predictions, axis=0)
    raise ValueError(f"Unknown aggregation method: {method}")


def selected_row(
    case_id: str,
    jaw: str,
    rows: list[dict],
    index: int,
    aggregate_score: float,
    method: str,
) -> dict[str, object]:
    selected = rows[index]
    oracle = min(rows, key=lambda row: float(row["mean_tre_mm"]))
    return {
        "case_id": case_id,
        "jaw": jaw,
        "ensemble_method": method,
        "ensemble_score": aggregate_score,
        "optimization_target": selected.get(
            "optimization_target", "mean_tre_mm"
        ),
        "mean_tre_mm": float(selected["mean_tre_mm"]),
        "translation_error_mm": selected.get("translation_error_mm", ""),
        "rotation_error_deg": selected.get("rotation_error_deg", ""),
        "official_balanced_error": selected.get("official_balanced_error", ""),
        "original_unsupervised_rank": int(selected["unsupervised_rank"]),
        "source_variant": selected["source_variant"],
        "target": selected["target"],
        "method": selected["method"],
        "candidate_run": selected.get("candidate_run", ""),
        "full_geometry_available": int("full_distance_p90_mm" in selected),
        "oracle_mean_tre_mm": float(oracle["mean_tre_mm"]),
        "transform_key": ",".join(
            f"{value:.6f}"
            for value in np.asarray(selected["transform"], dtype=np.float64).reshape(-1)
        ),
    }


def select_ensemble_pair(
    upper: list[dict[str, object]],
    lower: list[dict[str, object]],
    priors,
    top_k: int,
    angle_weight: float,
    translation_weight: float,
    allow_chirality_mismatch: bool,
) -> dict[str, object]:
    pairs = []
    for upper_item, lower_item in itertools.product(upper[:top_k], lower[:top_k]):
        if (
            not allow_chirality_mismatch
            and int(upper_item["row"].get("chirality", 1))
            != int(lower_item["row"].get("chirality", 1))
        ):
            continue
        geometry = [pair_geometry(upper_item, lower_item, prior) for prior in priors]
        angle = float(np.mean([value[0] for value in geometry]))
        translation = float(np.mean([value[1] for value in geometry]))
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
        if top_k < max(len(upper), len(lower)):
            return select_ensemble_pair(
                upper,
                lower,
                priors,
                max(len(upper), len(lower)),
                angle_weight,
                translation_weight,
                allow_chirality_mismatch,
            )
        raise RuntimeError("No chirality-consistent ensemble candidate pair")
    return min(pairs, key=lambda item: float(item["objective"]))


def atomic_joblib_dump(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary, compress=3)
    temporary.replace(path)


def validate_prediction_layout(
    predictions: dict[tuple[str, str], np.ndarray],
    groups: dict[tuple[str, str], list[dict]],
    checkpoint: Path,
) -> None:
    if set(predictions) != set(groups) or any(
        len(predictions[key]) != len(groups[key]) for key in groups
    ):
        raise RuntimeError(f"Seed checkpoint does not match candidate pool: {checkpoint}")


def main() -> None:
    args = parse_args()
    if args.roi_view_feature and not args.group_context_features:
        raise ValueError("--roi-view-feature requires --group-context-features")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(args.manifest)
    groups = load_candidate_groups(args.labeled_runs)
    enrich_candidate_registration_metrics(groups, records)
    for rows in groups.values():
        for row in rows:
            row["optimization_target"] = args.optimization_target
    if args.optimization_target != "mean_tre_mm" and args.pseudo_weight > 0.0:
        raise ValueError("Official-metric optimization cannot use pseudo TRE targets")
    train_cases = {key[0] for key in groups}
    pseudo_groups: dict[tuple[str, str], list[dict]] = {}
    pseudo_info: dict[tuple[str, str], dict[str, object]] = {}
    if args.pseudo_weight > 0.0:
        if not args.pseudo_runs or args.pseudo_labels is None:
            raise ValueError(
                "--pseudo-runs and --pseudo-labels are required when --pseudo-weight is positive"
            )
        pseudo_groups = load_candidate_groups(args.pseudo_runs)
        if args.require_pseudo_full_geometry:
            pseudo_groups = {
                key: [
                    row
                    for row in rows
                    if float(row.get("full_geometry_available", 0.0)) > 0.0
                ]
                for key, rows in pseudo_groups.items()
            }
            pseudo_groups = {key: rows for key, rows in pseudo_groups.items() if rows}
        pseudo_labels = json.loads(args.pseudo_labels.read_text(encoding="utf-8"))
        pseudo_info = add_pseudo_targets(
            pseudo_groups,
            pseudo_labels,
            args.sample_points,
            args.pseudo_target_seed,
            args.pseudo_target_mode,
            args.min_pseudo_confidence,
            args.max_pseudo_full_median_mm,
            args.max_pseudo_full_p90_mm,
        )
        if not pseudo_info:
            raise RuntimeError("No accepted pseudo labels matched the pseudo candidate runs")
        pseudo_groups = {key: pseudo_groups[key] for key in pseudo_info}
    relevant_cases = train_cases | {key[0] for key in pseudo_info}
    relevant_cases |= {
        str(info.get("source_labeled_case_id", ""))
        for info in pseudo_info.values()
        if info.get("source_labeled_case_id")
    }
    case_group = cbct_groups_by_case(records, relevant_cases, args.cbct_hash_cache)
    exact = load_exact_errors(args.exact_loo)
    model_args = estimator_args(
        args,
        args.top_unsupervised,
        args.top_oracle,
        args.min_samples_leaf,
        args.max_features,
        args.model_scope,
        args.model_type,
        args.target_transform,
        args.optimization_target,
        args.tree_criterion,
        args.tree_max_depth,
    )
    model_args.eval_top_candidates = args.eval_top_candidates
    model_args.pseudo_top_unsupervised = args.pseudo_top_unsupervised
    model_args.pseudo_top_consensus = args.pseudo_top_consensus
    model_args.exact_template_weight_multiplier = (
        args.exact_template_pseudo_weight_multiplier
    )
    model_args.geometry_pseudo_weight_multiplier = (
        args.geometry_pseudo_weight_multiplier
    )
    model_args.threshold_pseudo_weight_multiplier = (
        args.threshold_pseudo_weight_multiplier
    )
    model_args.cross_modal_pseudo_weight_multiplier = (
        args.cross_modal_pseudo_weight_multiplier
    )

    pseudo_checkpoint_config = {
        "pseudo_weight": args.pseudo_weight,
        "pseudo_groups": len(pseudo_info),
        "pseudo_target_mode": args.pseudo_target_mode,
        "pseudo_target_seed": args.pseudo_target_seed,
        "pseudo_top_unsupervised": args.pseudo_top_unsupervised,
        "pseudo_top_consensus": args.pseudo_top_consensus,
        "min_pseudo_confidence": args.min_pseudo_confidence,
        "max_pseudo_full_median_mm": args.max_pseudo_full_median_mm,
        "max_pseudo_full_p90_mm": args.max_pseudo_full_p90_mm,
        "require_pseudo_full_geometry": args.require_pseudo_full_geometry,
        "exact_template_pseudo_weight_multiplier": (
            args.exact_template_pseudo_weight_multiplier
        ),
        "geometry_pseudo_weight_multiplier": args.geometry_pseudo_weight_multiplier,
        "threshold_pseudo_weight_multiplier": args.threshold_pseudo_weight_multiplier,
        "cross_modal_pseudo_weight_multiplier": args.cross_modal_pseudo_weight_multiplier,
        "include_threshold_pseudo_in_cv": args.include_threshold_pseudo_in_cv,
        "include_geometry_pseudo_in_cv": args.include_geometry_pseudo_in_cv,
        "include_cross_modal_pseudo_in_cv": not args.exclude_cross_modal_pseudo_in_cv,
    }
    checkpoint_config = {
        "pseudo_configuration": pseudo_checkpoint_config,
        "folds": args.folds,
        "top_unsupervised": args.top_unsupervised,
        "top_oracle": args.top_oracle,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "eval_top_candidates": args.eval_top_candidates,
        "model_scope": args.model_scope,
        "model_type": args.model_type,
        "target_transform": args.target_transform,
        "optimization_target": args.optimization_target,
        "tree_criterion": args.tree_criterion,
        "tree_max_depth": args.tree_max_depth,
        "cv_trees": args.cv_trees,
        "group_context_features": args.group_context_features,
        "roi_view_feature": args.roi_view_feature,
        "modality_features": args.modality_features,
        "balance_candidate_runs": args.balance_candidate_runs,
        "exclude_upper_opposite_axial": args.exclude_upper_opposite_axial,
        "candidate_layout": sorted(
            f"{case_id}:{jaw}:{len(rows)}" for (case_id, jaw), rows in groups.items()
        ),
    }

    seed_predictions: list[dict[tuple[str, str], np.ndarray]] = []
    seed_pair_priors = []
    for seed_index, seed in enumerate(args.seeds, start=1):
        checkpoint = args.output_dir / f"seed_{seed}_oof.joblib"
        partial_checkpoint = args.output_dir / f"seed_{seed}_oof.partial.joblib"
        if args.resume_seeds and checkpoint.is_file():
            payload = joblib.load(checkpoint)
            saved_config = payload.get("checkpoint_configuration")
            if saved_config is not None and saved_config != checkpoint_config:
                raise RuntimeError(f"Seed checkpoint configuration mismatch: {checkpoint}")
            if args.pseudo_weight > 0.0 and payload.get("pseudo_configuration") != (
                pseudo_checkpoint_config
            ):
                raise RuntimeError(
                    f"Seed checkpoint pseudo configuration mismatch: {checkpoint}"
                )
            predictions = payload["predictions"]
            validate_prediction_layout(predictions, groups, checkpoint)
            seed_predictions.append(predictions)
            seed_pair_priors.append(payload["pair_priors"])
            print(
                f"[{seed_index}/{len(args.seeds)}] reused strict OOF seed {seed}",
                flush=True,
            )
            continue
        folds = stratified_folds(groups, args.folds, seed, case_group)
        predictions = {
            key: np.full(len(rows), np.nan, dtype=np.float64) for key, rows in groups.items()
        }
        pair_priors: dict[str, object] = {}
        completed_cases: set[str] = set()
        if args.resume_seeds and partial_checkpoint.is_file():
            payload = joblib.load(partial_checkpoint)
            if payload.get("checkpoint_configuration") != checkpoint_config:
                raise RuntimeError(
                    f"Partial seed checkpoint configuration mismatch: {partial_checkpoint}"
                )
            predictions = payload["predictions"]
            validate_prediction_layout(predictions, groups, partial_checkpoint)
            pair_priors = payload["pair_priors"]
            completed_cases = set(payload["completed_cases"])
            if not completed_cases <= train_cases:
                raise RuntimeError(
                    f"Partial seed checkpoint has unknown cases: {partial_checkpoint}"
                )
            print(
                f"[{seed_index}/{len(args.seeds)}] resumed seed {seed} after "
                f"{len(completed_cases)}/{len(train_cases)} cases",
                flush=True,
            )
        for fold_index, validation_cases in enumerate(folds):
            validation_cases = set(validation_cases)
            if validation_cases <= completed_cases:
                continue
            if validation_cases & completed_cases:
                raise RuntimeError(
                    f"Partial seed checkpoint splits a validation fold: {partial_checkpoint}"
                )
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
            pseudo_features = feature_groups(
                pseudo_groups,
                priors,
                args.group_context_features,
                args.roi_view_feature,
                args.modality_features,
            )
            allowed_pseudo_keys = leakage_safe_pseudo_keys(
                pseudo_info,
                validation_cases,
                case_group,
                include_threshold=args.include_threshold_pseudo_in_cv,
                include_geometry=args.include_geometry_pseudo_in_cv,
                include_cross_modal=not args.exclude_cross_modal_pseudo_in_cv,
            )
            model = fit_model(
                groups,
                features,
                train_cases - validation_cases,
                pseudo_groups,
                pseudo_features,
                pseudo_info,
                args.pseudo_weight,
                model_args,
                seed + fold_index,
                allowed_pseudo_keys,
            )
            pair_prior = fit_pair_prior(records, validation_cases)
            for case_id in validation_cases:
                pair_priors[case_id] = pair_prior
            for key, rows in groups.items():
                if key[0] not in validation_cases:
                    continue
                indices = candidate_pool(rows, key[1], args)
                estimator = model[key[1]] if isinstance(model, dict) else model
                predictions[key][indices] = inverse_tre_target(
                    estimator.predict(features[key][indices]), args.target_transform
                )
            completed_cases.update(validation_cases)
            if args.resume_seeds:
                atomic_joblib_dump(
                    {
                        "predictions": predictions,
                        "pair_priors": pair_priors,
                        "completed_cases": sorted(completed_cases),
                        "checkpoint_configuration": checkpoint_config,
                    },
                    partial_checkpoint,
                )
        if any(np.all(np.isnan(values)) for values in predictions.values()):
            raise RuntimeError(f"Seed {seed} did not predict every candidate group")
        atomic_joblib_dump(
            {
                "predictions": predictions,
                "pair_priors": pair_priors,
                "pseudo_configuration": pseudo_checkpoint_config,
                "checkpoint_configuration": checkpoint_config,
            },
            checkpoint,
        )
        partial_checkpoint.unlink(missing_ok=True)
        seed_predictions.append(predictions)
        seed_pair_priors.append(pair_priors)
        print(f"[{seed_index}/{len(args.seeds)}] completed strict OOF seed {seed}", flush=True)

    methods = ("mean", "median", "rank_mean", "vote")
    policy_rows: list[dict[str, object]] = []
    candidate_score_rows: list[dict[str, object]] = []
    aggregated_candidates: dict[str, dict[str, dict[str, list[dict[str, object]]]]] = {
        method: {} for method in methods
    }
    summary: dict[str, object] = {
        "cases": len(train_cases),
        "jaws": len(groups),
        "cbct_groups": len({case_group[case_id] for case_id in train_cases}),
        "relevant_cbct_groups": len(set(case_group.values())),
        "seeds": args.seeds,
        "configuration": {
            "top_unsupervised": args.top_unsupervised,
            "top_oracle": args.top_oracle,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "eval_top_candidates": args.eval_top_candidates,
            "model_scope": args.model_scope,
            "model_type": args.model_type,
        "target_transform": args.target_transform,
        "optimization_target": args.optimization_target,
            "tree_criterion": args.tree_criterion,
            "tree_max_depth": args.tree_max_depth,
            "cv_trees": args.cv_trees,
            **pseudo_checkpoint_config,
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
            items = sorted(
                (
                    {
                        "row": rows[int(index)],
                        "prediction_mm": float(score),
                        "candidate_index": int(index),
                    }
                    for index, score in zip(available, scores)
                ),
                key=lambda item: float(item["prediction_mm"]),
            )
            aggregated_candidates[method].setdefault(key[0], {})[key[1]] = items
            for item in items:
                candidate_score_rows.append(
                    {
                        "case_id": key[0],
                        "jaw": key[1],
                        "ensemble_method": method,
                        "candidate_index": item["candidate_index"],
                        "ensemble_score": item["prediction_mm"],
                        "mean_tre_mm": float(item["row"]["mean_tre_mm"]),
                        "candidate_run": item["row"].get("candidate_run", ""),
                    }
                )
            local_index = int(np.argmin(scores))
            index = int(available[local_index])
            rows_out.append(
                selected_row(key[0], key[1], rows, index, float(scores[local_index]), method)
            )
        raw = summarize([float(row["mean_tre_mm"]) for row in rows_out])
        combined = summarize(exact_fallback_errors(rows_out, exact))
        summary["policies"][method] = {"raw": raw, "exact_fallback": combined}
        write_csv(args.output_dir / f"oof_{method}.csv", rows_out)
        for row in rows_out:
            policy_rows.append(row)
    write_csv(args.output_dir / "all_policy_selections.csv", policy_rows)
    write_csv(args.output_dir / "candidate_ensemble_scores.csv", candidate_score_rows)

    if args.joint_pair_selection:
        joint_grid = []
        best_key = None
        best_rows = []
        for method, top_k, angle_weight, translation_weight in itertools.product(
            methods, args.pair_top_k, args.angle_weights, args.translation_weights
        ):
            rows_out = []
            for case_id, jaws in sorted(aggregated_candidates[method].items()):
                if set(jaws) != {"upper", "lower"}:
                    continue
                pair = select_ensemble_pair(
                    jaws["upper"],
                    jaws["lower"],
                    [payload[case_id] for payload in seed_pair_priors],
                    top_k,
                    angle_weight,
                    translation_weight,
                    args.allow_chirality_mismatch,
                )
                for jaw in ("upper", "lower"):
                    item = pair[jaw]
                    row = selected_row(
                        case_id,
                        jaw,
                        groups[(case_id, jaw)],
                        int(item["candidate_index"]),
                        float(item["prediction_mm"]),
                        method,
                    )
                    row.update(
                        pair_objective=pair["objective"],
                        pair_relative_angle_deg=pair["relative_angle_deg"],
                        pair_translation_deviation_mm=pair[
                            "relative_translation_deviation_mm"
                        ],
                    )
                    rows_out.append(row)
            raw = summarize([float(row["mean_tre_mm"]) for row in rows_out])
            combined = summarize(exact_fallback_errors(rows_out, exact))
            grid_row = {
                "ensemble_method": method,
                "pair_top_k": top_k,
                "angle_weight_mm_per_deg": angle_weight,
                "translation_weight": translation_weight,
                **raw,
                "exact_fallback_mean_tre_mm": combined["mean_tre_mm"],
                "exact_fallback_median_tre_mm": combined["median_tre_mm"],
                "exact_fallback_p90_tre_mm": combined["p90_tre_mm"],
                "exact_fallback_max_tre_mm": combined["max_tre_mm"],
            }
            joint_grid.append(grid_row)
            selection_key = (
                combined["mean_tre_mm"],
                combined["p90_tre_mm"],
                combined["max_tre_mm"],
                raw["mean_tre_mm"],
            )
            if best_key is None or selection_key < best_key:
                best_key = selection_key
                best_rows = rows_out
        joint_grid.sort(
            key=lambda row: (
                float(row["exact_fallback_mean_tre_mm"]),
                float(row["exact_fallback_p90_tre_mm"]),
                float(row["exact_fallback_max_tre_mm"]),
            )
        )
        write_csv(args.output_dir / "joint_grid.csv", joint_grid)
        write_csv(args.output_dir / "best_joint_oof.csv", best_rows)
        summary["best_joint"] = joint_grid[0]
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
