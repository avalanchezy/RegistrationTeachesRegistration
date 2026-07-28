from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.candidate_learning import (
    FEATURE_NAMES,
    GROUP_FEATURE_NAMES,
    MULTIMODAL_FEATURE_NAMES,
    MULTIMODAL_GROUP_FEATURE_NAMES,
    MULTIMODAL_ROI_GROUP_FEATURE_NAMES,
    REGISTRATION_TARGET_NAMES,
    ROI_GROUP_FEATURE_NAMES,
    candidate_features,
    candidate_group_features,
    candidate_multimodal_features,
    candidate_multimodal_group_features,
    enrich_candidate_registration_metrics,
    is_opposite_axial_target,
    load_candidate_groups,
    registration_target_value,
)
from task2reg.data import apply_transform, load_ios_points, load_manifest
from task2reg.models import MeanRegressor
from task2reg.priors import fit_rotation_prior
from task2reg.template_transfer import sha256_nifti_payload


def transform_tre_target(values, mode: str):
    array = np.asarray(values, dtype=np.float64)
    if mode == "log1p":
        return np.log1p(array)
    if mode == "sqrt":
        return np.sqrt(np.maximum(array, 0.0))
    if mode == "identity":
        return array
    raise ValueError(f"Unknown TRE target transform: {mode}")


def inverse_tre_target(values, mode: str):
    array = np.asarray(values, dtype=np.float64)
    if mode == "log1p":
        return np.expm1(array)
    if mode == "sqrt":
        return np.square(np.maximum(array, 0.0))
    if mode == "identity":
        return np.maximum(array, 0.0)
    raise ValueError(f"Unknown TRE target transform: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a candidate TRE regressor with confidence-weighted unlabeled consensus."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labeled-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--pseudo-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--pseudo-labels", type=Path, required=True)
    parser.add_argument("--eval-runs", type=Path, nargs="*", default=())
    parser.add_argument("--eval-case-ids", nargs="*", default=())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pseudo-weights", type=float, nargs="+", default=(0.0, 0.1, 0.25, 0.5, 1.0))
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--top-unsupervised", type=int, default=20)
    parser.add_argument("--top-oracle", type=int, default=8)
    parser.add_argument("--pseudo-top-unsupervised", type=int, default=20)
    parser.add_argument("--pseudo-top-consensus", type=int, default=8)
    parser.add_argument(
        "--pseudo-target-mode",
        choices=("distance", "additive", "quadrature"),
        default="distance",
        help="Convert teacher-relative transform distance into a calibrated pseudo TRE.",
    )
    parser.add_argument("--min-pseudo-confidence", type=float, default=0.0)
    parser.add_argument("--max-pseudo-full-median-mm", type=float, default=float("inf"))
    parser.add_argument("--max-pseudo-full-p90-mm", type=float, default=float("inf"))
    parser.add_argument(
        "--require-pseudo-full-geometry",
        action="store_true",
        help="Discard pseudo candidates without the full-IOS geometry augmentation.",
    )
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", type=float, default=0.6)
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
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=0,
        help="Maximum tree depth; zero keeps the sklearn default of unlimited depth.",
    )
    parser.add_argument("--eval-top-candidates", type=int, default=20)
    parser.add_argument("--cv-trees", type=int, default=200)
    parser.add_argument("--final-trees", type=int, default=1200)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgb-l2", type=float, default=1.0)
    parser.add_argument(
        "--hgb-ensemble-leaf-nodes",
        type=int,
        nargs="*",
        default=(),
        help="Fit multiple HGB estimators with these leaf counts and average predictions.",
    )
    parser.add_argument(
        "--hgb-early-stopping",
        action="store_true",
        help="Enable sklearn's internal random validation split for HGB.",
    )
    parser.add_argument("--sample-points", type=int, default=5000)
    parser.add_argument(
        "--pseudo-target-seed",
        type=int,
        help="Fix pseudo-target point sampling independently from the model seed.",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--exact-template-weight-multiplier",
        type=float,
        default=1.0,
        help="Additional weight for verified exact-IOS template pseudo labels.",
    )
    parser.add_argument(
        "--threshold-pseudo-weight-multiplier",
        type=float,
        default=1.0,
        help="Additional weight for learned threshold-teacher pseudo labels.",
    )
    parser.add_argument(
        "--geometry-pseudo-weight-multiplier",
        type=float,
        default=0.0,
        help=(
            "Additional weight for accepted geometry self-teachers. The default "
            "keeps the historical training behavior unchanged."
        ),
    )
    parser.add_argument(
        "--cross-modal-pseudo-weight-multiplier",
        type=float,
        default=1.0,
        help="Additional weight for threshold/ToothSeg agreement pseudo labels.",
    )
    parser.add_argument(
        "--leakage-safe-cv",
        action="store_true",
        help=(
            "During CV, use only exact-template pseudo labels whose labeled source "
            "case is outside the validation fold. Final fitting still uses every pseudo label."
        ),
    )
    parser.add_argument(
        "--fold-group-by-cbct",
        action="store_true",
        help="Keep byte-identical CBCT scans in the same labeled CV fold.",
    )
    parser.add_argument(
        "--cbct-hash-cache",
        type=Path,
        help="Persistent decompressed-NIfTI hash cache shared by repeated experiments.",
    )
    parser.add_argument(
        "--include-threshold-pseudo-in-cv",
        action="store_true",
        help=(
            "Include learned threshold-teacher pseudo labels in CV training. "
            "Leave disabled unless those teachers were generated without using "
            "the current validation fold."
        ),
    )
    parser.add_argument(
        "--include-geometry-pseudo-in-cv",
        action="store_true",
        help=(
            "Include geometry self-teachers in CV training only when they were "
            "generated by fold-specific teachers. Final fitting always follows "
            "the configured geometry weight."
        ),
    )
    parser.add_argument(
        "--include-cross-modal-pseudo-in-cv",
        action="store_true",
        help=(
            "Include label-free threshold/ToothSeg agreement teachers in CV. "
            "Their target CBCT group is still excluded from matching validation folds."
        ),
    )
    parser.add_argument(
        "--jaw-specific-models",
        action="store_true",
        help="Fit independent upper and lower candidate regressors.",
    )
    parser.add_argument(
        "--group-context-features",
        action="store_true",
        help="Append within-case ranks, frequencies, and transform-consensus features.",
    )
    parser.add_argument(
        "--roi-view-feature",
        action="store_true",
        help="Append an explicit full-view versus ROI-view candidate indicator.",
    )
    parser.add_argument(
        "--modality-features",
        action="store_true",
        help=(
            "Append ToothSeg target, detected-tooth, and crown-fraction features. "
            "Use this for mixed ToothSeg/threshold candidate pools."
        ),
    )
    parser.add_argument(
        "--exclude-upper-opposite-axial",
        action="store_true",
        help="Do not select an upper-jaw candidate extracted from an explicit lower-arch target.",
    )
    parser.add_argument(
        "--balance-candidate-runs",
        action="store_true",
        help="Keep an equal candidate quota from full and ROI views when both exist.",
    )
    return parser.parse_args()


def top_indices(
    rows: list[dict], field: str, limit: int, balance_runs: bool = False
) -> list[int]:
    pool = list(range(len(rows)))
    if not balance_runs:
        return sorted(pool, key=lambda index: float(rows[index][field]))[:limit]
    run_names = sorted(
        {
            str(row.get("source_candidate_run", row.get("candidate_run", "")))
            for row in rows
        }
    )
    if len(run_names) <= 1:
        return sorted(pool, key=lambda index: float(rows[index][field]))[:limit]
    per_run = max(1, math.ceil(limit / len(run_names)))
    indices = []
    for run_name in run_names:
        run_indices = [
            index
            for index in pool
            if str(
                rows[index].get(
                    "source_candidate_run", rows[index].get("candidate_run", "")
                )
            )
            == run_name
        ]
        indices.extend(
            sorted(run_indices, key=lambda index: float(rows[index][field]))[:per_run]
        )
    if len(indices) < limit:
        selected = set(indices)
        remainder = sorted(
            (index for index in pool if index not in selected),
            key=lambda index: float(rows[index][field]),
        )
        indices.extend(remainder[: limit - len(indices)])
    return indices[:limit]


def subset_by_scores(
    rows: list[dict],
    truth_field: str,
    top_unsupervised: int,
    top_truth: int,
    balance_runs: bool = False,
) -> list[dict]:
    selected: dict[tuple[float, ...], dict] = {}
    for index in top_indices(
        rows, "selection_score_mm", top_unsupervised, balance_runs
    ):
        row = rows[index]
        selected[tuple(np.asarray(row["transform"]).round(7).reshape(-1))] = row
    for row in sorted(rows, key=lambda item: float(item[truth_field]))[:top_truth]:
        selected[tuple(np.asarray(row["transform"]).round(7).reshape(-1))] = row
    return list(selected.values())


def stratified_folds(
    groups: dict[tuple[str, str], list[dict]],
    folds: int,
    seed: int,
    case_group: dict[str, str] | None = None,
) -> list[set[str]]:
    chirality: dict[str, int] = {}
    for (case_id, _), rows in groups.items():
        if rows:
            chirality[case_id] = int(rows[0].get("ground_truth_chirality", 1))
    rng = np.random.default_rng(seed)
    result = [set() for _ in range(folds)]
    case_group = case_group or {case_id: case_id for case_id in chirality}
    for sign in (-1, 1):
        grouped: dict[str, set[str]] = {}
        for case_id, value in chirality.items():
            if value == sign:
                grouped.setdefault(case_group.get(case_id, case_id), set()).add(case_id)
        units = list(grouped.values())
        rng.shuffle(units)
        for unit in sorted(units, key=len, reverse=True):
            target = min(range(folds), key=lambda index: len(result[index]))
            result[target].update(unit)
    return result


def cbct_groups_by_case(
    records, case_ids: set[str], cache_path: Path | None = None
) -> dict[str, str]:
    """Group gzip-repacked copies by the decompressed NIfTI payload."""
    path_by_case: dict[str, Path] = {}
    for record in records:
        if record.case_id in case_ids and record.cbct_path:
            path_by_case.setdefault(record.case_id, Path(record.cbct_path))
    cache: dict[str, dict[str, object]] = {}
    if cache_path is not None and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    hash_by_path: dict[Path, str] = {}
    changed = False
    for path in sorted(set(path_by_case.values())):
        resolved = path.resolve()
        stat = resolved.stat()
        key = str(resolved)
        cached = cache.get(key, {})
        if (
            int(cached.get("size", -1)) == stat.st_size
            and int(cached.get("mtime_ns", -1)) == stat.st_mtime_ns
            and cached.get("sha256")
        ):
            digest = str(cached["sha256"])
        else:
            digest = sha256_nifti_payload(resolved)
            cache[key] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digest,
            }
            changed = True
        hash_by_path[path] = digest
    if cache_path is not None and changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        temporary.replace(cache_path)
    return {case_id: hash_by_path[path] for case_id, path in path_by_case.items()}


def leakage_safe_pseudo_keys(
    pseudo_info: dict[tuple[str, str], dict[str, object]],
    validation_cases: set[str],
    case_group: dict[str, str],
    include_threshold: bool,
    include_geometry: bool,
    include_cross_modal: bool = False,
) -> set[tuple[str, str]]:
    """Exclude pseudo targets or teachers sharing CBCT content with validation."""
    validation_groups = {
        case_group.get(case_id, case_id) for case_id in validation_cases
    }
    allowed: set[tuple[str, str]] = set()
    for key, info in pseudo_info.items():
        target_group = case_group.get(key[0], key[0])
        if target_group in validation_groups:
            continue
        teacher = str(info["teacher"])
        source_case = str(info.get("source_labeled_case_id", ""))
        source_group = case_group.get(source_case, source_case)
        if source_case and source_group in validation_groups:
            continue
        if teacher == "exact_ios_template_transfer":
            allowed.add(key)
        elif teacher == "geometry_self_teacher" and include_geometry:
            allowed.add(key)
        elif teacher == "cross_modal_consensus_teacher" and include_cross_modal:
            allowed.add(key)
        elif teacher not in {
            "exact_ios_template_transfer",
            "geometry_self_teacher",
            "cross_modal_consensus_teacher",
        } and include_threshold:
            allowed.add(key)
    return allowed


def add_pseudo_targets(
    groups: dict[tuple[str, str], list[dict]],
    pseudo_labels: list[dict],
    sample_points: int,
    seed: int,
    target_mode: str,
    min_confidence: float,
    max_full_median_mm: float = float("inf"),
    max_full_p90_mm: float = float("inf"),
) -> dict[tuple[str, str], dict[str, object]]:
    pseudo_info: dict[tuple[str, str], dict[str, object]] = {}
    for index, payload in enumerate(pseudo_labels):
        payload_confidence = float(payload.get("confidence", 1.0))
        if payload_confidence < min_confidence:
            continue
        key = (str(payload["case_id"]), str(payload["jaw"]))
        if key not in groups:
            continue
        points, _ = load_ios_points(Path(payload["ios_path"]), sample_points, seed + index)
        consensus = apply_transform(points, np.asarray(payload["transform"], dtype=np.float64))
        closest_row = None
        closest_distance = float("inf")
        for row in groups[key]:
            candidate = apply_transform(points, np.asarray(row["transform"], dtype=np.float64))
            distance = float(np.linalg.norm(candidate - consensus, axis=1).mean())
            if distance < closest_distance:
                closest_distance = distance
                closest_row = row
            teacher_tre = float(payload.get("predicted_tre_mm", 0.0))
            if target_mode == "distance":
                pseudo_tre = distance
            elif target_mode == "additive":
                pseudo_tre = teacher_tre + distance
            else:
                pseudo_tre = float(np.hypot(teacher_tre, distance))
            row["pseudo_tre_mm"] = pseudo_tre
        if closest_row is None:
            continue
        if float(closest_row.get("full_distance_median_mm", float("inf"))) > max_full_median_mm:
            continue
        if float(closest_row.get("full_distance_p90_mm", float("inf"))) > max_full_p90_mm:
            continue
        pseudo_info[key] = {
            "confidence": payload_confidence,
            "teacher": str(payload.get("teacher", "learned_threshold_teacher")),
            "source_labeled_case_id": str(payload.get("source_labeled_case_id", "")),
        }
    return pseudo_info


def feature_groups(
    groups,
    priors,
    group_context: bool,
    roi_view_feature: bool = False,
    modality_features: bool = False,
) -> dict[tuple[str, str], np.ndarray]:
    if group_context:
        feature_function = (
            candidate_multimodal_group_features
            if modality_features
            else candidate_group_features
        )
        return {
            key: feature_function(
                rows, priors[key[1]], key[1], include_roi_view=roi_view_feature
            )
            for key, rows in groups.items()
        }
    feature_function = candidate_multimodal_features if modality_features else candidate_features
    return {
        key: np.stack([feature_function(row, priors[key[1]]) for row in rows])
        for key, rows in groups.items()
    }


def pseudo_teacher_multiplier(
    teacher: str,
    exact_multiplier: float,
    geometry_multiplier: float,
    threshold_multiplier: float,
    cross_modal_multiplier: float = 1.0,
) -> float:
    if teacher == "exact_ios_template_transfer":
        return exact_multiplier
    if teacher == "geometry_self_teacher":
        return geometry_multiplier
    if teacher == "cross_modal_consensus_teacher":
        return cross_modal_multiplier
    return threshold_multiplier


def build_estimator(args: argparse.Namespace, seed: int) -> object:
    tree_criterion = getattr(args, "tree_criterion", "squared_error")
    tree_max_depth_value = int(getattr(args, "tree_max_depth", 0))
    tree_max_depth = tree_max_depth_value or None
    if args.model_type == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=args.cv_trees,
            criterion=tree_criterion,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            max_depth=tree_max_depth,
            n_jobs=args.n_jobs,
            random_state=seed,
        )
    if args.model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=args.cv_trees,
            criterion=tree_criterion,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            max_depth=tree_max_depth,
            n_jobs=args.n_jobs,
            random_state=seed,
        )
    leaf_nodes = tuple(args.hgb_ensemble_leaf_nodes) or (args.hgb_max_leaf_nodes,)
    estimators = [
        HistGradientBoostingRegressor(
            max_iter=args.cv_trees,
            min_samples_leaf=args.min_samples_leaf,
            learning_rate=args.hgb_learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=args.hgb_l2,
            early_stopping=args.hgb_early_stopping,
            random_state=seed + index,
        )
        for index, max_leaf_nodes in enumerate(leaf_nodes)
    ]
    return estimators[0] if len(estimators) == 1 else MeanRegressor(estimators)


def fit_model(
    labeled_groups,
    labeled_features,
    training_cases: set[str],
    pseudo_groups,
    pseudo_features,
    pseudo_info,
    pseudo_weight: float,
    args: argparse.Namespace,
    seed: int,
    allowed_pseudo_keys: set[tuple[str, str]] | None = None,
) -> object:
    optimization_target = getattr(args, "optimization_target", "mean_tre_mm")
    if pseudo_weight > 0 and optimization_target != "mean_tre_mm":
        raise ValueError("Pseudo TRE targets cannot supervise an official-metric regressor")
    x_by_jaw: dict[str, list[np.ndarray]] = {"upper": [], "lower": []}
    y_by_jaw: dict[str, list[float]] = {"upper": [], "lower": []}
    weights_by_jaw: dict[str, list[float]] = {"upper": [], "lower": []}
    for key, rows in labeled_groups.items():
        if key[0] not in training_cases:
            continue
        selected = subset_by_scores(
            rows,
            optimization_target,
            args.top_unsupervised,
            args.top_oracle,
            args.balance_candidate_runs,
        )
        index_by_id = {id(row): index for index, row in enumerate(rows)}
        for row in selected:
            jaw = key[1]
            x_by_jaw[jaw].append(labeled_features[key][index_by_id[id(row)]])
            y_by_jaw[jaw].append(
                float(
                    transform_tre_target(
                        registration_target_value(row, optimization_target),
                        getattr(args, "target_transform", "log1p"),
                    )
                )
            )
            weights_by_jaw[jaw].append(1.0 / max(len(selected), 1))
    if pseudo_weight > 0:
        for key, info in pseudo_info.items():
            if allowed_pseudo_keys is not None and key not in allowed_pseudo_keys:
                continue
            confidence = float(info["confidence"])
            teacher_multiplier = pseudo_teacher_multiplier(
                str(info["teacher"]),
                args.exact_template_weight_multiplier,
                args.geometry_pseudo_weight_multiplier,
                args.threshold_pseudo_weight_multiplier,
                args.cross_modal_pseudo_weight_multiplier,
            )
            if teacher_multiplier <= 0.0:
                continue
            rows = pseudo_groups[key]
            selected = subset_by_scores(
                rows,
                "pseudo_tre_mm",
                args.pseudo_top_unsupervised,
                args.pseudo_top_consensus,
                args.balance_candidate_runs,
            )
            index_by_id = {id(row): index for index, row in enumerate(rows)}
            for row in selected:
                jaw = key[1]
                x_by_jaw[jaw].append(pseudo_features[key][index_by_id[id(row)]])
                y_by_jaw[jaw].append(
                    float(
                        transform_tre_target(
                            float(row["pseudo_tre_mm"]),
                            getattr(args, "target_transform", "log1p"),
                        )
                    )
                )
                weights_by_jaw[jaw].append(
                    pseudo_weight
                    * confidence
                    * teacher_multiplier
                    / max(len(selected), 1)
                )
    if args.jaw_specific_models:
        models = {}
        for jaw_index, jaw in enumerate(("upper", "lower")):
            model = build_estimator(args, seed + 1009 * jaw_index)
            model.fit(
                np.stack(x_by_jaw[jaw]),
                np.asarray(y_by_jaw[jaw]),
                sample_weight=np.asarray(weights_by_jaw[jaw]),
            )
            models[jaw] = model
        return models
    model = build_estimator(args, seed)
    x_train = x_by_jaw["upper"] + x_by_jaw["lower"]
    y_train = y_by_jaw["upper"] + y_by_jaw["lower"]
    weights = weights_by_jaw["upper"] + weights_by_jaw["lower"]
    model.fit(np.stack(x_train), np.asarray(y_train), sample_weight=np.asarray(weights))
    return model


def evaluate(model, groups, features, cases: set[str], args: argparse.Namespace):
    optimization_target = getattr(args, "optimization_target", "mean_tre_mm")
    errors: list[float] = []
    rows_out: list[dict[str, object]] = []
    for (case_id, jaw), rows in sorted(groups.items()):
        if case_id not in cases:
            continue
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
            candidate_indices = [candidate_pool[index] for index in local_indices]
        else:
            candidate_indices = sorted(
                candidate_pool,
                key=lambda index: float(rows[index]["selection_score_mm"]),
            )[: args.eval_top_candidates]
        estimator = model[jaw] if isinstance(model, dict) else model
        prediction = inverse_tre_target(
            estimator.predict(features[(case_id, jaw)][candidate_indices]),
            getattr(args, "target_transform", "log1p"),
        )
        local_index = int(np.argmin(prediction))
        index = candidate_indices[local_index]
        selected = rows[index]
        oracle = min(rows, key=lambda row: float(row["mean_tre_mm"]))
        error = float(selected["mean_tre_mm"])
        errors.append(error)
        rows_out.append(
            {
                "case_id": case_id,
                "jaw": jaw,
                "predicted_tre_mm": float(prediction[local_index]),
                "predicted_optimization_score": float(prediction[local_index]),
                "optimization_target": optimization_target,
                "mean_tre_mm": error,
                "translation_error_mm": selected.get("translation_error_mm", ""),
                "rotation_error_deg": selected.get("rotation_error_deg", ""),
                "official_balanced_error": selected.get(
                    "official_balanced_error", ""
                ),
                "original_unsupervised_rank": int(selected["unsupervised_rank"]),
                "source_variant": selected["source_variant"],
                "target": selected["target"],
                "method": selected["method"],
                "candidate_run": selected.get("candidate_run", ""),
                "roi_view_selected": int(
                    "roi" in str(selected.get("candidate_run", "")).lower()
                    or "roi"
                    in str(
                        selected.get("target_metadata", {}).get("volume_path", "")
                    ).lower()
                ),
                "full_geometry_available": int(
                    "full_distance_p90_mm" in selected
                ),
                "full_distance_median_mm": selected.get(
                    "full_distance_median_mm", float("inf")
                ),
                "full_distance_p90_mm": selected.get(
                    "full_distance_p90_mm", float("inf")
                ),
                "transform_key": ",".join(
                    f"{value:.6f}"
                    for value in np.asarray(selected["transform"], dtype=np.float64).reshape(-1)
                ),
                "oracle_mean_tre_mm": float(oracle["mean_tre_mm"]),
            }
        )
    return errors, rows_out


def candidate_predictions(
    model, groups, features, cases: set[str], args: argparse.Namespace, method: str
) -> list[dict[str, object]]:
    optimization_target = getattr(args, "optimization_target", "mean_tre_mm")
    rows_out: list[dict[str, object]] = []
    for (case_id, jaw), rows in sorted(groups.items()):
        if case_id not in cases:
            continue
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
            candidate_indices = [candidate_pool[index] for index in local_indices]
        else:
            candidate_indices = sorted(
                candidate_pool,
                key=lambda index: float(rows[index]["selection_score_mm"]),
            )[: args.eval_top_candidates]
        estimator = model[jaw] if isinstance(model, dict) else model
        predictions = inverse_tre_target(
            estimator.predict(features[(case_id, jaw)][candidate_indices]),
            getattr(args, "target_transform", "log1p"),
        )
        for index, prediction in zip(candidate_indices, predictions):
            rows_out.append(
                {
                    "case_id": case_id,
                    "jaw": jaw,
                    "ensemble_method": method,
                    "candidate_index": index,
                    "ensemble_score": float(prediction),
                    "optimization_target": optimization_target,
                    "mean_tre_mm": float(rows[index]["mean_tre_mm"]),
                    "translation_error_mm": rows[index].get(
                        "translation_error_mm", ""
                    ),
                    "rotation_error_deg": rows[index].get("rotation_error_deg", ""),
                    "official_balanced_error": rows[index].get(
                        "official_balanced_error", ""
                    ),
                    "candidate_run": rows[index].get("candidate_run", ""),
                }
            )
    return rows_out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.roi_view_feature and not args.group_context_features:
        raise ValueError("--roi-view-feature requires --group-context-features")
    cv_trees = args.cv_trees
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_cases = set(args.eval_case_ids)
    records = load_manifest(args.manifest)
    labeled_groups = {
        key: rows
        for key, rows in load_candidate_groups(args.labeled_runs).items()
        if key[0] not in eval_cases
    }
    eval_groups = load_candidate_groups(args.eval_runs) if args.eval_runs else {}
    pseudo_groups = load_candidate_groups(args.pseudo_runs)
    enrich_candidate_registration_metrics(labeled_groups, records)
    enrich_candidate_registration_metrics(eval_groups, records)
    if args.optimization_target != "mean_tre_mm" and any(
        weight > 0 for weight in args.pseudo_weights
    ):
        raise ValueError(
            "Official-metric optimization currently requires --pseudo-weights 0"
        )
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
        args.pseudo_target_seed if args.pseudo_target_seed is not None else args.seed,
        args.pseudo_target_mode,
        args.min_pseudo_confidence,
        args.max_pseudo_full_median_mm,
        args.max_pseudo_full_p90_mm,
    )
    if not pseudo_info:
        raise RuntimeError("No accepted pseudo labels matched the pseudo candidate runs")
    pseudo_groups = {key: pseudo_groups[key] for key in pseudo_info}
    train_cases = {key[0] for key in labeled_groups}
    case_group = {case_id: case_id for case_id in train_cases}
    eval_excluded_cases = set(eval_cases)
    if args.fold_group_by_cbct:
        relevant_cases = (
            train_cases
            | eval_cases
            | {key[0] for key in pseudo_info}
            | {
                str(info.get("source_labeled_case_id", ""))
                for info in pseudo_info.values()
                if info.get("source_labeled_case_id")
            }
        )
        case_group = cbct_groups_by_case(
            records, relevant_cases, args.cbct_hash_cache
        )
        eval_cbct_groups = {
            case_group.get(case_id, case_id) for case_id in eval_cases
        }
        duplicate_eval_cases = {
            case_id
            for case_id in train_cases
            if case_group.get(case_id, case_id) in eval_cbct_groups
        }
        if duplicate_eval_cases:
            labeled_groups = {
                key: rows
                for key, rows in labeled_groups.items()
                if key[0] not in duplicate_eval_cases
            }
            eval_excluded_cases.update(duplicate_eval_cases)
            train_cases = {key[0] for key in labeled_groups}
    folds = stratified_folds(labeled_groups, args.folds, args.seed, case_group)

    cv_rows: list[dict[str, object]] = []
    for pseudo_weight in args.pseudo_weights:
        errors: list[float] = []
        oof_rows: list[dict[str, object]] = []
        weight_name = str(pseudo_weight).replace(".", "p")
        candidate_score_rows: list[dict[str, object]] = []
        for fold_index, validation_cases in enumerate(folds):
            excluded = eval_excluded_cases | validation_cases
            priors = {
                jaw: fit_rotation_prior(records, jaw, excluded_cases=excluded)
                for jaw in ("upper", "lower")
            }
            labeled_features = feature_groups(
                labeled_groups,
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
            allowed_pseudo_keys = None
            if args.leakage_safe_cv:
                allowed_pseudo_keys = leakage_safe_pseudo_keys(
                    pseudo_info,
                    validation_cases,
                    case_group,
                    args.include_threshold_pseudo_in_cv,
                    args.include_geometry_pseudo_in_cv,
                    args.include_cross_modal_pseudo_in_cv,
                )
            model = fit_model(
                labeled_groups,
                labeled_features,
                train_cases - validation_cases,
                pseudo_groups,
                pseudo_features,
                pseudo_info,
                pseudo_weight,
                args,
                args.seed + fold_index,
                allowed_pseudo_keys,
            )
            fold_errors, fold_rows = evaluate(
                model, labeled_groups, labeled_features, validation_cases, args
            )
            fold_scores = candidate_predictions(
                model,
                labeled_groups,
                labeled_features,
                validation_cases,
                args,
                f"pseudo_weight_{weight_name}",
            )
            errors.extend(fold_errors)
            for row in fold_rows:
                row["fold"] = fold_index
                row["pseudo_weight"] = pseudo_weight
            for row in fold_scores:
                row["fold"] = fold_index
            oof_rows.extend(fold_rows)
            candidate_score_rows.extend(fold_scores)
        values = np.asarray(errors, dtype=np.float64)
        cv_rows.append(
            {
                "pseudo_weight": pseudo_weight,
                "mean_tre_mm": float(values.mean()),
                "median_tre_mm": float(np.median(values)),
                "p90_tre_mm": float(np.quantile(values, 0.9)),
            }
        )
        print(
            f"pseudo_weight={pseudo_weight:g}: CV mean={values.mean():.3f} mm; "
            f"median={np.median(values):.3f} mm"
        )
        write_csv(args.output_dir / f"oof_weight_{weight_name}.csv", oof_rows)
        write_csv(
            args.output_dir / f"candidate_scores_weight_{weight_name}.csv",
            candidate_score_rows,
        )
    write_csv(args.output_dir / "cv_pseudo_weights.csv", cv_rows)
    best = min(cv_rows, key=lambda row: (row["mean_tre_mm"], row["p90_tre_mm"]))

    priors = {
        jaw: fit_rotation_prior(records, jaw, excluded_cases=eval_excluded_cases)
        for jaw in ("upper", "lower")
    }
    labeled_features = feature_groups(
        labeled_groups,
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
    eval_features = (
        feature_groups(
            eval_groups,
            priors,
            args.group_context_features,
            args.roi_view_feature,
            args.modality_features,
        )
        if eval_groups
        else {}
    )
    args.cv_trees = args.final_trees
    final_model = None
    final_errors: list[float] = []
    final_evaluation: list[dict[str, object]] = []
    dev_ablation: list[dict[str, object]] = []
    for pseudo_weight in args.pseudo_weights:
        model = fit_model(
            labeled_groups,
            labeled_features,
            train_cases,
            pseudo_groups,
            pseudo_features,
            pseudo_info,
            pseudo_weight,
            args,
            args.seed,
        )
        if eval_cases:
            errors, evaluation = evaluate(
                model, eval_groups, eval_features, eval_cases, args
            )
            values = np.asarray(errors, dtype=np.float64)
            dev_ablation.append(
                {
                    "pseudo_weight": pseudo_weight,
                    "selected_by_cv": int(np.isclose(pseudo_weight, float(best["pseudo_weight"]))),
                    "mean_tre_mm": float(values.mean()),
                    "median_tre_mm": float(np.median(values)),
                    "p90_tre_mm": float(np.quantile(values, 0.9)),
                }
            )
            weight_name = str(pseudo_weight).replace(".", "p")
            write_csv(args.output_dir / f"evaluation_weight_{weight_name}.csv", evaluation)
        if np.isclose(pseudo_weight, float(best["pseudo_weight"])):
            final_model = model
            final_errors = errors if eval_cases else []
            final_evaluation = evaluation if eval_cases else []
    if final_model is None:
        raise RuntimeError("The CV-selected pseudo weight was not present in --pseudo-weights")
    if dev_ablation:
        write_csv(args.output_dir / "dev_weight_ablation.csv", dev_ablation)
    if final_evaluation:
        write_csv(args.output_dir / "evaluation.csv", final_evaluation)
    write_csv(
        args.output_dir / "pseudo_training_groups.csv",
        [
            {
                "case_id": key[0],
                "jaw": key[1],
                "confidence": info["confidence"],
                "teacher": info["teacher"],
                "source_labeled_case_id": info["source_labeled_case_id"],
                "candidate_count": len(pseudo_groups[key]),
            }
            for key, info in sorted(pseudo_info.items())
        ],
    )
    joblib.dump(
        {
            "model": final_model,
            "feature_names": (
                (
                    (
                        MULTIMODAL_ROI_GROUP_FEATURE_NAMES
                        if args.roi_view_feature
                        else MULTIMODAL_GROUP_FEATURE_NAMES
                    )
                    if args.modality_features
                    else (
                        ROI_GROUP_FEATURE_NAMES
                        if args.roi_view_feature
                        else GROUP_FEATURE_NAMES
                    )
                )
                if args.group_context_features
                else (
                    MULTIMODAL_FEATURE_NAMES
                    if args.modality_features
                    else FEATURE_NAMES
                )
            ),
            "best_pseudo_weight": best,
            "pseudo_groups": len(pseudo_info),
            "effective_pseudo_groups": sum(
                pseudo_teacher_multiplier(
                    str(info["teacher"]),
                    args.exact_template_weight_multiplier,
                    args.geometry_pseudo_weight_multiplier,
                    args.threshold_pseudo_weight_multiplier,
                    args.cross_modal_pseudo_weight_multiplier,
                )
                > 0.0
                for info in pseudo_info.values()
            ),
            "pseudo_target_mode": args.pseudo_target_mode,
            "min_pseudo_confidence": args.min_pseudo_confidence,
            "max_pseudo_full_median_mm": args.max_pseudo_full_median_mm,
            "max_pseudo_full_p90_mm": args.max_pseudo_full_p90_mm,
            "model_type": args.model_type,
            "target_transform": args.target_transform,
            "optimization_target": args.optimization_target,
            "tree_criterion": args.tree_criterion,
            "tree_max_depth": args.tree_max_depth,
            "hgb_learning_rate": args.hgb_learning_rate,
            "hgb_max_leaf_nodes": args.hgb_max_leaf_nodes,
            "hgb_l2": args.hgb_l2,
            "hgb_ensemble_leaf_nodes": list(args.hgb_ensemble_leaf_nodes),
            "hgb_early_stopping": args.hgb_early_stopping,
            "pseudo_target_seed": args.pseudo_target_seed,
            "exact_template_weight_multiplier": args.exact_template_weight_multiplier,
            "geometry_pseudo_weight_multiplier": args.geometry_pseudo_weight_multiplier,
            "threshold_pseudo_weight_multiplier": args.threshold_pseudo_weight_multiplier,
            "cross_modal_pseudo_weight_multiplier": args.cross_modal_pseudo_weight_multiplier,
            "leakage_safe_cv": args.leakage_safe_cv,
            "fold_group_by_cbct": args.fold_group_by_cbct,
            "cbct_group_hash": "decompressed_nifti_payload",
            "cbct_hash_cache": str(args.cbct_hash_cache) if args.cbct_hash_cache else None,
            "jaw_specific_models": args.jaw_specific_models,
            "group_context_features": args.group_context_features,
            "roi_view_feature": args.roi_view_feature,
            "modality_features": args.modality_features,
            "exclude_upper_opposite_axial": args.exclude_upper_opposite_axial,
            "balance_candidate_runs": args.balance_candidate_runs,
            "include_threshold_pseudo_in_cv": args.include_threshold_pseudo_in_cv,
            "include_geometry_pseudo_in_cv": args.include_geometry_pseudo_in_cv,
            "include_cross_modal_pseudo_in_cv": args.include_cross_modal_pseudo_in_cv,
            "folds": args.folds,
            "top_unsupervised": args.top_unsupervised,
            "top_oracle": args.top_oracle,
            "pseudo_top_unsupervised": args.pseudo_top_unsupervised,
            "pseudo_top_consensus": args.pseudo_top_consensus,
            "eval_top_candidates": args.eval_top_candidates,
            "cv_trees": cv_trees,
            "final_trees": args.final_trees,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "n_jobs": args.n_jobs,
            "training_cases": sorted(train_cases),
        },
        args.output_dir / "semisupervised_candidate_reranker.joblib",
    )
    values = np.asarray(final_errors, dtype=np.float64)
    summary = {
        "best_cv": best,
        "pseudo_groups": len(pseudo_info),
        "effective_pseudo_groups": sum(
            pseudo_teacher_multiplier(
                str(info["teacher"]),
                args.exact_template_weight_multiplier,
                args.geometry_pseudo_weight_multiplier,
                args.threshold_pseudo_weight_multiplier,
                args.cross_modal_pseudo_weight_multiplier,
            )
            > 0.0
            for info in pseudo_info.values()
        ),
        "pseudo_teacher_counts": {
            teacher: sum(info["teacher"] == teacher for info in pseudo_info.values())
            for teacher in sorted({str(info["teacher"]) for info in pseudo_info.values()})
        },
        "mean_confidence": float(
            np.mean([float(info["confidence"]) for info in pseudo_info.values()])
        ),
        "pseudo_target_mode": args.pseudo_target_mode,
        "min_pseudo_confidence": args.min_pseudo_confidence,
        "max_pseudo_full_median_mm": args.max_pseudo_full_median_mm,
        "max_pseudo_full_p90_mm": args.max_pseudo_full_p90_mm,
        "model_type": args.model_type,
        "target_transform": args.target_transform,
        "optimization_target": args.optimization_target,
        "tree_criterion": args.tree_criterion,
        "tree_max_depth": args.tree_max_depth,
        "hgb_learning_rate": args.hgb_learning_rate,
        "hgb_max_leaf_nodes": args.hgb_max_leaf_nodes,
        "hgb_l2": args.hgb_l2,
        "hgb_ensemble_leaf_nodes": list(args.hgb_ensemble_leaf_nodes),
        "hgb_early_stopping": args.hgb_early_stopping,
        "pseudo_target_seed": args.pseudo_target_seed,
        "exact_template_weight_multiplier": args.exact_template_weight_multiplier,
        "geometry_pseudo_weight_multiplier": args.geometry_pseudo_weight_multiplier,
        "threshold_pseudo_weight_multiplier": args.threshold_pseudo_weight_multiplier,
        "cross_modal_pseudo_weight_multiplier": args.cross_modal_pseudo_weight_multiplier,
        "leakage_safe_cv": args.leakage_safe_cv,
        "fold_group_by_cbct": args.fold_group_by_cbct,
        "cbct_group_hash": "decompressed_nifti_payload",
        "cbct_hash_cache": str(args.cbct_hash_cache) if args.cbct_hash_cache else None,
        "jaw_specific_models": args.jaw_specific_models,
        "group_context_features": args.group_context_features,
        "roi_view_feature": args.roi_view_feature,
        "exclude_upper_opposite_axial": args.exclude_upper_opposite_axial,
        "balance_candidate_runs": args.balance_candidate_runs,
        "include_threshold_pseudo_in_cv": args.include_threshold_pseudo_in_cv,
        "include_geometry_pseudo_in_cv": args.include_geometry_pseudo_in_cv,
        "include_cross_modal_pseudo_in_cv": args.include_cross_modal_pseudo_in_cv,
        "folds": args.folds,
        "top_unsupervised": args.top_unsupervised,
        "top_oracle": args.top_oracle,
        "pseudo_top_unsupervised": args.pseudo_top_unsupervised,
        "pseudo_top_consensus": args.pseudo_top_consensus,
        "eval_top_candidates": args.eval_top_candidates,
        "cv_trees": cv_trees,
        "final_trees": args.final_trees,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "n_jobs": args.n_jobs,
        "training_cases": len(train_cases),
        "dev_mean_tre_mm": float(values.mean()) if len(values) else None,
        "dev_median_tre_mm": float(np.median(values)) if len(values) else None,
        "dev_p90_tre_mm": float(np.quantile(values, 0.9)) if len(values) else None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
