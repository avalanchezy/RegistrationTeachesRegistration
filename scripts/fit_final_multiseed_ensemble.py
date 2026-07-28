from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweep_multimodal_reranker import estimator_args
from scripts.sweep_joint_pair_reranker import fit_pair_prior
from scripts.sweep_pairwise_multimodal_reranker import fit_pairwise_model
from scripts.train_semisupervised_candidate_reranker import (
    add_pseudo_targets,
    feature_groups,
    fit_model,
)
from task2reg.candidate_learning import (
    FEATURE_NAMES,
    GROUP_FEATURE_NAMES,
    MULTIMODAL_FEATURE_NAMES,
    MULTIMODAL_GROUP_FEATURE_NAMES,
    MULTIMODAL_ROI_GROUP_FEATURE_NAMES,
    REGISTRATION_TARGET_NAMES,
    ROI_GROUP_FEATURE_NAMES,
    enrich_candidate_registration_metrics,
    load_candidate_groups,
)
from task2reg.data import load_manifest
from task2reg.priors import fit_rotation_prior, save_rotation_priors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit deployment reranker ensembles on every labeled Task 2 case."
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--top-unsupervised", type=int, default=20)
    parser.add_argument("--top-oracle", type=int, default=8)
    parser.add_argument("--eval-top-candidates", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", type=float, default=0.2)
    parser.add_argument("--model-scope", choices=("jaw", "shared"), default="jaw")
    parser.add_argument(
        "--model-type",
        choices=("extra_trees", "random_forest", "hist_gradient_boosting"),
        default="extra_trees",
    )
    parser.add_argument(
        "--target-transform", choices=("log1p", "sqrt", "identity"), default="log1p"
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
    parser.add_argument("--regression-trees", type=int, default=400)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgb-l2", type=float, default=1.0)
    parser.add_argument("--hgb-ensemble-leaf-nodes", type=int, nargs="*", default=())
    parser.add_argument("--hgb-early-stopping", action="store_true")
    parser.add_argument("--fit-pairwise", action="store_true")
    parser.add_argument("--pairwise-top-unsupervised", type=int, default=0)
    parser.add_argument("--pairwise-top-oracle", type=int, default=0)
    parser.add_argument(
        "--pairwise-model-scope", choices=("jaw", "shared"), default=None
    )
    parser.add_argument("--pairwise-min-samples-leaf", type=int, default=4)
    parser.add_argument("--pairwise-max-features", type=float, default=0.2)
    parser.add_argument(
        "--pairwise-model-type",
        choices=("extra_trees", "random_forest"),
        default="extra_trees",
    )
    parser.add_argument(
        "--pairwise-criterion", choices=("gini", "entropy", "log_loss"), default="gini"
    )
    parser.add_argument(
        "--pairwise-optimization-target",
        choices=REGISTRATION_TARGET_NAMES,
        default="mean_tre_mm",
    )
    parser.add_argument("--pairwise-min-log-tre-gap", type=float, default=0.05)
    parser.add_argument("--pairwise-trees", type=int, default=250)
    parser.add_argument("--max-pairs-per-group", type=int, default=600)
    parser.add_argument("--eval-opponents", type=int, default=30)
    parser.add_argument("--group-context-features", action="store_true")
    parser.add_argument("--roi-view-feature", action="store_true")
    parser.add_argument("--modality-features", action="store_true")
    parser.add_argument("--balance-candidate-runs", action="store_true")
    parser.add_argument("--exclude-upper-opposite-axial", action="store_true")
    parser.add_argument(
        "--regression-aggregation",
        choices=("mean", "median", "rank_mean", "vote"),
        default="vote",
    )
    parser.add_argument(
        "--pairwise-aggregation",
        choices=("mean", "median", "rank_mean", "vote"),
        default="median",
    )
    parser.add_argument("--blend-alpha", type=float, default=0.525)
    parser.add_argument("--joint-pair-top-k", type=int, default=4)
    parser.add_argument("--joint-angle-weight", type=float, default=0.045)
    parser.add_argument("--joint-translation-weight", type=float, default=0.0025)
    parser.add_argument(
        "--global-crown-modes",
        nargs="*",
        choices=(
            "crown",
            "crown-probability",
            "crown-guided",
            "crown-guided-fine",
            "crown-guided-high",
        ),
        default=(),
        help="Global crown target families generated during deployment inference.",
    )
    parser.add_argument(
        "--exclude-legacy-threshold-candidates",
        action="store_true",
        help="Use only global crown candidates in the enhanced deployment selector.",
    )
    parser.add_argument(
        "--global-include-crown-refinement",
        action="store_true",
        help="Deploy crown-refinement candidates in addition to the selected global modes.",
    )
    parser.add_argument("--crown-tta-mode", choices=("none", "d4"), default="none")
    return parser.parse_args()


def feature_names(args: argparse.Namespace) -> tuple[str, ...]:
    if args.group_context_features:
        if args.modality_features:
            return (
                MULTIMODAL_ROI_GROUP_FEATURE_NAMES
                if args.roi_view_feature
                else MULTIMODAL_GROUP_FEATURE_NAMES
            )
        return ROI_GROUP_FEATURE_NAMES if args.roi_view_feature else GROUP_FEATURE_NAMES
    return MULTIMODAL_FEATURE_NAMES if args.modality_features else FEATURE_NAMES


def resolved_pairwise_selection(args: argparse.Namespace) -> dict[str, object]:
    return {
        "top_unsupervised": args.pairwise_top_unsupervised or args.top_unsupervised,
        "top_oracle": args.pairwise_top_oracle or args.top_oracle,
        "model_scope": args.pairwise_model_scope or args.model_scope,
    }


def atomic_joblib_dump(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary, compress=3)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.roi_view_feature and not args.group_context_features:
        raise ValueError("--roi-view-feature requires --group-context-features")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(args.manifest)
    groups = load_candidate_groups(args.labeled_runs)
    enrich_candidate_registration_metrics(groups, records)
    if not groups:
        raise RuntimeError("No labeled candidate groups were found")
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
    priors = {
        jaw: fit_rotation_prior(records, jaw, excluded_cases=set())
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
    save_rotation_priors(priors, args.output_dir / "rotation_prior_all_labeled.json")
    pair_prior = fit_pair_prior(records, excluded_cases=set())

    estimator_source = SimpleNamespace(
        cv_trees=args.regression_trees,
        n_jobs=args.n_jobs,
        hgb_learning_rate=args.hgb_learning_rate,
        hgb_max_leaf_nodes=args.hgb_max_leaf_nodes,
        hgb_l2=args.hgb_l2,
        hgb_ensemble_leaf_nodes=args.hgb_ensemble_leaf_nodes,
        hgb_early_stopping=args.hgb_early_stopping,
        balance_candidate_runs=args.balance_candidate_runs,
        exclude_upper_opposite_axial=args.exclude_upper_opposite_axial,
    )
    model_args = estimator_args(
        estimator_source,
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
    regression_models = []
    for index, seed in enumerate(args.seeds, 1):
        regression_models.append(
            fit_model(
                groups,
                features,
                train_cases,
                pseudo_groups,
                pseudo_features,
                pseudo_info,
                args.pseudo_weight,
                model_args,
                seed,
            )
        )
        print(f"[{index}/{len(args.seeds)}] fitted regression seed {seed}", flush=True)
    regression_config = {
        "top_unsupervised": args.top_unsupervised,
        "top_oracle": args.top_oracle,
        "eval_top_candidates": args.eval_top_candidates,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "model_scope": args.model_scope,
        "model_type": args.model_type,
        "target_transform": args.target_transform,
        "optimization_target": args.optimization_target,
        "tree_criterion": args.tree_criterion,
        "tree_max_depth": args.tree_max_depth,
        "trees_per_seed": args.regression_trees,
    }
    shared_metadata = {
        "seeds": args.seeds,
        "training_cases": sorted(train_cases),
        "training_jaws": len(groups),
        "candidate_rows": sum(len(rows) for rows in groups.values()),
        "feature_names": feature_names(args),
        "group_context_features": args.group_context_features,
        "roi_view_feature": args.roi_view_feature,
        "modality_features": args.modality_features,
        "balance_candidate_runs": args.balance_candidate_runs,
        "exclude_upper_opposite_axial": args.exclude_upper_opposite_axial,
        "pseudo_training": {
            "weight": args.pseudo_weight,
            "groups": len(pseudo_info),
            "target_mode": args.pseudo_target_mode,
            "target_seed": args.pseudo_target_seed,
            "top_unsupervised": args.pseudo_top_unsupervised,
            "top_consensus": args.pseudo_top_consensus,
            "min_confidence": args.min_pseudo_confidence,
            "max_full_median_mm": args.max_pseudo_full_median_mm,
            "max_full_p90_mm": args.max_pseudo_full_p90_mm,
            "require_full_geometry": args.require_pseudo_full_geometry,
            "exact_template_multiplier": args.exact_template_pseudo_weight_multiplier,
            "geometry_multiplier": args.geometry_pseudo_weight_multiplier,
            "threshold_multiplier": args.threshold_pseudo_weight_multiplier,
            "cross_modal_multiplier": args.cross_modal_pseudo_weight_multiplier,
            "pseudo_labels": str(args.pseudo_labels.resolve()) if args.pseudo_labels else None,
            "pseudo_labels_sha256": (
                sha256_file(args.pseudo_labels) if args.pseudo_labels else None
            ),
            "pseudo_runs": [str(path.resolve()) for path in args.pseudo_runs],
        },
    }
    atomic_joblib_dump(
        {
            **shared_metadata,
            "models": regression_models,
            "configuration": regression_config,
        },
        args.output_dir / "regression_ensemble.joblib",
    )

    pairwise_config = None
    if args.fit_pairwise:
        pairwise_selection = resolved_pairwise_selection(args)
        pairwise_config = {
            "top_unsupervised": pairwise_selection["top_unsupervised"],
            "top_oracle": pairwise_selection["top_oracle"],
            "min_samples_leaf": args.pairwise_min_samples_leaf,
            "max_features": args.pairwise_max_features,
            "model_scope": pairwise_selection["model_scope"],
            "model_type": args.pairwise_model_type,
            "criterion": args.pairwise_criterion,
            "optimization_target": args.pairwise_optimization_target,
            "min_log_tre_gap": args.pairwise_min_log_tre_gap,
            "eval_top_candidates": args.eval_top_candidates,
            "eval_opponents": args.eval_opponents,
            "max_pairs_per_group": args.max_pairs_per_group,
            "trees_per_seed": args.pairwise_trees,
        }
        pairwise_models = []
        for index, seed in enumerate(args.seeds, 1):
            pairwise_models.append(
                fit_pairwise_model(
                    groups,
                    features,
                    train_cases,
                    pairwise_config,
                    args,
                    seed,
                    args.pairwise_trees,
                )
            )
            print(f"[{index}/{len(args.seeds)}] fitted pairwise seed {seed}", flush=True)
        atomic_joblib_dump(
            {
                **shared_metadata,
                "models": pairwise_models,
                "configuration": pairwise_config,
            },
            args.output_dir / "pairwise_ensemble.joblib",
        )

    deployment_policy = {
        "candidate_score_mode": "learned_blend",
        "global_crown_modes": list(args.global_crown_modes),
        "global_geometry_candidate_budget": 30,
        "global_include_crown_refinement": args.global_include_crown_refinement,
        "include_legacy_threshold_candidates": not args.exclude_legacy_threshold_candidates,
        "crown_tta_mode": args.crown_tta_mode,
        "regression_aggregation": args.regression_aggregation,
        "pairwise_aggregation": args.pairwise_aggregation,
        "blend_alpha": args.blend_alpha,
        "joint_pair_top_k": args.joint_pair_top_k,
        "joint_angle_weight_mm_per_deg": args.joint_angle_weight,
        "joint_translation_weight": args.joint_translation_weight,
        "allow_chirality_mismatch": False,
        "pair_prior_relative_rotation": pair_prior.relative_rotation.tolist(),
        "pair_prior_relative_translation": pair_prior.relative_translation.tolist(),
    }
    (args.output_dir / "deployment_policy.json").write_text(
        json.dumps(deployment_policy, indent=2), encoding="utf-8"
    )

    summary = {
        **shared_metadata,
        "regression": regression_config,
        "pairwise": pairwise_config,
        "rotation_prior": "rotation_prior_all_labeled.json",
        "deployment_policy": deployment_policy,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
