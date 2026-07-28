from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial.transform import Rotation

from .candidate_learning import (
    GROUP_FEATURE_NAMES,
    MULTIMODAL_GROUP_FEATURE_NAMES,
    MULTIMODAL_ROI_GROUP_FEATURE_NAMES,
    ROI_GROUP_FEATURE_NAMES,
    candidate_group_features,
    candidate_multimodal_group_features,
    is_opposite_axial_target,
)
from .priors import proper_protocol_rotation


def fractional_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / float(max(len(values) - 1, 1))


def aggregate_seed_costs(costs: np.ndarray, method: str) -> np.ndarray:
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[0] == 0:
        raise ValueError("Seed costs must have shape (seeds, candidates)")
    if method == "mean":
        return costs.mean(axis=0)
    if method == "median":
        return np.median(costs, axis=0)
    if method == "rank_mean":
        return np.mean(np.stack([fractional_ranks(row) for row in costs]), axis=0)
    if method == "vote":
        votes = np.zeros(costs.shape[1], dtype=np.float64)
        for row in costs:
            votes[int(np.argmin(row))] += 1.0
        return -votes + 1e-6 * costs.mean(axis=0)
    raise ValueError(f"Unknown seed aggregation: {method}")


def candidate_indices(
    rows: list[dict],
    jaw: str,
    budget: int,
    balance_runs: bool,
    exclude_upper_opposite_axial: bool,
) -> list[int]:
    pool = list(range(len(rows)))
    if exclude_upper_opposite_axial and jaw == "upper":
        filtered = [index for index in pool if not is_opposite_axial_target(rows[index], jaw)]
        if filtered:
            pool = filtered
    if not balance_runs:
        return sorted(
            pool, key=lambda index: float(rows[index]["selection_score_mm"])
        )[:budget]
    run_names = sorted(
        {
            str(row.get("source_candidate_run", row.get("candidate_run", "")))
            for row in rows
        }
    )
    if len(run_names) <= 1:
        return sorted(
            pool, key=lambda index: float(rows[index]["selection_score_mm"])
        )[:budget]
    per_run = max(1, int(np.ceil(budget / len(run_names))))
    selected: list[int] = []
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
        selected.extend(
            sorted(
                run_indices,
                key=lambda index: float(rows[index]["selection_score_mm"]),
            )[:per_run]
        )
    if len(selected) < budget:
        selected_set = set(selected)
        remainder = sorted(
            (index for index in pool if index not in selected_set),
            key=lambda index: float(rows[index]["selection_score_mm"]),
        )
        selected.extend(remainder[: budget - len(selected)])
    return selected[:budget]


def inverse_target(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if mode == "log1p":
        return np.expm1(values)
    if mode == "sqrt":
        return np.square(np.maximum(values, 0.0))
    if mode == "identity":
        return np.maximum(values, 0.0)
    raise ValueError(f"Unknown target transform: {mode}")


def regression_seed_costs(
    models: list[object],
    features: np.ndarray,
    indices: list[int],
    jaw: str,
    target_transform: str,
) -> np.ndarray:
    costs = []
    for model in models:
        estimator = model[jaw] if isinstance(model, dict) else model
        costs.append(
            inverse_target(estimator.predict(features[indices]), target_transform)
        )
    return np.stack(costs)


def pairwise_seed_costs(
    models: list[object],
    features: np.ndarray,
    indices: list[int],
    jaw: str,
    opponent_budget: int,
) -> np.ndarray:
    opponents = indices[: min(opponent_budget, len(indices))]
    differences = []
    counts = []
    for candidate in indices:
        comparison = [index for index in opponents if index != candidate]
        counts.append(len(comparison))
        if comparison:
            differences.append(features[candidate][None, :] - features[comparison])
    stacked = np.concatenate(differences, axis=0) if differences else None
    seed_costs = []
    for model in models:
        estimator = model[jaw] if isinstance(model, dict) else model
        probabilities = (
            estimator.predict_proba(stacked)[:, 1]
            if stacked is not None
            else np.empty(0, dtype=np.float64)
        )
        scores = []
        offset = 0
        for count in counts:
            if count == 0:
                scores.append(0.5)
            else:
                scores.append(float(probabilities[offset : offset + count].mean()))
                offset += count
        seed_costs.append(-np.asarray(scores, dtype=np.float64))
    return np.stack(seed_costs)


def blended_candidates(
    rows: list[dict],
    indices: list[int],
    regression_costs: np.ndarray,
    pairwise_costs: np.ndarray,
    regression_aggregation: str,
    pairwise_aggregation: str,
    blend_alpha: float,
) -> list[dict[str, object]]:
    if not 0.0 <= blend_alpha <= 1.0:
        raise ValueError("blend_alpha must be between zero and one")
    regression = aggregate_seed_costs(regression_costs, regression_aggregation)
    pairwise = aggregate_seed_costs(pairwise_costs, pairwise_aggregation)
    if len(regression) != len(indices) or len(pairwise) != len(indices):
        raise ValueError("Model scores do not match the selected candidate pool")
    blended = (
        blend_alpha * fractional_ranks(regression)
        + (1.0 - blend_alpha) * fractional_ranks(pairwise)
    )
    return sorted(
        (
            {
                "row": rows[index],
                "candidate_index": index,
                "score": float(score),
                "regression_score": float(regression[local_index]),
                "regression_median_mm": float(
                    np.median(regression_costs[:, local_index])
                ),
                "pairwise_score": float(pairwise[local_index]),
            }
            for local_index, (index, score) in enumerate(zip(indices, blended))
        ),
        key=lambda item: float(item["score"]),
    )


def unsupervised_candidates(
    rows: list[dict], indices: list[int], score_key: str
) -> list[dict[str, object]]:
    if score_key not in ("rank_score_mm", "selection_score_mm"):
        raise ValueError(f"Unsupported unsupervised score key: {score_key}")
    missing = [index for index in indices if score_key not in rows[index]]
    if missing:
        raise ValueError(
            f"Unsupervised deployment score {score_key} is missing for "
            f"{len(missing)} candidates"
        )
    return sorted(
        (
            {
                "row": rows[index],
                "candidate_index": index,
                "score": float(rows[index][score_key]),
                "regression_score": float(rows[index][score_key]),
                "regression_median_mm": float(rows[index][score_key]),
                "pairwise_score": float(rows[index][score_key]),
            }
            for index in indices
        ),
        key=lambda item: float(item["score"]),
    )


def pair_deviation(
    upper: dict[str, object],
    lower: dict[str, object],
    relative_rotation: np.ndarray,
    relative_translation: np.ndarray,
) -> tuple[float, float]:
    upper_transform = np.asarray(upper["row"]["transform"], dtype=np.float64)
    lower_transform = np.asarray(lower["row"]["transform"], dtype=np.float64)
    relative = (
        proper_protocol_rotation(upper_transform[:3, :3]).T
        @ proper_protocol_rotation(lower_transform[:3, :3])
    )
    angle = float(
        np.degrees(
            Rotation.from_matrix(np.asarray(relative_rotation).T @ relative).magnitude()
        )
    )
    translation = (
        upper_transform[:3, 3]
        - lower_transform[:3, 3]
        - np.asarray(relative_translation, dtype=np.float64)
    )
    return angle, float(np.linalg.norm(translation))


def select_joint_pair(
    upper: list[dict[str, object]],
    lower: list[dict[str, object]],
    relative_rotation: np.ndarray,
    relative_translation: np.ndarray,
    top_k: int,
    angle_weight: float,
    translation_weight: float,
    allow_chirality_mismatch: bool = False,
) -> dict[str, object]:
    pairs = []
    for upper_item, lower_item in itertools.product(upper[:top_k], lower[:top_k]):
        if (
            not allow_chirality_mismatch
            and int(upper_item["row"].get("chirality", 1))
            != int(lower_item["row"].get("chirality", 1))
        ):
            continue
        angle, translation = pair_deviation(
            upper_item,
            lower_item,
            relative_rotation,
            relative_translation,
        )
        objective = (
            float(upper_item["score"])
            + float(lower_item["score"])
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
    if not pairs and top_k < max(len(upper), len(lower)):
        return select_joint_pair(
            upper,
            lower,
            relative_rotation,
            relative_translation,
            max(len(upper), len(lower)),
            angle_weight,
            translation_weight,
            allow_chirality_mismatch,
        )
    if not pairs:
        raise RuntimeError("No chirality-consistent upper/lower candidate pair")
    return min(pairs, key=lambda item: float(item["objective"]))


def ensemble_feature_matrix(
    rows: list[dict],
    prior,
    jaw: str,
    model_payload: dict,
) -> np.ndarray:
    """Build exactly the feature layout recorded in a deployment payload."""
    if not model_payload.get("group_context_features", False):
        raise ValueError("Deployment ensembles require group-context features")
    include_roi_view = bool(model_payload.get("roi_view_feature", False))
    if model_payload.get("modality_features", False):
        features = candidate_multimodal_group_features(
            rows, prior, jaw, include_roi_view=include_roi_view
        )
        available_names = (
            MULTIMODAL_ROI_GROUP_FEATURE_NAMES
            if include_roi_view
            else MULTIMODAL_GROUP_FEATURE_NAMES
        )
    else:
        features = candidate_group_features(
            rows, prior, jaw, include_roi_view=include_roi_view
        )
        available_names = ROI_GROUP_FEATURE_NAMES if include_roi_view else GROUP_FEATURE_NAMES
    requested_names = tuple(model_payload.get("feature_names", ()))
    if requested_names and requested_names != available_names:
        positions = {name: index for index, name in enumerate(available_names)}
        missing = [name for name in requested_names if name not in positions]
        if missing:
            raise ValueError(f"Deployment model requests unavailable features: {missing}")
        features = features[:, [positions[name] for name in requested_names]]
    return features


def rank_ensemble_candidates(
    rows: list[dict],
    jaw: str,
    prior,
    regression_payload: dict,
    pairwise_payload: dict,
    policy: dict,
) -> list[dict[str, object]]:
    """Apply the fitted regression/pairwise rank blend to one jaw."""
    regression_names = tuple(regression_payload.get("feature_names", ()))
    pairwise_names = tuple(pairwise_payload.get("feature_names", ()))
    if regression_names and pairwise_names and regression_names != pairwise_names:
        raise ValueError("Regression and pairwise ensembles use different features")
    configuration = regression_payload["configuration"]
    budget = int(configuration.get("eval_top_candidates", 20))
    indices = candidate_indices(
        rows,
        jaw,
        budget,
        bool(regression_payload.get("balance_candidate_runs", False)),
        bool(regression_payload.get("exclude_upper_opposite_axial", False)),
    )
    if not indices:
        raise RuntimeError(f"No deployment candidates remain for {jaw}")
    score_mode = str(policy.get("candidate_score_mode", "learned_blend"))
    if score_mode == "unsupervised":
        return unsupervised_candidates(
            rows,
            indices,
            str(policy.get("unsupervised_score_key", "rank_score_mm")),
        )
    if score_mode != "learned_blend":
        raise ValueError(f"Unsupported deployment candidate score mode: {score_mode}")
    features = ensemble_feature_matrix(rows, prior, jaw, regression_payload)
    regression = regression_seed_costs(
        regression_payload["models"],
        features,
        indices,
        jaw,
        str(configuration.get("target_transform", "log1p")),
    )
    pairwise = pairwise_seed_costs(
        pairwise_payload["models"],
        features,
        indices,
        jaw,
        int(pairwise_payload["configuration"].get("eval_opponents", 30)),
    )
    return blended_candidates(
        rows,
        indices,
        regression,
        pairwise,
        str(policy["regression_aggregation"]),
        str(policy["pairwise_aggregation"]),
        float(policy["blend_alpha"]),
    )


def select_case_ensemble_pair(
    groups: dict[tuple[str, str], list[dict]],
    case_id: str,
    priors,
    regression_payload: dict,
    pairwise_payload: dict,
    policy: dict,
) -> dict[str, object]:
    ranked = {}
    for jaw in ("upper", "lower"):
        rows = groups.get((case_id, jaw), [])
        if not rows:
            raise RuntimeError(f"No deployment candidates for {case_id} {jaw}")
        ranked[jaw] = rank_ensemble_candidates(
            rows,
            jaw,
            priors[jaw],
            regression_payload,
            pairwise_payload,
            policy,
        )
    return select_joint_pair(
        ranked["upper"],
        ranked["lower"],
        np.asarray(policy["pair_prior_relative_rotation"], dtype=np.float64),
        np.asarray(policy["pair_prior_relative_translation"], dtype=np.float64),
        int(policy["joint_pair_top_k"]),
        float(policy["joint_angle_weight_mm_per_deg"]),
        float(policy["joint_translation_weight"]),
        bool(policy.get("allow_chirality_mismatch", False)),
    )
