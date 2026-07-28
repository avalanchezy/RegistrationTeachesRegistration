from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .priors import RotationPrior, proper_protocol_rotation
from .metrics import rotation_error_deg


REGISTRATION_TARGET_NAMES = (
    "mean_tre_mm",
    "translation_error_mm",
    "rotation_error_deg",
    "official_balanced_error",
)


def enrich_candidate_registration_metrics(groups: dict, records) -> None:
    """Attach label-derived official metrics to in-memory training candidates."""
    ground_truth = {
        (record.case_id, record.jaw): np.load(record.transform_path, allow_pickle=False)
        for record in records
        if record.split == "Train-Labeled" and record.transform_path
    }
    for key, rows in groups.items():
        if key not in ground_truth:
            continue
        truth = np.asarray(ground_truth[key], dtype=np.float64)
        for row in rows:
            transform = np.asarray(row["transform"], dtype=np.float64)
            translation = float(np.linalg.norm(transform[:3, 3] - truth[:3, 3]))
            rotation = rotation_error_deg(transform, truth)
            row["translation_error_mm"] = translation
            row["rotation_error_deg"] = rotation
            row["official_balanced_error"] = 0.5 * (
                translation / 10.0 + rotation / 5.0
            )


def registration_target_value(candidate: dict, target: str) -> float:
    if target not in REGISTRATION_TARGET_NAMES:
        raise ValueError(f"Unknown registration optimization target: {target}")
    if target not in candidate:
        raise ValueError(f"Candidate is missing registration target {target}")
    return float(candidate[target])


CROWN_CONSISTENCY_FEATURE_NAMES = (
    "crown_source_trim20",
    "crown_target_trim20",
    "crown_symmetric_trim20",
    "crown_source_median",
    "crown_target_median",
    "crown_source_overlap_2mm",
    "crown_target_coverage_2mm",
    "crown_centroid_error_scaled",
    "crown_consistency_available",
)

CROWN_REFINEMENT_FEATURE_NAMES = (
    "crown_refinement_initial_trim20",
    "crown_refinement_improvement",
    "crown_refinement_center_motion_scaled",
    "crown_refinement_angle_scaled",
    "crown_refinement_alpha",
    "crown_refinement_available",
)

TARGET_MODE_FEATURE_NAMES = (
    "target_crown_mask",
    "target_crown_probability",
    "target_crown_guided",
)


FEATURE_NAMES = (
    "selection_score",
    "source_fit",
    "initial_fit",
    "fit_median",
    "fit_p90",
    "overlap_2mm",
    "target_fit",
    "target_coverage_1mm",
    "target_coverage_2mm",
    "correspondences_scaled",
    "prior_angle_scaled",
    "center_dx_scaled",
    "center_dy_scaled",
    "center_dz_scaled",
    "center_distance_scaled",
    "threshold_scaled",
    "component_log_voxels_scaled",
    "target_aggregate",
    "aggregate_count_scaled",
    "target_adaptive",
    "axial_assignment_requested",
    "axial_assignment_opposite",
    "source_fraction",
    "source_low",
    "source_high",
    "method_basin",
    "chirality_negative",
    "full_trim_10",
    "full_trim_20",
    "full_trim_35",
    "full_distance_median",
    "full_distance_p90",
    "full_overlap_1mm",
    "full_overlap_2mm",
    "full_overlap_3mm",
    "full_target_coverage_2mm",
    "full_normal_abs_cosine",
    "full_geometry_available",
    *tuple(f"proper_rotation_{row}{column}" for row in range(3) for column in range(3)),
    *CROWN_CONSISTENCY_FEATURE_NAMES,
    *CROWN_REFINEMENT_FEATURE_NAMES,
    *TARGET_MODE_FEATURE_NAMES,
)


CONTEXT_FEATURE_NAMES = (
    "context_selection_rank",
    "context_source_fit_rank",
    "context_target_fit_rank",
    "context_full_trim20_rank",
    "context_full_median_rank",
    "context_full_p90_rank",
    "context_prior_angle_rank",
    "context_selection_robust_z",
    "context_source_fit_robust_z",
    "context_target_fit_robust_z",
    "context_full_trim20_robust_z",
    "context_full_median_robust_z",
    "context_full_p90_robust_z",
    "context_prior_angle_robust_z",
    "context_same_target_fraction",
    "context_same_source_fraction",
    "context_same_target_source_fraction",
    "context_opposite_axial_target",
    "context_jaw_upper",
    "context_transform_nearest_distance",
    "context_transform_density_0p10",
    "context_transform_density_0p25",
    "context_cross_run_available",
    "context_cross_run_nearest_distance",
    "context_cross_run_mean_nearest_distance",
    "context_cross_run_support_0p10",
    "context_cross_run_support_0p25",
    "context_crown_symmetric_rank",
    "context_crown_centroid_rank",
    "context_crown_overlap_deficit_rank",
    "context_crown_coverage_deficit_rank",
    "context_crown_symmetric_robust_z",
    "context_crown_centroid_robust_z",
    "context_crown_refinement_gain_rank",
    "context_crown_refinement_motion_rank",
)


GROUP_FEATURE_NAMES = (*FEATURE_NAMES, *CONTEXT_FEATURE_NAMES)
ROI_GROUP_FEATURE_NAMES = (
    *GROUP_FEATURE_NAMES,
    "context_roi_view",
    "context_roi_low_profile",
)

TOOTHSEG_FEATURE_NAMES = (
    "target_toothseg",
    "toothseg_detected_fraction",
    "toothseg_crown_fraction",
)
MULTIMODAL_FEATURE_NAMES = (*FEATURE_NAMES, *TOOTHSEG_FEATURE_NAMES)
MULTIMODAL_GROUP_FEATURE_NAMES = (*GROUP_FEATURE_NAMES, *TOOTHSEG_FEATURE_NAMES)
MULTIMODAL_ROI_GROUP_FEATURE_NAMES = (
    *ROI_GROUP_FEATURE_NAMES,
    *TOOTHSEG_FEATURE_NAMES,
)


def _number(candidate: dict, name: str, default: float = 0.0) -> float:
    value = candidate.get(name, default)
    return float(value) if value not in (None, "") else default


def candidate_features(candidate: dict, prior: RotationPrior) -> np.ndarray:
    transform = np.asarray(candidate["transform"], dtype=np.float64)
    proper = proper_protocol_rotation(transform[:3, :3])
    predicted_center = np.asarray(candidate["predicted_full_centroid"], dtype=np.float64)
    reference_center = np.asarray(candidate["jaw_reference_center"], dtype=np.float64)
    center_offset = (predicted_center - reference_center) / 50.0
    metadata = candidate.get("target_metadata", {})
    target_mode = str(metadata.get("mode", ""))
    candidate_jaw = str(candidate.get("candidate_jaw", ""))
    axial_assignment = str(metadata.get("axial_assignment", ""))
    if not axial_assignment and target_mode == "threshold":
        axial_assignment = candidate_jaw
    source_name = str(candidate.get("source_variant", ""))
    try:
        source_fraction = float(source_name.rsplit("_", 1)[-1])
    except ValueError:
        source_fraction = 1.0
    threshold = float(metadata.get("threshold", 0.0)) / 2000.0
    aggregate_indicator = float("aggregate" in target_mode)
    aggregate_parameter = float(metadata.get("aggregate_count", 0.0)) / 4.0
    if target_mode == "crown_probability":
        # Reuse the dimension-stable generic target-parameter slots so models
        # can distinguish probability thresholds from top-voxel targets.
        threshold = float(metadata.get("minimum_probability", 0.0))
        if metadata.get("selection") == "top_voxels":
            aggregate_indicator = 1.0
            aggregate_parameter = float(metadata.get("requested_voxels", 0.0)) / 4000.0
    elif target_mode == "crown_guided_cbct":
        # The primary slot remains HU/2000; the secondary slot carries the ROI
        # radius. Target-mode one-hot features disambiguate these semantics.
        aggregate_parameter = float(metadata.get("guidance_radius_mm", 0.0)) / 6.0
    voxels = np.log1p(
        float(metadata.get("component_voxels", metadata.get("support_voxels", 0.0)))
    ) / 12.0
    values = [
        _number(candidate, "selection_score_mm"),
        _number(candidate, "fit_score_mm"),
        _number(candidate, "fit_score_initial_mm"),
        _number(candidate, "fit_median_mm"),
        _number(candidate, "fit_p90_mm"),
        _number(candidate, "overlap_2mm"),
        _number(candidate, "target_trimmed_score_mm"),
        _number(candidate, "target_coverage_1mm"),
        _number(candidate, "target_coverage_2mm"),
        _number(candidate, "correspondences") / 5000.0,
        prior.angle_deg(transform[:3, :3]) / 180.0,
        *center_offset.tolist(),
        float(np.linalg.norm(center_offset)),
        threshold,
        voxels,
        aggregate_indicator,
        aggregate_parameter,
        float("adaptive" in target_mode),
        float(bool(axial_assignment) and axial_assignment == candidate_jaw),
        float(bool(axial_assignment) and axial_assignment != candidate_jaw),
        source_fraction,
        float("low" in source_name),
        float("high" in source_name),
        float("basin" in str(candidate.get("method", ""))),
        float(np.linalg.det(transform[:3, :3]) < 0),
        _number(candidate, "full_trim_10_mm", 10.0),
        _number(candidate, "full_trim_20_mm", 10.0),
        _number(candidate, "full_trim_35_mm", 10.0),
        _number(candidate, "full_distance_median_mm", 20.0),
        _number(candidate, "full_distance_p90_mm", 40.0),
        _number(candidate, "full_overlap_1mm", 0.0),
        _number(candidate, "full_overlap_2mm", 0.0),
        _number(candidate, "full_overlap_3mm", 0.0),
        _number(candidate, "full_target_coverage_2mm", 0.0),
        _number(candidate, "full_normal_abs_cosine", 0.0),
        float("full_trim_20_mm" in candidate),
        *proper.reshape(-1).tolist(),
        _number(candidate, "crown_source_trim20_mm", 10.0),
        _number(candidate, "crown_target_trim20_mm", 10.0),
        _number(candidate, "crown_symmetric_trim20_mm", 10.0),
        _number(candidate, "crown_source_median_mm", 20.0),
        _number(candidate, "crown_target_median_mm", 20.0),
        _number(candidate, "crown_source_overlap_2mm", 0.0),
        _number(candidate, "crown_target_coverage_2mm", 0.0),
        _number(candidate, "crown_centroid_error_mm", 50.0) / 50.0,
        float("crown_symmetric_trim20_mm" in candidate),
        _number(candidate, "crown_refinement_initial_trim20_mm", 10.0),
        _number(candidate, "crown_refinement_improvement_mm", 0.0),
        _number(candidate, "crown_refinement_center_motion_mm", 0.0) / 5.0,
        _number(candidate, "crown_refinement_angle_deg", 0.0) / 10.0,
        _number(candidate, "crown_refinement_alpha", 0.0),
        float("crown_refinement_improvement_mm" in candidate),
        float(target_mode == "crown_localizer"),
        float(target_mode == "crown_probability"),
        float(target_mode == "crown_guided_cbct"),
    ]
    features = np.asarray(values, dtype=np.float64)
    if features.shape != (len(FEATURE_NAMES),):
        raise AssertionError(f"Feature shape {features.shape} != {(len(FEATURE_NAMES),)}")
    return features


def toothseg_features(candidate: dict) -> np.ndarray:
    metadata = candidate.get("target_metadata", {})
    is_toothseg = str(metadata.get("mode", "")) == "toothseg"
    return np.asarray(
        [
            float(is_toothseg),
            float(metadata.get("detected_teeth", 0.0)) / 16.0 if is_toothseg else 0.0,
            float(metadata.get("crown_fraction", 0.0)) if is_toothseg else 0.0,
        ],
        dtype=np.float64,
    )


def candidate_multimodal_features(candidate: dict, prior: RotationPrior) -> np.ndarray:
    """Append modality-specific quality signals without changing legacy models."""
    return np.concatenate((candidate_features(candidate, prior), toothseg_features(candidate)))


def is_opposite_axial_target(candidate: dict, jaw: str) -> bool:
    """Return whether a target was explicitly derived from the other arch."""
    metadata = candidate.get("target_metadata", {})
    assignment = str(metadata.get("axial_assignment", ""))
    if assignment:
        return assignment in ("upper", "lower") and assignment != jaw
    opposite = "lower" if jaw == "upper" else "upper"
    return f"axial_{opposite}_" in str(candidate.get("target", ""))


def _fractional_rank(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float64)
    ordered = np.sort(values)
    return np.searchsorted(ordered, values, side="left") / float(len(values) - 1)


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    q25, q75 = np.quantile(values, (0.25, 0.75))
    scale = max(float(q75 - q25), 1e-6)
    return np.clip((values - median) / scale, -8.0, 8.0)


def candidate_group_features(
    candidates: list[dict],
    prior: RotationPrior,
    jaw: str,
    include_roi_view: bool = False,
) -> np.ndarray:
    """Add within-case ranks, frequencies, and transform-consensus context."""
    if not candidates:
        names = ROI_GROUP_FEATURE_NAMES if include_roi_view else GROUP_FEATURE_NAMES
        return np.empty((0, len(names)), dtype=np.float64)
    base = np.stack([candidate_features(candidate, prior) for candidate in candidates])
    contextual_columns = (0, 1, 6, 28, 30, 31, 10)
    ranks = np.column_stack(
        [_fractional_rank(base[:, column]) for column in contextual_columns]
    )
    robust = np.column_stack([_robust_z(base[:, column]) for column in contextual_columns])

    targets = [str(candidate.get("target", "")) for candidate in candidates]
    sources = [str(candidate.get("source_variant", "")) for candidate in candidates]
    target_counts = {value: targets.count(value) for value in set(targets)}
    source_counts = {value: sources.count(value) for value in set(sources)}
    pair_counts = {
        value: sum(pair == value for pair in zip(targets, sources))
        for value in set(zip(targets, sources))
    }
    count_scale = float(len(candidates))
    frequencies = np.asarray(
        [
            (
                target_counts[target] / count_scale,
                source_counts[source] / count_scale,
                pair_counts[(target, source)] / count_scale,
            )
            for target, source in zip(targets, sources)
        ],
        dtype=np.float64,
    )
    flags = np.asarray(
        [
            (float(is_opposite_axial_target(candidate, jaw)), float(jaw == "upper"))
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    descriptor = np.column_stack((base[:, 11:14], base[:, 38:47]))
    descriptor = np.column_stack([_robust_z(descriptor[:, index]) for index in range(descriptor.shape[1])])
    delta = descriptor[:, None, :] - descriptor[None, :, :]
    distance = np.linalg.norm(delta, axis=2) / np.sqrt(descriptor.shape[1])
    np.fill_diagonal(distance, np.inf)
    nearest = np.min(distance, axis=1) if len(candidates) > 1 else np.zeros(1)
    denominator = float(max(len(candidates) - 1, 1))
    density = np.column_stack(
        (
            np.sum(distance <= 0.10, axis=1) / denominator,
            np.sum(distance <= 0.25, axis=1) / denominator,
        )
    )
    run_names = [
        str(candidate.get("source_candidate_run", candidate.get("candidate_run", "")))
        for candidate in candidates
    ]
    unique_runs = sorted(set(run_names))
    cross_run_context = np.zeros((len(candidates), 5), dtype=np.float64)
    cross_run_context[:, 1:3] = 8.0
    if len(unique_runs) > 1:
        run_array = np.asarray(run_names, dtype=object)
        run_positions = {name: index for index, name in enumerate(unique_runs)}
        minimum_distance_by_run = np.column_stack(
            [
                np.min(distance[:, run_array == run_name], axis=1)
                for run_name in unique_runs
            ]
        )
        own_run_indices = np.asarray(
            [run_positions[run_name] for run_name in run_names], dtype=np.int64
        )
        other_run_mask = (
            np.arange(len(unique_runs))[None, :] != own_run_indices[:, None]
        )
        nearest_by_other_run = minimum_distance_by_run[other_run_mask].reshape(
            len(candidates), len(unique_runs) - 1
        )
        cross_run_context[:, 0] = 1.0
        cross_run_context[:, 1] = np.min(nearest_by_other_run, axis=1)
        cross_run_context[:, 2] = np.mean(nearest_by_other_run, axis=1)
        cross_run_context[:, 3] = np.mean(
            nearest_by_other_run <= 0.10, axis=1
        )
        cross_run_context[:, 4] = np.mean(
            nearest_by_other_run <= 0.25, axis=1
        )
    crown_symmetric = base[:, FEATURE_NAMES.index("crown_symmetric_trim20")]
    crown_centroid = base[:, FEATURE_NAMES.index("crown_centroid_error_scaled")]
    crown_overlap_deficit = 1.0 - base[:, FEATURE_NAMES.index("crown_source_overlap_2mm")]
    crown_coverage_deficit = 1.0 - base[:, FEATURE_NAMES.index("crown_target_coverage_2mm")]
    crown_gain = base[:, FEATURE_NAMES.index("crown_refinement_improvement")]
    crown_motion = base[:, FEATURE_NAMES.index("crown_refinement_center_motion_scaled")]
    crown_context = np.column_stack(
        (
            _fractional_rank(crown_symmetric),
            _fractional_rank(crown_centroid),
            _fractional_rank(crown_overlap_deficit),
            _fractional_rank(crown_coverage_deficit),
            _robust_z(crown_symmetric),
            _robust_z(crown_centroid),
            _fractional_rank(-crown_gain),
            _fractional_rank(crown_motion),
        )
    )
    context = np.column_stack(
        (
            ranks,
            robust,
            frequencies,
            flags,
            nearest,
            density,
            cross_run_context,
            crown_context,
        )
    )
    if include_roi_view:
        run_names = [
            str(
                candidate.get(
                    "source_candidate_run", candidate.get("candidate_run", "")
                )
            ).lower()
            for candidate in candidates
        ]
        roi_view = np.asarray(
            [
                float(
                    "roi" in run_name
                    or "roi"
                    in str(
                        candidate.get("target_metadata", {}).get(
                            "source_volume_path",
                            candidate.get("target_metadata", {}).get("volume_path", ""),
                        )
                    ).lower()
                )
                for candidate, run_name in zip(candidates, run_names)
            ],
            dtype=np.float64,
        )[:, None]
        low_profile = np.asarray(
            [
                float("lowadaptive" in run_name or "roi_low" in run_name)
                for run_name in run_names
            ],
            dtype=np.float64,
        )[:, None]
        context = np.column_stack((context, roi_view, low_profile))
    features = np.column_stack((base, context))
    names = ROI_GROUP_FEATURE_NAMES if include_roi_view else GROUP_FEATURE_NAMES
    expected = (len(candidates), len(names))
    if features.shape != expected:
        raise AssertionError(f"Group feature shape {features.shape} != {expected}")
    return features


def candidate_multimodal_group_features(
    candidates: list[dict],
    prior: RotationPrior,
    jaw: str,
    include_roi_view: bool = False,
) -> np.ndarray:
    base = candidate_group_features(
        candidates, prior, jaw, include_roi_view=include_roi_view
    )
    names = (
        MULTIMODAL_ROI_GROUP_FEATURE_NAMES
        if include_roi_view
        else MULTIMODAL_GROUP_FEATURE_NAMES
    )
    if not candidates:
        return np.empty((0, len(names)), dtype=np.float64)
    features = np.column_stack((base, np.stack([toothseg_features(row) for row in candidates])))
    if features.shape != (len(candidates), len(names)):
        raise AssertionError(
            f"Multimodal group feature shape {features.shape} != "
            f"{(len(candidates), len(names))}"
        )
    return features


def load_candidate_group(
    run_dirs: list[Path], key: tuple[str, str]
) -> list[dict]:
    case_id, jaw = key
    group_name = f"{case_id}_{jaw}"
    candidates: list[dict] = []
    for run_dir in run_dirs:
        path = run_dir / group_name / "candidates.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            row["candidate_run"] = str(run_dir)
            row["candidate_jaw"] = jaw
        candidates.extend(rows)
    return candidates


def load_candidate_groups(run_dirs: list[Path]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for run_dir in run_dirs:
        for path in sorted(run_dir.glob("*_*/candidates.json")):
            case_id, jaw = path.parent.name.rsplit("_", 1)
            if jaw not in ("upper", "lower"):
                continue
            rows = json.loads(path.read_text(encoding="utf-8"))
            for row in rows:
                row["candidate_run"] = str(run_dir)
                row["candidate_jaw"] = jaw
            groups.setdefault((case_id, jaw), []).extend(rows)
    return groups
