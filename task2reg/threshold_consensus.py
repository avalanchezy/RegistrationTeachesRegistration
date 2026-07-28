from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import apply_transform


@dataclass(frozen=True)
class ConsensusConfig:
    radius_mm: float
    max_selection_score_mm: float
    min_views: int
    min_families: int
    min_thresholds: int
    min_source_sides: int
    min_evidence: int
    joint_mode: str = "none"
    max_joint_angle_deg: float = 8.0
    max_joint_translation_mm: float = 10.0


def target_family(candidate: dict) -> str:
    metadata = candidate.get("target_metadata", {})
    mode = str(metadata.get("mode", "")).lower()
    target = str(candidate.get("target", "")).lower()
    value = f"{mode} {target}"
    if "adaptive" in value:
        return "adaptive"
    if "aggregate" in value:
        return "aggregate"
    return "tracked"


def source_side(candidate: dict) -> str:
    value = str(candidate.get("source_variant", "")).lower()
    if "low" in value:
        return "low"
    if "high" in value:
        return "high"
    return "full"


def evidence_key(candidate: dict) -> tuple[str, str, int, str]:
    metadata = candidate.get("target_metadata", {})
    threshold = int(round(float(metadata.get("threshold", 0.0)) / 100.0) * 100)
    return (
        str(candidate.get("candidate_view", Path(candidate.get("candidate_run", "run")).name)),
        target_family(candidate),
        threshold,
        source_side(candidate),
    )


def candidate_quality(candidate: dict) -> tuple[float, float, float, float]:
    return (
        float(candidate.get("selection_score_mm", candidate.get("fit_score_mm", np.inf))),
        float(candidate.get("fit_score_mm", np.inf)),
        float(candidate.get("fit_p90_mm", np.inf)),
        float(candidate.get("target_coverage_2mm", 0.0)),
    )


def prune_candidates(
    candidates: list[dict],
    *,
    max_source_score_mm: float,
    max_source_p90_mm: float,
    min_target_coverage_2mm: float,
    per_evidence: int,
) -> list[dict]:
    cells: dict[tuple[str, str, int, str], list[dict]] = {}
    for candidate in candidates:
        _, source_score, source_p90, coverage = candidate_quality(candidate)
        if (
            source_score <= max_source_score_mm
            and source_p90 <= max_source_p90_mm
            and coverage >= min_target_coverage_2mm
        ):
            cells.setdefault(evidence_key(candidate), []).append(candidate)
    retained = []
    for rows in cells.values():
        retained.extend(sorted(rows, key=candidate_quality)[:per_evidence])
    return retained


def transform_distance_matrix(candidates: list[dict], points: np.ndarray) -> np.ndarray:
    if not candidates:
        return np.empty((0, 0), dtype=np.float32)
    moved = np.stack(
        [apply_transform(points, np.asarray(row["transform"], dtype=np.float64)) for row in candidates]
    ).astype(np.float32)
    count = len(candidates)
    distances = np.empty((count, count), dtype=np.float32)
    for index in range(count):
        distances[index] = np.linalg.norm(moved - moved[index], axis=2).mean(axis=1)
    return distances


def select_consensus(
    candidates: list[dict],
    distances: np.ndarray,
    config: ConsensusConfig,
) -> dict | None:
    if not candidates:
        return None
    selection_scores = np.asarray([candidate_quality(row)[0] for row in candidates])
    determinants = np.asarray(
        [int(np.sign(np.linalg.det(np.asarray(row["transform"])[:3, :3]))) for row in candidates]
    )
    eligible = np.flatnonzero(selection_scores <= config.max_selection_score_mm)
    best: tuple[tuple[float, ...], dict] | None = None
    for center in eligible:
        near = eligible[
            (distances[center, eligible] <= config.radius_mm)
            & (determinants[eligible] == determinants[center])
        ]
        # Repeated refinements of the same evidence cell must count only once.
        by_evidence: dict[tuple[str, str, int, str], int] = {}
        for index in near:
            key = evidence_key(candidates[int(index)])
            previous = by_evidence.get(key)
            if previous is None or distances[center, index] < distances[center, previous]:
                by_evidence[key] = int(index)
        support = np.asarray(list(by_evidence.values()), dtype=np.int64)
        evidence = [evidence_key(candidates[index]) for index in support]
        views = {item[0] for item in evidence}
        families = {item[1] for item in evidence}
        thresholds = {item[2] for item in evidence if item[2] > 0}
        sides = {item[3] for item in evidence}
        if (
            len(support) < config.min_evidence
            or len(views) < config.min_views
            or len(families) < config.min_families
            or len(thresholds) < config.min_thresholds
            or len(sides) < config.min_source_sides
        ):
            continue
        local = distances[np.ix_(support, support)]
        medoid_local = int(np.argmin(local.mean(axis=1)))
        medoid = int(support[medoid_local])
        disagreement = local[np.triu_indices(len(support), 1)]
        mean_disagreement = float(disagreement.mean()) if len(disagreement) else 0.0
        max_disagreement = float(disagreement.max()) if len(disagreement) else 0.0
        diversity = (
            4.0 * len(views)
            + 2.0 * len(families)
            + 0.75 * len(thresholds)
            + 0.35 * len(sides)
            + 0.15 * len(support)
        )
        objective = diversity - mean_disagreement / config.radius_mm - 0.5 * selection_scores[medoid]
        tie_break = (
            objective,
            float(len(views)),
            float(len(families)),
            float(len(thresholds)),
            float(len(support)),
            -mean_disagreement,
            -selection_scores[medoid],
        )
        payload = {
            "candidate": candidates[medoid],
            "support_indices": support.tolist(),
            "support_candidates": [candidates[index] for index in support],
            "consensus_count": int(len(support)),
            "view_count": len(views),
            "family_count": len(families),
            "threshold_count": len(thresholds),
            "source_side_count": len(sides),
            "views": sorted(views),
            "families": sorted(families),
            "thresholds": sorted(thresholds),
            "source_sides": sorted(sides),
            "mean_disagreement_mm": mean_disagreement,
            "max_disagreement_mm": max_disagreement,
            "consensus_objective": float(objective),
        }
        if best is None or tie_break > best[0]:
            best = (tie_break, payload)
    return None if best is None else best[1]


def joint_transform_difference(upper: dict, lower: dict) -> tuple[float, float]:
    upper_transform = np.asarray(upper["candidate"]["transform"], dtype=np.float64)
    lower_transform = np.asarray(lower["candidate"]["transform"], dtype=np.float64)
    if np.sign(np.linalg.det(upper_transform[:3, :3])) != np.sign(
        np.linalg.det(lower_transform[:3, :3])
    ):
        return float("inf"), float("inf")
    relative = upper_transform[:3, :3].T @ lower_transform[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cosine)))
    translation = float(np.linalg.norm(upper_transform[:3, 3] - lower_transform[:3, 3]))
    return angle, translation


def apply_joint_gate(
    selections: dict[tuple[str, str], dict], config: ConsensusConfig
) -> tuple[dict[tuple[str, str], dict], dict[str, dict]]:
    if config.joint_mode not in {"none", "filter", "require"}:
        raise ValueError(f"Unknown joint mode: {config.joint_mode}")
    if config.joint_mode == "none":
        return selections, {}
    accepted = dict(selections)
    diagnostics: dict[str, dict] = {}
    case_ids = sorted({case_id for case_id, _ in selections})
    for case_id in case_ids:
        upper = selections.get((case_id, "upper"))
        lower = selections.get((case_id, "lower"))
        if upper is None or lower is None:
            passed = False
            angle = translation = float("inf")
        else:
            angle, translation = joint_transform_difference(upper, lower)
            passed = (
                angle <= config.max_joint_angle_deg
                and translation <= config.max_joint_translation_mm
            )
        diagnostics[case_id] = {
            "joint_passed": int(passed),
            "joint_angle_deg": angle,
            "joint_translation_mm": translation,
        }
        if config.joint_mode == "require" and not passed:
            accepted.pop((case_id, "upper"), None)
            accepted.pop((case_id, "lower"), None)
        elif config.joint_mode == "filter" and upper is not None and lower is not None and not passed:
            accepted.pop((case_id, "upper"), None)
            accepted.pop((case_id, "lower"), None)
    return accepted, diagnostics
