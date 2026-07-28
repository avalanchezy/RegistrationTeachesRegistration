from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from task2reg.geometry import (
    bidirectional_fit_score,
    register_geometry,
    transform_points,
)


@dataclass(frozen=True)
class SurfaceTemplateMatch:
    transform: np.ndarray
    reference_case_id: str
    template_kind: str
    teacher_predicted_tre_mm: float
    registration_score_mm: float
    median_distance_mm: float
    p90_distance_mm: float
    overlap_2mm: float
    target_coverage_2mm: float
    chirality: int
    cbct_match_kind: str


def sample_vertices(vertices: np.ndarray, sample_points: int, seed: int) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("vertices must be an Nx3 array with at least three rows")
    if len(vertices) <= sample_points:
        return vertices
    rng = np.random.default_rng(seed)
    return vertices[rng.choice(len(vertices), size=sample_points, replace=False)]


def match_surface_template(
    query_vertices: np.ndarray,
    jaw: str,
    cbct_hash: str,
    entries: list[dict],
    *,
    cbct_payload_hash: str | None = None,
    sample_points: int = 6000,
    seed: int = 2026,
    max_teacher_predicted_tre_mm: float = 1.5,
    pca_refine_top_k: int = 12,
    max_registration_score_mm: float = 0.8,
    max_median_distance_mm: float = 1.0,
    max_p90_distance_mm: float = 2.0,
    min_overlap_2mm: float = 0.95,
    min_target_coverage_2mm: float = 0.95,
) -> SurfaceTemplateMatch | None:
    """Transfer a teacher transform across different IOS tessellations.

    The route is intentionally restricted to byte-identical CBCT volumes. A
    full-arch, bidirectional surface gate must also pass before a teacher
    transform can be transferred to the query IOS coordinate system.
    """

    normalized_hash = str(cbct_hash).lower()
    normalized_payload_hash = (
        str(cbct_payload_hash).lower() if cbct_payload_hash is not None else None
    )
    candidates = [
        entry
        for entry in entries
        if entry["jaw"] == jaw
        and (
            str(entry["cbct_sha256"]).lower() == normalized_hash
            or (
                normalized_payload_hash is not None
                and str(entry.get("cbct_payload_sha256", "")).lower()
                == normalized_payload_hash
            )
        )
        and float(entry.get("predicted_tre_mm", 0.0))
        <= max_teacher_predicted_tre_mm
    ]
    if not candidates:
        return None

    query = sample_vertices(query_vertices, sample_points, seed)
    accepted: list[SurfaceTemplateMatch] = []
    for entry_index, entry in enumerate(candidates):
        reference = np.asarray(entry["reference_points"], dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) < 32:
            continue
        results = register_geometry(
            query,
            reference,
            methods=("pca",),
            # Two IOS exports of the same physical arch differ by a proper
            # rigid transform. Allowing a reflection makes a nearly symmetric
            # dental arch unnecessarily ambiguous and can flip CBCT chirality.
            allow_reflection=False,
            pca_refine_top_k=pca_refine_top_k,
            seed=seed + 1009 * entry_index,
        )
        for result in results[:4]:
            moved = transform_points(query, result.transform)
            bidirectional = bidirectional_fit_score(
                moved,
                reference,
                source_trim_fraction=0.9,
                target_trim_fraction=0.9,
                target_weight=1.0,
            )
            if (
                bidirectional["score"] > max_registration_score_mm
                or result.median_distance > max_median_distance_mm
                or result.p90_distance > max_p90_distance_mm
                or result.overlap_2mm < min_overlap_2mm
                or bidirectional["target_coverage_2mm"] < min_target_coverage_2mm
            ):
                continue
            transferred = (
                np.asarray(entry["reference_transform"], dtype=np.float64)
                @ result.transform
            )
            accepted.append(
                SurfaceTemplateMatch(
                    transform=transferred,
                    reference_case_id=str(entry["case_id"]),
                    template_kind=str(entry.get("template_kind", "surface_teacher")),
                    teacher_predicted_tre_mm=float(entry.get("predicted_tre_mm", 0.0)),
                    registration_score_mm=float(bidirectional["score"]),
                    median_distance_mm=float(result.median_distance),
                    p90_distance_mm=float(result.p90_distance),
                    overlap_2mm=float(result.overlap_2mm),
                    target_coverage_2mm=float(bidirectional["target_coverage_2mm"]),
                    chirality=int(result.chirality),
                    cbct_match_kind=(
                        "raw"
                        if str(entry["cbct_sha256"]).lower() == normalized_hash
                        else "payload"
                    ),
                )
            )

    if not accepted:
        return None
    return min(
        accepted,
        key=lambda item: (
            item.teacher_predicted_tre_mm + item.p90_distance_mm,
            item.registration_score_mm,
            item.p90_distance_mm,
            item.teacher_predicted_tre_mm,
        ),
    )
