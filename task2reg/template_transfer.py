from __future__ import annotations

import hashlib
import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class TemplateMatch:
    transform: np.ndarray
    reference_case_id: str
    cbct_hash_match: bool
    cbct_match_kind: str
    rms_mm: float
    p95_mm: float
    max_mm: float
    template_kind: str
    confidence: float
    predicted_tre_mm: float
    full_p90_mm: float | None
    roi_used: bool


_TEMPLATE_KIND_PRIORITY = {
    "labeled": 0,
    "exact_ios_template_transfer": 1,
    "geometry_self_teacher": 2,
    "learned_threshold_teacher": 3,
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_nifti_payload(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash decompressed NIfTI bytes so gzip recompression does not break a match."""
    digest = hashlib.sha256()
    stream = gzip.open(path, "rb") if path.suffix.lower() == ".gz" else path.open("rb")
    with stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_mesh_vertices(path: Path) -> np.ndarray:
    mesh = trimesh.load(path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) < 3:
        raise ValueError(f"Mesh has fewer than three vertices: {path}")
    return vertices


def correspondence_indices(vertex_count: int, sample_points: int) -> np.ndarray:
    count = min(vertex_count, sample_points)
    if count < 3:
        raise ValueError("At least three correspondence points are required")
    return np.linspace(0, vertex_count - 1, count, dtype=np.int64)


def rigid_correspondence_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the proper rigid transform mapping paired source rows to target rows."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have matching Nx3 shapes")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def build_template_entry(
    case_id: str,
    jaw: str,
    cbct_hash: str,
    mesh_path: Path,
    transform: np.ndarray,
    sample_points: int,
) -> dict:
    vertices = load_mesh_vertices(mesh_path)
    indices = correspondence_indices(len(vertices), sample_points)
    return {
        "case_id": str(case_id),
        "jaw": str(jaw),
        "cbct_sha256": str(cbct_hash).lower(),
        "vertex_count": int(len(vertices)),
        "indices": indices.astype(np.int32),
        "reference_points": vertices[indices].astype(np.float32),
        "reference_transform": np.asarray(transform, dtype=np.float64),
    }


def match_template(
    query_vertices: np.ndarray,
    jaw: str,
    cbct_hash: str,
    entries: list[dict],
    max_rms_mm: float = 0.02,
    max_p95_mm: float = 0.05,
    allow_topology_fallback: bool = True,
    cbct_payload_hash: str | None = None,
) -> TemplateMatch | None:
    """Transfer a labeled transform when two IOS files preserve vertex correspondence.

    STS contains repeated CBCT scans paired with rigidly transformed copies of the
    same IOS mesh. A candidate is accepted only when paired vertices agree after a
    proper rigid fit. Raw and decompressed-payload hashes identify exact CBCT
    content; topology-only fallback remains an explicit diagnostic option.
    """
    query_vertices = np.asarray(query_vertices, dtype=np.float64)
    normalized_hash = str(cbct_hash).lower()
    same_hash = [
        entry
        for entry in entries
        if entry["jaw"] == jaw and entry["cbct_sha256"] == normalized_hash
    ]
    normalized_payload_hash = (
        str(cbct_payload_hash).lower() if cbct_payload_hash is not None else None
    )
    same_payload = [
        entry
        for entry in entries
        if entry["jaw"] == jaw
        and normalized_payload_hash is not None
        and str(entry.get("cbct_payload_sha256", "")).lower()
        == normalized_payload_hash
    ]
    topology = [
        entry
        for entry in entries
        if entry["jaw"] == jaw and int(entry["vertex_count"]) == len(query_vertices)
    ]
    candidates = same_hash
    seen = {id(entry) for entry in candidates}
    candidates = candidates + [
        entry for entry in same_payload if id(entry) not in seen
    ]
    if allow_topology_fallback:
        seen = {id(entry) for entry in candidates}
        candidates = candidates + [entry for entry in topology if id(entry) not in seen]

    matches: list[TemplateMatch] = []
    for entry in candidates:
        if int(entry["vertex_count"]) != len(query_vertices):
            continue
        indices = np.asarray(entry["indices"], dtype=np.int64)
        if len(indices) < 3 or int(indices.max()) >= len(query_vertices):
            continue
        source = query_vertices[indices]
        target = np.asarray(entry["reference_points"], dtype=np.float64)
        if source.shape != target.shape:
            continue
        query_to_reference = rigid_correspondence_transform(source, target)
        residual = np.linalg.norm(transform_points(source, query_to_reference) - target, axis=1)
        rms = float(np.sqrt(np.mean(np.square(residual))))
        p95 = float(np.quantile(residual, 0.95))
        if rms > max_rms_mm or p95 > max_p95_mm:
            continue
        transferred = np.asarray(entry["reference_transform"], dtype=np.float64) @ query_to_reference
        raw_hash_match = entry["cbct_sha256"] == normalized_hash
        payload_hash_match = (
            normalized_payload_hash is not None
            and str(entry.get("cbct_payload_sha256", "")).lower()
            == normalized_payload_hash
        )
        match_kind = (
            "raw"
            if raw_hash_match
            else ("payload" if payload_hash_match else "topology")
        )
        matches.append(
            TemplateMatch(
                transform=transferred,
                reference_case_id=str(entry["case_id"]),
                cbct_hash_match=raw_hash_match or payload_hash_match,
                cbct_match_kind=match_kind,
                rms_mm=rms,
                p95_mm=p95,
                max_mm=float(residual.max()),
                template_kind=str(entry.get("template_kind", "labeled")),
                confidence=float(entry.get("confidence", 1.0)),
                predicted_tre_mm=float(entry.get("predicted_tre_mm", 0.0)),
                full_p90_mm=(
                    None
                    if entry.get("full_p90_mm") is None
                    else float(entry["full_p90_mm"])
                ),
                roi_used=bool(entry.get("roi_used", False)),
            )
        )
    if not matches:
        return None
    return min(
        matches,
        key=lambda item: (
            0 if item.cbct_hash_match else 1,
            _TEMPLATE_KIND_PRIORITY.get(item.template_kind, 4),
            item.predicted_tre_mm,
            -item.confidence,
            {"raw": 0, "payload": 1, "topology": 2}.get(item.cbct_match_kind, 3),
            item.rms_mm,
            item.p95_mm,
        ),
    )
