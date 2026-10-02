"""Portable journal inputs and strict split/provenance checks.

Patient IDs must be reconciled across institutions before using this module.
Declared exclusion lists are auditable provenance, not proof of upstream model
training. Hashes detect identical image content, not all repeated patients.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


SPLITS = {"train", "val", "test", "external_test", "unlabeled", "pseudo"}
REFERENCES = {"manual", "silver", "none", "pseudo"}
EVALUATION_SPLITS = {"val", "test", "external_test"}


def _exclusions(provenance: dict, identity_key: str, description: str) -> set[str]:
    if not isinstance(provenance.get(identity_key), str) or not provenance[identity_key].strip():
        raise ValueError(f"{description} requires {identity_key}")
    excluded = provenance.get("excluded_patient_ids")
    if not isinstance(excluded, list) or any(not isinstance(item, str) or not item for item in excluded):
        raise ValueError(f"{description} requires an excluded_patient_ids list")
    return set(excluded)


def validate_protocol(records: list[dict[str, Any]]) -> None:
    """Reject split, reference-role and declared learned-model leakage.

    Supply the *complete* experiment manifest, including locked evaluation
    records. Validating a training-only subset cannot discover omitted groups.
    Patient/content groups stay within a partition. Unlabeled and pseudo are
    the same training-pool partition, allowing a budgeted export to accept one
    scan while retaining another scan of that patient as unlabeled. Manual
    train/val/test/external partitions remain distinct. The two available jaws
    of each case must still share the exact split, and accepted pseudo records
    replace their unlabeled versions rather than duplicating them.
    """
    if not isinstance(records, list) or not records:
        raise ValueError("A journal manifest needs a nonempty records list")
    patient_splits: dict[str, str] = {}
    content_splits: dict[str, str] = {}
    identities: set[tuple[str, str, str]] = set()
    case_metadata: dict[tuple[str, str], tuple[str, str]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every journal record must be an object")
        for key in ("case_id", "patient_id", "source", "split", "jaw", "npz_path", "reference_kind"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"Record requires a nonempty {key}")
        split, reference = record["split"], record["reference_kind"]
        if split not in SPLITS or reference not in REFERENCES:
            raise ValueError(f"Unknown split/reference_kind: {split}/{reference}")
        if record["jaw"] not in {"upper", "lower"}:
            raise ValueError("jaw must be upper or lower")
        if reference == "silver" and split not in {"test", "external_test"}:
            raise ValueError("silver references are reserved for evaluation, never core training or validation")
        if split in {"train", "val"} and reference != "manual":
            raise ValueError("train and val require manual references; use split=pseudo for pseudo supervision")
        if split == "unlabeled" and reference != "none":
            raise ValueError("unlabeled records must have reference_kind=none")
        if (split == "pseudo") != (reference == "pseudo"):
            raise ValueError("pseudo split and pseudo reference_kind must be used together")
        identity = record["source"], record["case_id"], record["jaw"]
        if identity in identities:
            raise ValueError(f"Duplicate case/jaw identity: {identity}")
        identities.add(identity)
        case_key = record["source"], record["case_id"]
        metadata = record["patient_id"], split
        if case_key in case_metadata and case_metadata[case_key] != metadata:
            raise ValueError("The two jaws of one case must share patient and split")
        case_metadata[case_key] = metadata
        partition = "unlabeled_pool" if split in {"unlabeled", "pseudo"} else split
        for group, table, name in ((record["patient_id"], patient_splits, "patient"),
                                    (record.get("content_hash"), content_splits, "content hash")):
            if group:
                if not isinstance(group, str):
                    raise ValueError(f"{name} must be a string")
                if group in table and table[group] != partition:
                    raise ValueError(f"{name} overlap across splits: {group}")
                table[group] = partition

    external_sources = {row["source"] for row in records if row["split"] == "external_test"}
    if any(row["source"] in external_sources and row["split"] != "external_test" for row in records):
        raise ValueError("A locked external source may appear only in external_test")
    evaluation_patients = {row["patient_id"] for row in records if row["split"] in EVALUATION_SPLITS}
    for record in records:
        provenance = record.get("candidate_provenance", {"kind": "none"})
        if not isinstance(provenance, dict):
            raise ValueError("candidate_provenance must be an object")
        kind = provenance.get("kind")
        if kind not in {"none", "geometry_only", "learned"}:
            raise ValueError("Unknown candidate provenance; declare none, geometry_only or learned")
        if kind == "learned":
            excluded = _exclusions(provenance, "model_id", "Learned candidate provenance")
            required = evaluation_patients | {record["patient_id"]}
            if not required.issubset(excluded):
                raise ValueError(f"Learned candidates must exclude own and all evaluation patients: {sorted(required - excluded)}")
        if record["reference_kind"] == "pseudo":
            provenance = record.get("pseudo_provenance")
            if not isinstance(provenance, dict):
                raise ValueError("pseudo supervision requires pseudo_provenance")
            excluded = _exclusions(provenance, "teacher_id", "pseudo provenance")
            if not evaluation_patients.issubset(excluded):
                raise ValueError(f"Pseudo teacher must exclude every evaluation patient: {sorted(evaluation_patients - excluded)}")
            if record.get("pseudo_kind", "field") not in {"field", "support"}:
                raise ValueError("pseudo_kind must be field or support")


def load_journal_manifest(path: str | Path, data_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Resolve relative NPZ paths against manifest parent or a relocation root."""
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported journal manifest schema_version (expected 1)")
    records = payload.get("records")
    validate_protocol(records)
    root = Path(data_root).expanduser().resolve() if data_root is not None else path.parent
    resolved = []
    for record in records:
        row = dict(record)
        npz_path = Path(row["npz_path"]).expanduser()
        row["npz_path"] = str((root / npz_path).resolve())
        resolved.append(row)
    return resolved


def _array(value: Any, name: str, dimensions: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    if dimensions is not None and array.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    return array


def _matrix(value: Any, name: str, rigid: bool = False) -> np.ndarray:
    matrix = _array(value, name, 2)
    if matrix.shape != (4, 4) or not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-6):
        raise ValueError(f"{name} must be a homogeneous 4x4 matrix")
    if abs(np.linalg.det(matrix[:3, :3])) < 1e-10:
        raise ValueError(f"{name} must be invertible")
    if rigid and not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=5e-3):
        raise ValueError(f"{name} must be rigid (proper or reflected)")
    return matrix


def load_case(record: dict[str, Any]) -> dict[str, np.ndarray]:
    """Load and validate physical arrays without pickle or implicit GT lookup."""
    validate_protocol([record])
    with np.load(record["npz_path"], allow_pickle=False) as archive:
        case = {key: archive[key].copy() for key in archive.files}
    for key in ("image", "affine", "points", "anchors"):
        if key not in case:
            raise ValueError(f"Missing case array: {key}")
    image = _array(case["image"], "image", 3)
    if min(image.shape) < 2:
        raise ValueError("image dimensions must each contain at least two voxels")
    _matrix(case["affine"], "affine")
    for key in ("points", "anchors"):
        points = _array(case[key], key, 2)
        if points.shape[1] != 3 or len(points) < 3:
            raise ValueError(f"{key} must be Nx3 with at least three points")
    reference = record["reference_kind"]
    if reference == "none" and ("transform" in case or "label" in case):
        raise ValueError("An unlabeled/reference-none case cannot contain a transform or GT label")
    support_pseudo = reference == "pseudo" and record.get("pseudo_kind", "field") == "support"
    if reference in {"manual", "silver", "pseudo"} and not support_pseudo and "transform" not in case:
        raise ValueError(f"{reference} field supervision requires transform")
    if "transform" in case:
        _matrix(case["transform"], "transform", rigid=True)
    if "label" in case:
        labels = _array(case["label"], "label", 3)
        if labels.shape != image.shape or not np.isin(labels, (0, 1, 2)).all():
            raise ValueError("label must match image shape with values 0, 1, 2")
    if support_pseudo and ("label" not in case or "transform" in case):
        raise ValueError("Support pseudo supervision requires label and no transform")
    if reference == "pseudo" and not support_pseudo:
        if "point_weights" not in case or "label" in case:
            raise ValueError("Field pseudo supervision needs point_weights and must not invent background label")
        radius = float(record.get("supervision_radius_mm", 2.0))
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("supervision_radius_mm must be positive")
        case["supervision_radius_mm"] = np.asarray(radius, dtype=np.float32)
    if "point_weights" in case:
        weights = _array(case["point_weights"], "point_weights", 1)
        if len(weights) != len(case["points"]) or np.any((weights < 0) | (weights > 1)) or not np.any(weights > 0):
            raise ValueError("point_weights must match points, lie in [0,1], and include positive mass")
    if "candidates" in case:
        candidates = _array(case["candidates"], "candidates", 3)
        if candidates.shape[1:] != (4, 4):
            raise ValueError("candidates must be Kx4x4")
        if len(candidates) and record.get("candidate_provenance", {}).get("kind", "none") == "none":
            raise ValueError("Candidate transforms require explicit candidate provenance")
        for candidate in candidates:
            _matrix(candidate, "candidate transform", rigid=True)
    return case


def sample_queries(
    case: dict[str, np.ndarray], count: int, rng: np.random.Generator,
    truncation_mm: float = 8.0,
) -> dict[str, np.ndarray]:
    """Draw surface/near/candidate/outer/uniform physical TUDF supervision.

    Ratios are 20/40/15/15/10 percent; absent candidates use near-surface draws.
    All reference distances use transformed points in millimetres. Pseudo cases
    loaded above mask unknown regions beyond their supervision radius.
    """
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("query count must be a positive integer")
    if not np.isfinite(truncation_mm) or truncation_mm <= 0:
        raise ValueError("truncation_mm must be positive")
    if "transform" not in case:
        raise ValueError("Distance queries need an explicit reference transform")
    points = np.asarray(case["points"], dtype=np.float64)
    transform = np.asarray(case["transform"], dtype=np.float64)
    surface = points @ transform[:3, :3].T + transform[:3, 3]
    amounts = np.floor(np.asarray((0.20, 0.40, 0.15, 0.15)) * count).astype(int)
    surface_count, near_count, difficult_count, outer_count = amounts
    uniform_count = count - int(amounts.sum())

    def draw_surface(number: int) -> np.ndarray:
        return surface[rng.integers(len(surface), size=number)].copy()

    near = draw_surface(near_count) + rng.normal(0.0, min(1.0, truncation_mm / 3), (near_count, 3))
    candidates = case.get("candidates")
    if candidates is not None and len(candidates):
        selected = np.asarray(candidates)[rng.integers(len(candidates), size=difficult_count)]
        source = points[rng.integers(len(points), size=difficult_count)]
        difficult = np.einsum("nij,nj->ni", selected[:, :3, :3], source) + selected[:, :3, 3]
    else:
        difficult = draw_surface(difficult_count) + rng.normal(0.0, 2.0, (difficult_count, 3))
    outer = draw_surface(outer_count) + rng.normal(0.0, truncation_mm, (outer_count, 3))
    voxel = rng.random((uniform_count, 3)) * (np.asarray(case["image"].shape) - 1)
    affine = np.asarray(case["affine"], dtype=np.float64)
    uniform = voxel @ affine[:3, :3].T + affine[:3, 3]
    queries = np.concatenate((draw_surface(surface_count), near, difficult, outer, uniform)).astype(np.float32)
    # Compute on the actual returned precision so targets match query locations.
    distance, nearest = cKDTree(surface).query(queries.astype(np.float64), workers=1)
    weights = np.asarray(case.get("point_weights", np.ones(len(points))), dtype=np.float32)[nearest]
    # Queries outside the image clamp to its boundary in both field heads. A
    # varying outside-distance target is therefore unrepresentable there; the
    # separate physical ROI penalty handles such points during registration.
    inverse = np.linalg.inv(affine)
    query_voxel = queries.astype(np.float64) @ inverse[:3, :3].T + inverse[:3, 3]
    in_roi = ((query_voxel >= -1e-5) &
              (query_voxel <= np.asarray(case["image"].shape) - 1 + 1e-5)).all(axis=1)
    weights = weights * in_roi
    if "supervision_radius_mm" in case:
        weights = weights * (distance <= float(case["supervision_radius_mm"]))
    order = rng.permutation(count)
    return {
        "points_world": queries[order],
        "targets": np.minimum(distance, truncation_mm).astype(np.float32)[order],
        "weights": weights.astype(np.float32)[order],
        "in_roi": in_roi[order],
    }
