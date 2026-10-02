"""Convert historical RTR inputs into portable journal supervision.

Metadata CSV columns: source,case_id,patient_id,split; optional reference_kind,
original_split,jaw,crown_points_path,prepared_npz_path,content_hash. Patient IDs
are global across sources. crown_points_path is a fixed upstream crop in IOS
coordinates, as NPY or an NPZ with a points array. Neither surface selection nor
evaluation anchors uses GT. Without a fixed crop, full IOS is used and declared.

Candidate provenance JSON is either one declaration, or {"records": [...]} with
source/case_id/jaw plus candidate_provenance for each imported record. A learned
prior, learned support mask, or learned candidate proposal requires kind=learned;
geometry_only certifies that none of those learned inputs were used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.data import load_manifest
from task2reg.journal.data import load_case, validate_protocol
from task2reg.template_transfer import sha256_nifti_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="Existing CaseRecord CSV")
    parser.add_argument("--data-dir", type=Path, required=True, help="Existing crown-prepared NPZ directory")
    parser.add_argument("--metadata", type=Path, required=True, help="Explicit patient/source/split metadata CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", help="Import one metadata source, disambiguating reused case IDs")
    parser.add_argument("--points", type=int, default=12000, help="Fixed surface points per jaw")
    parser.add_argument("--anchors", type=int, default=4096, help="Independent fixed full-IOS evaluation anchors")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--candidate-runs", nargs="*", type=Path, default=[])
    parser.add_argument("--candidate-provenance", type=Path, help="Required JSON provenance when importing candidates")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve()


def _seed(seed: int, identity: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{seed}:{identity}:{purpose}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def sample_mesh_surface(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic area-uniform triangle sampling, independent of vertex density."""
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    usable = np.isfinite(areas) & (areas > 0)
    if not np.any(usable) or not np.isfinite(triangles).all():
        raise ValueError("IOS mesh needs finite triangles with positive area")
    triangles, areas = triangles[usable], areas[usable]
    chosen = triangles[rng.choice(len(triangles), size=count, p=areas / areas.sum())]
    u, v = rng.random((2, count))
    root_u = np.sqrt(u)
    barycentric = np.column_stack((1 - root_u, root_u * (1 - v), root_u * v))
    return np.einsum("ni,nij->nj", barycentric, chosen).astype(np.float32)


def _fixed_crop(path: Path, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
    weights = None
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            points = np.asarray(archive["points"])
            if "point_weights" in archive:
                weights = np.asarray(archive["point_weights"])
    else:
        points = np.load(path, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError(f"Fixed crown points must be finite Nx3 IOS coordinates: {path}")
    if weights is not None and weights.shape != (len(points),):
        raise ValueError("Fixed crop point_weights must match points")
    indices = rng.choice(len(points), size=count, replace=len(points) < count)
    return points[indices].astype(np.float32), None if weights is None else weights[indices].astype(np.float32)


def _provenance(payload: dict, source: str, case_id: str, jaw: str) -> dict:
    if "records" not in payload:
        return dict(payload)
    matching = [row for row in payload["records"]
                if row.get("source") == source and row.get("case_id") in {case_id, f"{source}:{case_id}"}
                and row.get("jaw") == jaw]
    if len(matching) != 1:
        raise ValueError(f"Need exactly one candidate provenance declaration for {source}/{case_id}/{jaw}")
    return dict(matching[0]["candidate_provenance"])


def prepare(args: argparse.Namespace) -> Path:
    """Prepare all selected records, then write the validated relative manifest."""
    if args.points < 3 or args.anchors < 3:
        raise ValueError("--points and --anchors must each be at least three")
    if args.candidate_runs and args.candidate_provenance is None:
        raise ValueError("--candidate-runs requires --candidate-provenance; OOF cannot be inferred")
    with args.metadata.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"source", "case_id", "patient_id", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Metadata CSV needs columns {sorted(required)}")
        metadata = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    metadata = [row for row in metadata if args.source is None or row["source"] == args.source]
    if not metadata:
        raise ValueError("No metadata rows selected")
    for row in metadata:
        if any(not row[key] for key in required):
            raise ValueError("Metadata source/case_id/patient_id/split values must be explicit")
        if row["split"] == "pseudo":
            raise ValueError("Use the verified pseudo exporter or support-pseudo importer, not the manual adapter")
    old_records = load_manifest(args.manifest)
    provenance_payload = (json.loads(args.candidate_provenance.read_text(encoding="utf-8"))
                          if args.candidate_provenance is not None else {})
    if not isinstance(provenance_payload, dict):
        raise ValueError("Candidate provenance must be a JSON object")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = args.output_dir / "manifest.json"
    if output_manifest.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest exists; use --overwrite: {output_manifest}")
    output_records: list[dict[str, Any]] = []
    staged_payloads: list[tuple[Path, Path]] = []
    used_metadata: set[int] = set()
    hash_cache: dict[Path, str] = {}
    for old in old_records:
        if not old.complete:
            continue
        matches = [(index, row) for index, row in enumerate(metadata) if row["case_id"] == old.case_id
                   and (not row.get("original_split") or row["original_split"] == old.split)
                   and (not row.get("jaw") or row["jaw"] == old.jaw)]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError(f"Ambiguous metadata for {old.key}; specify --source, original_split and/or jaw")
        metadata_index, meta = matches[0]
        used_metadata.add(metadata_index)
        source = meta["source"]
        case_id = f"{source}:{old.case_id}"
        identity = f"{case_id}:{old.jaw}"
        # Hash filenames avoid unsafe source/case characters while identities remain readable.
        basename = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24] + ".npz"
        relative_npz = Path("data") / basename
        destination = args.output_dir / relative_npz
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Prepared case exists; use --overwrite: {destination}")
        old_path = (_path(meta["prepared_npz_path"], args.metadata.parent) if meta.get("prepared_npz_path")
                    else args.data_dir / f"{old.case_id}.npz")
        duplicates = {record.split for record in old_records if record.complete and record.case_id == old.case_id}
        if len(duplicates) > 1 and not meta.get("prepared_npz_path"):
            split_path = args.data_dir / old.split / f"{old.case_id}.npz"
            if not split_path.is_file():
                raise ValueError(f"Reused case ID {old.case_id} needs prepared_npz_path or split-specific data directory")
            old_path = split_path
        with np.load(old_path, allow_pickle=False) as archive:
            case = {key: archive[key].copy() for key in ("image", "affine")}
            if "label" in archive and meta["split"] != "unlabeled":
                case["label"] = archive["label"].copy()
            if "label" in archive and meta["split"] == "unlabeled":
                raise ValueError(f"unlabeled case has a prepared GT label: {identity}")
        reference = meta.get("reference_kind") or ("manual" if old.transform_path else "none")
        if meta["split"] == "unlabeled" and old.transform_path:
            raise ValueError(f"unlabeled case has a GT transform path; remove withheld references explicitly: {identity}")
        if reference != "none":
            if not old.transform_path:
                raise ValueError(f"{reference} reference requires a transform path: {identity}")
            case["transform"] = np.load(_path(old.transform_path, args.manifest.parent), allow_pickle=False).astype(np.float64)
        elif old.transform_path:
            raise ValueError(f"reference_kind=none conflicts with a GT transform path: {identity}")
        mesh_path = _path(old.ios_path, args.manifest.parent)
        mesh = trimesh.load(mesh_path, process=False, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        case["anchors"] = sample_mesh_surface(mesh, args.anchors, np.random.default_rng(_seed(args.seed, identity, "anchors")))
        point_rng = np.random.default_rng(_seed(args.seed, identity, "surface"))
        if meta.get("crown_points_path"):
            crop = _path(meta["crown_points_path"], args.metadata.parent)
            case["points"], weights = _fixed_crop(crop, args.points, point_rng)
            if weights is not None:
                case["point_weights"] = weights
            surface_definition = "fixed_upstream_crop"
        else:
            case["points"] = sample_mesh_surface(mesh, args.points, point_rng)
            surface_definition = "full_ios"
            print(f"WARNING {identity}: using full IOS; supply a fixed upstream crown crop for the crown-field study. "
                  "This control is not a crown-only target, and no GT-dependent crop is used.", file=sys.stderr)
        candidates = []
        for run in args.candidate_runs:
            candidate_path = run / f"{old.case_id}_{old.jaw}" / "candidates.json"
            if not candidate_path.is_file():
                raise FileNotFoundError(f"Missing explicitly requested candidates: {candidate_path}")
            rows = json.loads(candidate_path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"Candidate file must be a nonempty list: {candidate_path}")
            candidates.extend(np.asarray(row["transform"], dtype=np.float64) for row in rows)
        candidate_provenance = {"kind": "none"}
        if candidates:
            case["candidates"] = np.stack(candidates)
            candidate_provenance = _provenance(provenance_payload, source, old.case_id, old.jaw)
        content_hash = meta.get("content_hash", "")
        if not content_hash:
            cbct_path = _path(old.cbct_path, args.manifest.parent) if old.cbct_path else None
            if cbct_path is not None and cbct_path.is_file():
                if cbct_path not in hash_cache:
                    hash_cache[cbct_path] = sha256_nifti_payload(cbct_path)
                content_hash = "nifti:" + hash_cache[cbct_path]
            else:
                digest = hashlib.sha256()
                digest.update(np.asarray(case["image"], dtype="<f4").tobytes())
                digest.update(np.asarray(case["image"].shape, dtype="<i8").tobytes())
                digest.update(np.asarray(case["affine"], dtype="<f8").tobytes())
                content_hash = "prepared:" + digest.hexdigest()
        row = dict(case_id=case_id, original_case_id=old.case_id, patient_id=meta["patient_id"],
                   source=source, split=meta["split"], jaw=old.jaw, npz_path=relative_npz.as_posix(),
                   reference_kind=reference, content_hash=content_hash, candidate_provenance=candidate_provenance,
                   surface_definition=surface_definition, anchor_definition="area_uniform_full_ios",
                   sampling_seed=args.seed, original_split=old.split)
        output_records.append(row)
        # Stage one jaw at a time instead of retaining all resampled volumes in
        # RAM. Only the fully validated manifest publishes these staged files.
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **case)
        try:
            load_case({**row, "npz_path": str(temporary.resolve())})
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        staged_payloads.append((temporary, destination))
    missing = [f"{metadata[i]['source']}:{metadata[i]['case_id']}" for i in range(len(metadata)) if i not in used_metadata]
    if missing:
        raise ValueError(f"Metadata rows had no complete matching legacy record: {missing}")
    validate_protocol(output_records)
    # Partial .tmp.npz files from an interrupted conversion are not referenced
    # by a manifest and are safely replaced by the next preparation attempt.
    for temporary, destination in staged_payloads:
        temporary.replace(destination)
    temporary_manifest = output_manifest.with_suffix(".tmp.json")
    temporary_manifest.write_text(json.dumps({"schema_version": 1, "records": output_records}, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(output_manifest)
    print(f"Prepared {len(output_records)} jaw records: {output_manifest}")
    print("Validate the combined experiment manifest again after merging sources; exclusion declarations cannot prove upstream training.")
    return output_manifest


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
