"""Import the original crown-confidence control into a journal experiment.

The legacy selector writes audit.csv, not a selected-cases CSV. Its columns
accepted_filter and selection_score describe filter eligibility and ranking;
actual top-N/content-diverse membership is the copied NPZ set in top_N/fold_F.
Use that copied directory with --selection-csv audit.csv. A raw prediction
label_arrays directory instead imports the eligible pool before the legacy
top-N/diversity selection. --max-cases caps cases, preserving available jaws.

Teacher exclusions are declarations that must be audited against upstream
training; this converter cannot reconstruct a teacher's training history.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.journal.data import load_case, load_journal_manifest, validate_protocol


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Optional journal NPZ relocation root")
    parser.add_argument("--pseudo-label-dir", type=Path, required=True, help="Prefer legacy top_N/fold_F containing selected NPZ files")
    parser.add_argument("--selection-csv", type=Path, required=True, help="The old selector's audit.csv")
    parser.add_argument("--teacher-provenance", type=Path, required=True, help="JSON teacher_id and excluded_patient_ids")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=0, help="0 keeps every eligible selected case")
    parser.add_argument("--source", help="Required when original IDs collide between sources")
    parser.add_argument("--fold", type=int, help="Teacher fold; inferred from fold_F directory or a single-fold CSV")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_selection(path: Path, label_dir: Path, requested_fold: int | None) -> tuple[int, list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "fold", "accepted_filter", "selection_score"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Use legacy audit.csv with columns {sorted(required)}; summary.csv has no case membership")
        rows = list(reader)
    if not rows:
        raise ValueError("Selection audit.csv is empty")
    folds = {int(row["fold"]) for row in rows}
    inferred = re.fullmatch(r"fold_(\d+)", label_dir.name)
    fold = requested_fold
    if fold is None:
        if inferred:
            fold = int(inferred.group(1))
        elif len(folds) == 1:
            fold = next(iter(folds))
        else:
            raise ValueError("Multiple teacher folds in selection CSV; specify --fold or a fold_F label directory")
    elif inferred and int(inferred.group(1)) != fold:
        raise ValueError("--fold disagrees with the selected fold_F label directory")
    if fold not in folds:
        raise ValueError(f"Teacher fold {fold} is absent from selection CSV")
    selected = []
    seen = set()
    for row in rows:
        if int(row["fold"]) != fold:
            continue
        old_id = row["case_id"]
        if not old_id or Path(old_id).name != old_id or old_id in {".", ".."}:
            raise ValueError("Legacy case_id must be a single nonempty filename component")
        if old_id in seen:
            raise ValueError(f"Duplicate audit case in fold {fold}: {old_id}")
        seen.add(old_id)
        if row["accepted_filter"] not in {"0", "1"}:
            raise ValueError("Legacy accepted_filter must be 0 or 1")
        score = float(row["selection_score"])
        if not np.isfinite(score):
            raise ValueError("selection_score must be finite")
        if row["accepted_filter"] == "1" and (label_dir / f"{old_id}.npz").is_file():
            selected.append({"original_case_id": old_id, "selection_score": score})
    return fold, sorted(selected, key=lambda row: (-row["selection_score"], row["original_case_id"]))


def _teacher(path: Path, records: list[dict], fold: int) -> dict:
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict) or not isinstance(provenance.get("teacher_id"), str) or not provenance["teacher_id"].strip():
        raise ValueError("Teacher provenance needs a nonempty teacher_id")
    excluded = provenance.get("excluded_patient_ids")
    if not isinstance(excluded, list) or any(not isinstance(value, str) or not value for value in excluded):
        raise ValueError("Teacher provenance needs excluded_patient_ids as a list of patient IDs")
    required = {row["patient_id"] for row in records if row["split"] in {"val", "test", "external_test"}}
    if not required.issubset(set(excluded)):
        raise ValueError(f"Legacy teacher must exclude all evaluation patients: {sorted(required - set(excluded))}")
    if "fold" in provenance and provenance["fold"] != fold:
        raise ValueError("Teacher provenance fold disagrees with selected legacy fold")
    return {**provenance, "excluded_patient_ids": sorted(set(excluded)), "fold": fold}


def _prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if "transform" in archive:
            raise ValueError("A support pseudo-label file must not contain a transform")
        if "label" not in archive or "affine" not in archive:
            raise ValueError("Legacy prediction NPZ requires label and affine")
        labels, affine = archive["label"].copy(), archive["affine"].copy()
    if labels.ndim != 3 or labels.dtype.kind not in "biuf" or not np.isfinite(labels).all() or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("Legacy label must be a finite 3D array using only 0, 1, 2")
    if affine.shape != (4, 4) or affine.dtype.kind not in "biuf" or not np.isfinite(affine).all():
        raise ValueError("Legacy affine must be a finite 4x4 matrix")
    return labels.astype(np.uint8), affine


def import_pseudo_labels(args: argparse.Namespace) -> Path:
    if args.max_cases < 0:
        raise ValueError("--max-cases must be nonnegative")
    records = load_journal_manifest(args.manifest, data_root=args.data_root)
    fold, selection = _read_selection(args.selection_csv, args.pseudo_label_dir, args.fold)
    provenance = _teacher(args.teacher_provenance, records, fold)
    groups: dict[tuple[str, str], list[dict]] = {}
    by_old_id: dict[str, list[tuple[str, str]]] = {}
    for row in records:
        if row["split"] != "unlabeled" or (args.source is not None and row["source"] != args.source):
            continue
        old_id = row.get("original_case_id")
        if not isinstance(old_id, str) or not old_id:
            raise ValueError("Unlabeled records need original_case_id to map legacy predictions")
        key = row["source"], row["case_id"]
        if key in groups and groups[key][0]["original_case_id"] != old_id:
            raise ValueError("Available jaws of one case must have the same original_case_id")
        groups.setdefault(key, []).append(row)
        if key not in by_old_id.setdefault(old_id, []):
            by_old_id[old_id].append(key)
    chosen: dict[tuple[str, str], dict] = {}
    for selected in selection:
        keys = by_old_id.get(selected["original_case_id"], [])
        if len(keys) > 1:
            raise ValueError(f"Legacy ID {selected['original_case_id']} is ambiguous; specify --source (and distinct original IDs within a source)")
        if keys:
            chosen[keys[0]] = selected
    if args.max_cases:
        chosen = dict(list(chosen.items())[:args.max_cases])
    output_dir = args.output_dir.resolve()
    output_manifest = output_dir / "manifest.json"
    if output_manifest.exists() and not args.overwrite:
        raise FileExistsError(f"Output manifest exists; use --overwrite: {output_manifest}")
    output_records = []
    for row in records:
        copied = dict(row)
        key = row["source"], row["case_id"]
        if row["split"] == "unlabeled" and key in chosen:
            identity = json.dumps((row["source"], row["case_id"], row["jaw"]), ensure_ascii=False)
            name = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24] + ".npz"
            destination = output_dir / "data" / name
            copied.update(split="pseudo", reference_kind="pseudo", pseudo_kind="support",
                          candidate_provenance={"kind": "none"}, pseudo_provenance=provenance,
                          npz_path=str(destination),
                          legacy_support_selection={**chosen[key], "fold": fold, "accepted_filter": 1})
        copied["npz_path"] = os.path.relpath(copied["npz_path"], output_dir)
        output_records.append(copied)
    validate_protocol(output_records)
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_files = []
    with tempfile.TemporaryDirectory(prefix=".support-import-", dir=output_dir) as staging:
        for original, copied in zip(records, output_records):
            key = original["source"], original["case_id"]
            if original["split"] != "unlabeled" or key not in chosen:
                continue
            old_id = chosen[key]["original_case_id"]
            labels, affine = _prediction(args.pseudo_label_dir / f"{old_id}.npz")
            case = load_case(original)
            if labels.shape != case["image"].shape:
                raise ValueError(f"Legacy label shape disagrees with journal image: {original['case_id']}/{original['jaw']}")
            if not np.allclose(affine, case["affine"], rtol=0, atol=1e-5):
                raise ValueError(f"Legacy label affine disagrees with journal grid: {original['case_id']}/{original['jaw']}")
            expected_class = 1 if original["jaw"] == "upper" else 2
            if not np.any(labels == expected_class):
                raise ValueError(f"Legacy label has no support for available {original['jaw']} jaw")
            # Explicit whitelist prevents propagation of any transform-like
            # arrays or incidental diagnostics from the old inference route.
            payload = {name: case[name] for name in ("image", "affine", "points", "anchors")}
            payload["label"] = labels
            destination = (output_dir / copied["npz_path"]).resolve()
            if destination.exists() and not args.overwrite:
                raise FileExistsError(f"Output pseudo NPZ exists; use --overwrite: {destination}")
            temporary = Path(staging) / destination.name
            np.savez_compressed(temporary, **payload)
            load_case({**copied, "npz_path": str(temporary)})
            staged_files.append((temporary, destination))
        for temporary, destination in staged_files:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(destination)
        manifest_payload = {"schema_version": 1, "records": output_records}
        temporary_manifest = Path(staging) / "manifest.json"
        temporary_manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
        temporary_manifest.replace(output_manifest)
    summary = dict(teacher_id=provenance["teacher_id"], fold=fold, accepted_cases=len(chosen),
                   accepted_jaws=len(staged_files), max_cases=args.max_cases,
                   selection_rule="accepted_filter=1 AND copied NPZ exists; selection_score descending, case_id tie break")
    (output_dir / "import_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Imported {len(chosen)} cases / {len(staged_files)} jaws from teacher fold {fold}: {output_manifest}")
    return output_manifest


def main() -> None:
    import_pseudo_labels(parse_args())


if __name__ == "__main__":
    main()
