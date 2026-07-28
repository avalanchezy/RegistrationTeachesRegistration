from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.data import load_manifest
from task2reg.template_transfer import (
    build_template_entry,
    load_mesh_vertices,
    match_template,
    sha256_file,
    sha256_nifti_payload,
)


GEOMETRY_TEACHER_METHODS = frozenset({"geometry", "geometry_crown_ensemble"})


def is_geometry_teacher_method(value: object) -> bool:
    return str(value) in GEOMETRY_TEACHER_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append conservative geometry self-teacher entries to a template bank."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-bank", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fingerprint-cache",
        type=Path,
        help="Reuse a prior template-bank fingerprint CSV for unchanged CBCT files.",
    )
    parser.add_argument("--split", default="Train-Unlabeled")
    parser.add_argument("--max-predicted-tre-mm", type=float, default=3.5)
    parser.add_argument("--max-full-p90-mm", type=float, default=26.0)
    parser.add_argument(
        "--max-upper-tracked-predicted-tre-mm",
        type=float,
        default=2.5,
        help="Use a stricter calibrated gate for ambiguous upper single-component targets.",
    )
    parser.add_argument(
        "--missing-geometry-max-predicted-mm",
        type=float,
        default=2.5,
        help="Allow a missing full-P90 only for exceptionally confident predictions.",
    )
    parser.add_argument(
        "--propagate-exact",
        action="store_true",
        help="Add exact same-CBCT IOS topology variants while preserving teacher quality.",
    )
    parser.add_argument("--propagation-max-rms-mm", type=float, default=0.02)
    parser.add_argument("--propagation-max-p95-mm", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bank = joblib.load(args.base_bank)
    entries = list(bank["entries"])
    sample_points = int(bank.get("sample_points", 8192))
    manifest_records = load_manifest(args.manifest)
    records = {
        (record.case_id, record.jaw): record
        for record in manifest_records
        if record.split == args.split and record.complete
    }
    all_records = {
        (record.case_id, record.jaw): record
        for record in manifest_records
        if record.complete
    }
    hash_cache: dict[str, str] = {}
    payload_hash_cache: dict[str, str] = {}
    if args.fingerprint_cache is not None:
        with args.fingerprint_cache.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                cbct_path = str(row["cbct_path"])
                hash_cache[cbct_path] = str(row["cbct_sha256"])
                payload_hash_cache[cbct_path] = str(row["cbct_payload_sha256"])
    accepted = []
    propagated = []
    rejected = []
    seen = {(str(entry["case_id"]), str(entry["jaw"])) for entry in entries}

    for run_dir in args.runs:
        audit_path = run_dir / "audit.json"
        if not audit_path.is_file():
            raise FileNotFoundError(f"Missing audit file: {audit_path}")
        for row in json.loads(audit_path.read_text(encoding="utf-8")):
            key = (str(row["case_id"]), str(row["jaw"]))
            reason = ""
            predicted = row.get("predicted_tre_mm")
            full_p90 = row.get("full_p90_mm")
            if not is_geometry_teacher_method(row.get("method")):
                reason = f"method={row.get('method')}"
            elif key in seen:
                reason = "already_in_bank"
            elif key not in records:
                reason = "manifest_record_missing"
            elif predicted is None or float(predicted) > args.max_predicted_tre_mm:
                reason = "predicted_tre_gate"
            elif (
                key[1] == "upper"
                and "tracked" in str(row.get("target", ""))
                and float(predicted) > args.max_upper_tracked_predicted_tre_mm
            ):
                reason = "upper_tracked_predicted_tre_gate"
            elif full_p90 is None and float(predicted) > args.missing_geometry_max_predicted_mm:
                reason = "missing_full_geometry"
            elif full_p90 is not None and float(full_p90) > args.max_full_p90_mm:
                reason = "full_p90_gate"
            transform_path = run_dir / key[0] / f"{key[1]}_gt.npy"
            if not reason and not transform_path.is_file():
                reason = "transform_missing"
            if reason:
                rejected.append({**row, "reason": reason})
                continue

            record = records[key]
            if record.cbct_path not in hash_cache:
                hash_cache[record.cbct_path] = sha256_file(Path(record.cbct_path))
            transform = np.load(transform_path, allow_pickle=False).astype(np.float64)
            entry = build_template_entry(
                key[0],
                key[1],
                hash_cache[record.cbct_path],
                Path(record.ios_path),
                transform,
                sample_points,
            )
            confidence = float(np.clip(1.0 - float(predicted) / 12.0, 0.0, 1.0))
            entry.update(
                {
                    "template_kind": "geometry_self_teacher",
                    "confidence": confidence,
                    "predicted_tre_mm": float(predicted),
                    "full_p90_mm": None if full_p90 is None else float(full_p90),
                    "roi_used": bool(row.get("roi_used", False)),
                }
            )
            entries.append(entry)
            seen.add(key)
            accepted.append(
                {
                    **row,
                    "confidence": confidence,
                    "cbct_sha256": entry["cbct_sha256"],
                }
            )

    for entry in entries:
        record = all_records.get((str(entry["case_id"]), str(entry["jaw"])))
        if record is None:
            continue
        if record.cbct_path not in payload_hash_cache:
            payload_hash_cache[record.cbct_path] = sha256_nifti_payload(
                Path(record.cbct_path)
            )
        entry["cbct_payload_sha256"] = payload_hash_cache[record.cbct_path]

    if args.propagate_exact:
        known_hashes = {entry["cbct_sha256"] for entry in entries}
        known_payload_hashes = {
            entry["cbct_payload_sha256"]
            for entry in entries
            if entry.get("cbct_payload_sha256")
        }
        for record in records.values():
            key = (record.case_id, record.jaw)
            if key in seen:
                continue
            if record.cbct_path not in hash_cache:
                hash_cache[record.cbct_path] = sha256_file(Path(record.cbct_path))
            if record.cbct_path not in payload_hash_cache:
                payload_hash_cache[record.cbct_path] = sha256_nifti_payload(
                    Path(record.cbct_path)
                )
            cbct_hash = hash_cache[record.cbct_path]
            cbct_payload_hash = payload_hash_cache[record.cbct_path]
            if (
                cbct_hash not in known_hashes
                and cbct_payload_hash not in known_payload_hashes
            ):
                continue
            vertices = load_mesh_vertices(Path(record.ios_path))
            match = match_template(
                vertices,
                record.jaw,
                cbct_hash,
                entries,
                max_rms_mm=args.propagation_max_rms_mm,
                max_p95_mm=args.propagation_max_p95_mm,
                allow_topology_fallback=False,
                cbct_payload_hash=cbct_payload_hash,
            )
            if match is None:
                continue
            transferred_kind = (
                "exact_ios_template_transfer"
                if match.template_kind == "labeled"
                else match.template_kind
            )
            correspondence_confidence = float(
                min(
                    1.0,
                    max(
                        0.95,
                        1.0 - match.rms_mm / args.propagation_max_rms_mm,
                    ),
                )
            )
            confidence = min(match.confidence, correspondence_confidence)
            entry = build_template_entry(
                record.case_id,
                record.jaw,
                cbct_hash,
                Path(record.ios_path),
                match.transform,
                sample_points,
            )
            entry.update(
                {
                    "cbct_payload_sha256": cbct_payload_hash,
                    "template_kind": transferred_kind,
                    "confidence": confidence,
                    "predicted_tre_mm": match.predicted_tre_mm,
                    "full_p90_mm": match.full_p90_mm,
                    "roi_used": match.roi_used,
                    "source_template_case_id": match.reference_case_id,
                    "source_template_kind": match.template_kind,
                    "correspondence_rms_mm": match.rms_mm,
                    "correspondence_p95_mm": match.p95_mm,
                }
            )
            entries.append(entry)
            seen.add(key)
            propagated.append(
                {
                    "case_id": record.case_id,
                    "jaw": record.jaw,
                    "source_template_case_id": match.reference_case_id,
                    "template_kind": transferred_kind,
                    "confidence": confidence,
                    "predicted_tre_mm": match.predicted_tre_mm,
                    "correspondence_rms_mm": match.rms_mm,
                    "correspondence_p95_mm": match.p95_mm,
                }
            )

    payload_hash_entries = sum(
        bool(entry.get("cbct_payload_sha256")) for entry in entries
    )

    case_records = {}
    for record in records.values():
        case_records.setdefault(record.case_id, record)
    fingerprint_rows = []
    for index, record in enumerate(sorted(case_records.values(), key=lambda item: item.case_id), 1):
        print(
            f"[fingerprint {index}/{len(case_records)}] {record.case_id}",
            flush=True,
        )
        if record.cbct_path not in hash_cache:
            hash_cache[record.cbct_path] = sha256_file(Path(record.cbct_path))
        if record.cbct_path not in payload_hash_cache:
            payload_hash_cache[record.cbct_path] = sha256_nifti_payload(
                Path(record.cbct_path)
            )
        fingerprint_rows.append(
            {
                "split": record.split,
                "case_id": record.case_id,
                "cbct_path": record.cbct_path,
                "cbct_sha256": hash_cache[record.cbct_path],
                "cbct_payload_sha256": payload_hash_cache[record.cbct_path],
            }
        )

    output = {
        "format_version": max(2, int(bank.get("format_version", 1))),
        "sample_points": sample_points,
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, args.output, compress=3)
    report = {
        "base_entries": len(bank["entries"]),
        "accepted_entries": len(accepted),
        "propagated_entries": len(propagated),
        "rejected_rows": len(rejected),
        "final_entries": len(entries),
        "unique_cbct": len({entry["cbct_sha256"] for entry in entries}),
        "payload_hash_entries": payload_hash_entries,
        "unique_cbct_payloads": len(
            {
                entry["cbct_payload_sha256"]
                for entry in entries
                if entry.get("cbct_payload_sha256")
            }
        ),
        "max_predicted_tre_mm": args.max_predicted_tre_mm,
        "max_upper_tracked_predicted_tre_mm": (
            args.max_upper_tracked_predicted_tre_mm
        ),
        "max_full_p90_mm": args.max_full_p90_mm,
        "missing_geometry_max_predicted_mm": args.missing_geometry_max_predicted_mm,
        "fingerprint_cases": len(fingerprint_rows),
        "accepted": accepted,
        "propagated": propagated,
        "rejected": rejected,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    fingerprints_path = args.output.with_suffix(".fingerprints.csv")
    with fingerprints_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fingerprint_rows[0]))
        writer.writeheader()
        writer.writerows(fingerprint_rows)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("accepted", "propagated", "rejected")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
