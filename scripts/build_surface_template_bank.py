from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.data import load_manifest
from task2reg.template_transfer import (
    build_template_entry,
    sha256_file,
    sha256_nifti_payload,
)


GEOMETRY_TEACHER_METHODS = frozenset({"geometry", "geometry_crown_ensemble"})


def is_geometry_teacher_method(value: object) -> bool:
    return str(value) in GEOMETRY_TEACHER_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a conservative same-CBCT surface-transfer bank from labeled, "
            "exact-pseudo, and high-confidence geometry teachers."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-bank", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="Train-Unlabeled")
    parser.add_argument("--max-predicted-tre-mm", type=float, default=1.5)
    parser.add_argument(
        "--max-roi-predicted-tre-mm",
        type=float,
        help="Optional stricter predicted-TRE gate for ROI-derived teachers.",
    )
    parser.add_argument("--max-full-p90-mm", type=float, default=25.0)
    parser.add_argument(
        "--include-roi-teachers",
        action="store_true",
        help=(
            "Include ROI-retry teachers from both the base bank and new runs. "
            "Enable only after calibrating the selected predicted-TRE gate."
        ),
    )
    return parser.parse_args()


def eligible_base_entry(
    entry: dict[str, object],
    max_predicted_tre_mm: float,
    include_roi_teachers: bool,
    max_roi_predicted_tre_mm: float | None = None,
) -> bool:
    roi_used = bool(entry.get("roi_used", False))
    gate = (
        max_roi_predicted_tre_mm
        if roi_used and max_roi_predicted_tre_mm is not None
        else max_predicted_tre_mm
    )
    if float(entry.get("predicted_tre_mm", 0.0)) > gate:
        return False
    return include_roi_teachers or not roi_used


def main() -> None:
    args = parse_args()
    base = joblib.load(args.base_bank)
    base_entries = [
        dict(entry)
        for entry in base["entries"]
        if eligible_base_entry(
            entry,
            args.max_predicted_tre_mm,
            args.include_roi_teachers,
            args.max_roi_predicted_tre_mm,
        )
    ]
    records = {
        (record.case_id, record.jaw): record
        for record in load_manifest(args.manifest)
        if record.split == args.split and record.complete
    }
    sample_points = int(base.get("sample_points", 8192))
    seen = {(str(entry["case_id"]), str(entry["jaw"])) for entry in base_entries}
    hash_cache: dict[str, str] = {}
    payload_hash_cache: dict[str, str] = {}
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []

    for run_dir in args.runs:
        audit_path = run_dir / "audit.json"
        if not audit_path.is_file():
            raise FileNotFoundError(f"Missing audit file: {audit_path}")
        for row in json.loads(audit_path.read_text(encoding="utf-8")):
            key = (str(row["case_id"]), str(row["jaw"]))
            predicted = row.get("predicted_tre_mm")
            full_p90 = row.get("full_p90_mm")
            reason = ""
            if not is_geometry_teacher_method(row.get("method")):
                reason = f"method={row.get('method')}"
            elif key not in records:
                reason = "manifest_record_missing"
            elif key in seen:
                reason = "already_in_bank"
            elif predicted is None or float(predicted) > args.max_predicted_tre_mm:
                reason = "predicted_tre_gate"
            elif bool(row.get("roi_used", False)) and not args.include_roi_teachers:
                reason = "roi_teacher_rejected"
            elif (
                bool(row.get("roi_used", False))
                and args.max_roi_predicted_tre_mm is not None
                and float(predicted) > args.max_roi_predicted_tre_mm
            ):
                reason = "roi_predicted_tre_gate"
            elif full_p90 is None:
                reason = "full_p90_missing"
            elif float(full_p90) > args.max_full_p90_mm:
                reason = "full_p90_gate"
            transform_path = run_dir / key[0] / f"{key[1]}_gt.npy"
            if not reason and not transform_path.is_file():
                reason = "transform_missing"
            if reason:
                rejected.append({**row, "reason": reason, "run": str(run_dir)})
                continue

            record = records[key]
            cbct_path = str(record.cbct_path)
            if cbct_path not in hash_cache:
                hash_cache[cbct_path] = sha256_file(Path(cbct_path))
                payload_hash_cache[cbct_path] = sha256_nifti_payload(Path(cbct_path))
            transform = np.load(transform_path, allow_pickle=False).astype(np.float64)
            entry = build_template_entry(
                key[0],
                key[1],
                hash_cache[cbct_path],
                Path(record.ios_path),
                transform,
                sample_points,
            )
            entry.update(
                {
                    "cbct_payload_sha256": payload_hash_cache[cbct_path],
                    "template_kind": "high_confidence_geometry_teacher",
                    "confidence": float(
                        np.clip(1.0 - float(predicted) / 12.0, 0.0, 1.0)
                    ),
                    "predicted_tre_mm": float(predicted),
                    "full_p90_mm": float(full_p90),
                    "roi_used": bool(row.get("roi_used", False)),
                    "teacher_run": str(run_dir),
                }
            )
            base_entries.append(entry)
            seen.add(key)
            accepted.append(
                {
                    "case_id": key[0],
                    "jaw": key[1],
                    "predicted_tre_mm": float(predicted),
                    "full_p90_mm": float(full_p90),
                    "roi_used": bool(row.get("roi_used", False)),
                    "cbct_sha256": entry["cbct_sha256"],
                    "run": str(run_dir),
                }
            )

    output = {
        "format_version": 1,
        "sample_points": sample_points,
        "entries": base_entries,
        "surface_transfer": {
            "requires_same_cbct": True,
            "max_teacher_predicted_tre_mm": args.max_predicted_tre_mm,
            "max_roi_teacher_predicted_tre_mm": args.max_roi_predicted_tre_mm,
            "max_full_p90_mm": args.max_full_p90_mm,
            "roi_teachers_included": args.include_roi_teachers,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, args.output, compress=3)
    metadata = {
        "base_entries_before_quality_filter": len(base["entries"]),
        "base_entries_after_quality_filter": len(base_entries) - len(accepted),
        "accepted_geometry_teachers": len(accepted),
        "rejected_audit_rows": len(rejected),
        "final_entries": len(base_entries),
        "unique_cases": len({str(entry["case_id"]) for entry in base_entries}),
        "unique_cbct": len({str(entry["cbct_sha256"]) for entry in base_entries}),
        "max_predicted_tre_mm": args.max_predicted_tre_mm,
        "max_roi_predicted_tre_mm": args.max_roi_predicted_tre_mm,
        "max_full_p90_mm": args.max_full_p90_mm,
        "roi_teachers_included": args.include_roi_teachers,
        "accepted": accepted,
        "rejected": rejected,
        "size_bytes": args.output.stat().st_size,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in metadata.items() if key not in {"accepted", "rejected"}},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
