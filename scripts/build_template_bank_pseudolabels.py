from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.data import load_manifest
from task2reg.template_transfer import (
    load_mesh_vertices,
    match_template,
    sha256_file,
    sha256_nifti_payload,
)


def training_teacher_kind(template_kind: str) -> str:
    if template_kind in {"labeled", "exact_ios_template_transfer"}:
        return "exact_ios_template_transfer"
    if template_kind == "geometry_self_teacher":
        return "geometry_self_teacher"
    return "learned_threshold_teacher"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export confidence- and error-aware pseudo labels from a mixed "
            "same-CBCT template bank."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fingerprint-cache",
        type=Path,
        help="Reuse a prior template-bank fingerprint CSV for unchanged CBCT files.",
    )
    parser.add_argument("--split", default="Train-Unlabeled")
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--max-predicted-tre-mm", type=float, default=3.5)
    parser.add_argument("--max-rms-mm", type=float, default=0.02)
    parser.add_argument("--max-p95-mm", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bank = joblib.load(args.template_bank)
    entries = bank["entries"]
    wanted = {str(case_id).zfill(3) for case_id in (args.case_ids or ())}
    records = [
        record
        for record in load_manifest(args.manifest)
        if record.split == args.split
        and record.complete
        and (not wanted or record.case_id in wanted)
    ]
    raw_hash_cache: dict[str, str] = {}
    payload_hash_cache: dict[str, str] = {}
    if args.fingerprint_cache is not None:
        with args.fingerprint_cache.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                cbct_path = str(row["cbct_path"])
                raw_hash_cache[cbct_path] = str(row["cbct_sha256"])
                payload_hash_cache[cbct_path] = str(row["cbct_payload_sha256"])
    payloads: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []

    for index, record in enumerate(records, 1):
        cbct_path = str(record.cbct_path)
        if cbct_path not in raw_hash_cache:
            raw_hash_cache[cbct_path] = sha256_file(Path(cbct_path))
            payload_hash_cache[cbct_path] = sha256_nifti_payload(Path(cbct_path))
        try:
            vertices = load_mesh_vertices(Path(record.ios_path))
        except (ValueError, RuntimeError) as error:
            rejected.append(
                {
                    "case_id": record.case_id,
                    "jaw": record.jaw,
                    "reason": f"invalid_ios: {error}",
                }
            )
            print(
                f"[{index}/{len(records)}] {record.case_id} {record.jaw} skipped: {error}",
                flush=True,
            )
            continue
        match = match_template(
            vertices,
            record.jaw,
            raw_hash_cache[cbct_path],
            entries,
            max_rms_mm=args.max_rms_mm,
            max_p95_mm=args.max_p95_mm,
            allow_topology_fallback=False,
            cbct_payload_hash=payload_hash_cache[cbct_path],
        )
        if match is None or match.predicted_tre_mm > args.max_predicted_tre_mm:
            continue

        correspondence_confidence = max(
            0.95,
            1.0 - match.rms_mm / max(args.max_rms_mm, 1e-8),
        )
        confidence = float(min(match.confidence, correspondence_confidence))
        teacher = training_teacher_kind(match.template_kind)
        payloads.append(
            {
                "case_id": record.case_id,
                "jaw": record.jaw,
                "accepted": 1,
                "confidence": confidence,
                "consensus_count": 1,
                "ios_path": record.ios_path,
                "cbct_path": record.cbct_path,
                "transform": match.transform.tolist(),
                "predicted_tre_mm": float(match.predicted_tre_mm),
                "full_p90_mm": match.full_p90_mm,
                "roi_used": match.roi_used,
                "teacher": teacher,
                "source_labeled_case_id": (
                    match.reference_case_id
                    if teacher == "exact_ios_template_transfer"
                    else ""
                ),
                "source_teacher_case_id": match.reference_case_id,
                "source_template_kind": match.template_kind,
                "cbct_match_kind": match.cbct_match_kind,
                "correspondence_rms_mm": match.rms_mm,
                "correspondence_p95_mm": match.p95_mm,
                "correspondence_max_mm": match.max_mm,
            }
        )
        audit.append(
            {
                "case_id": record.case_id,
                "jaw": record.jaw,
                "teacher": teacher,
                "source_teacher_case_id": match.reference_case_id,
                "predicted_tre_mm": match.predicted_tre_mm,
                "confidence": confidence,
                "cbct_match_kind": match.cbct_match_kind,
                "correspondence_rms_mm": match.rms_mm,
                "correspondence_p95_mm": match.p95_mm,
            }
        )
        print(
            f"[{index}/{len(records)}] {record.case_id} {record.jaw} <- "
            f"{match.reference_case_id} ({teacher}); "
            f"pred={match.predicted_tre_mm:.3f} mm rms={match.rms_mm:.6g} mm",
            flush=True,
        )

    if not payloads:
        raise RuntimeError("No eligible mixed-bank pseudo labels were found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pseudo_labels.json").write_text(
        json.dumps(payloads, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    summary = {
        "matched_jaws": len(payloads),
        "matched_cases": len({str(row["case_id"]) for row in payloads}),
        "teacher_counts": {
            teacher: sum(row["teacher"] == teacher for row in payloads)
            for teacher in sorted({str(row["teacher"]) for row in payloads})
        },
        "max_predicted_tre_mm": args.max_predicted_tre_mm,
        "rejected_invalid_ios": len(rejected),
        "max_correspondence_rms_mm": max(
            float(row["correspondence_rms_mm"]) for row in payloads
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if rejected:
        with (args.output_dir / "rejected.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rejected[0]))
            writer.writeheader()
            writer.writerows(rejected)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
