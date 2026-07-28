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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a compact labeled IOS template bank for repeated STS CBCT scans."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-points", type=int, default=8192)
    parser.add_argument("--split", default="Train-Labeled")
    parser.add_argument("--pseudo-labels", type=Path)
    parser.add_argument("--min-pseudo-confidence", type=float, default=0.7)
    args = parser.parse_args()

    records = [
        record
        for record in load_manifest(args.manifest)
        if record.split == args.split and record.complete and record.transform_path
    ]
    hash_cache: dict[str, str] = {}
    payload_hash_cache: dict[str, str] = {}
    entries = []
    for index, record in enumerate(records, 1):
        print(f"[{index}/{len(records)}] {record.case_id} {record.jaw}", flush=True)
        if record.cbct_path not in hash_cache:
            hash_cache[record.cbct_path] = sha256_file(Path(record.cbct_path))
            payload_hash_cache[record.cbct_path] = sha256_nifti_payload(
                Path(record.cbct_path)
            )
        entry = build_template_entry(
            record.case_id,
            record.jaw,
            hash_cache[record.cbct_path],
            Path(record.ios_path),
            np.load(record.transform_path),
            args.sample_points,
        )
        entry.update(
            {
                "cbct_payload_sha256": payload_hash_cache[record.cbct_path],
                "template_kind": "labeled",
                "confidence": 1.0,
                "predicted_tre_mm": 0.0,
            }
        )
        entries.append(entry)

    pseudo_count = 0
    if args.pseudo_labels:
        pseudo_labels = json.loads(args.pseudo_labels.read_text(encoding="utf-8"))
        accepted = [
            payload
            for payload in pseudo_labels
            if float(payload.get("confidence", 1.0)) >= args.min_pseudo_confidence
        ]
        for index, payload in enumerate(accepted, 1):
            print(
                f"[pseudo {index}/{len(accepted)}] {payload['case_id']} {payload['jaw']}",
                flush=True,
            )
            cbct_path = str(payload["cbct_path"])
            if cbct_path not in hash_cache:
                hash_cache[cbct_path] = sha256_file(Path(cbct_path))
                payload_hash_cache[cbct_path] = sha256_nifti_payload(Path(cbct_path))
            entry = build_template_entry(
                str(payload["case_id"]),
                str(payload["jaw"]),
                hash_cache[cbct_path],
                Path(payload["ios_path"]),
                np.asarray(payload["transform"], dtype=np.float64),
                args.sample_points,
            )
            entry.update(
                {
                    "cbct_payload_sha256": payload_hash_cache[cbct_path],
                    "template_kind": str(
                        payload.get("teacher", "learned_threshold_teacher")
                    ),
                    "confidence": float(payload.get("confidence", 1.0)),
                    "predicted_tre_mm": float(payload.get("predicted_tre_mm", 0.0)),
                    "source_labeled_case_id": str(
                        payload.get("source_labeled_case_id", "")
                    ),
                }
            )
            entries.append(entry)
            pseudo_count += 1

    payload = {
        "format_version": 2,
        "sample_points": args.sample_points,
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.output, compress=3)
    metadata = {
        "format_version": 2,
        "entries": len(entries),
        "labeled_entries": len(records),
        "pseudo_entries": pseudo_count,
        "cases": len({entry["case_id"] for entry in entries}),
        "unique_cbct": len({entry["cbct_sha256"] for entry in entries}),
        "unique_cbct_payloads": len(
            {entry["cbct_payload_sha256"] for entry in entries}
        ),
        "sample_points": args.sample_points,
        "size_bytes": args.output.stat().st_size,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
