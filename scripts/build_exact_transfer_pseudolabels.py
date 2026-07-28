from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.data import load_manifest
from task2reg.template_transfer import load_mesh_vertices, match_template, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer labeled GT to unlabelled IOS copies sharing an STS CBCT scan."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="Train-Unlabeled")
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--max-rms-mm", type=float, default=0.02)
    parser.add_argument("--max-p95-mm", type=float, default=0.05)
    args = parser.parse_args()

    bank = joblib.load(args.template_bank)
    entries = bank["entries"]
    known_hashes = {entry["cbct_sha256"] for entry in entries}
    wanted = set(args.case_ids or ())
    records = [
        record
        for record in load_manifest(args.manifest)
        if record.split == args.split
        and record.complete
        and (not wanted or record.case_id in wanted)
    ]
    hash_cache: dict[str, str] = {}
    payloads = []
    audit = []
    for index, record in enumerate(records, 1):
        if record.cbct_path not in hash_cache:
            hash_cache[record.cbct_path] = sha256_file(Path(record.cbct_path))
        cbct_hash = hash_cache[record.cbct_path]
        if cbct_hash not in known_hashes:
            continue
        vertices = load_mesh_vertices(Path(record.ios_path))
        match = match_template(
            vertices,
            record.jaw,
            cbct_hash,
            entries,
            args.max_rms_mm,
            args.max_p95_mm,
            allow_topology_fallback=False,
        )
        if match is None:
            continue
        confidence = float(min(1.0, max(0.95, 1.0 - match.rms_mm / args.max_rms_mm)))
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
                "predicted_tre_mm": match.rms_mm,
                "teacher": "exact_ios_template_transfer",
                "source_labeled_case_id": match.reference_case_id,
                "cbct_sha256": cbct_hash,
                "correspondence_rms_mm": match.rms_mm,
                "correspondence_p95_mm": match.p95_mm,
                "correspondence_max_mm": match.max_mm,
            }
        )
        audit.append(
            {
                "case_id": record.case_id,
                "jaw": record.jaw,
                "source_labeled_case_id": match.reference_case_id,
                "confidence": confidence,
                "correspondence_rms_mm": match.rms_mm,
                "correspondence_p95_mm": match.p95_mm,
                "correspondence_max_mm": match.max_mm,
            }
        )
        print(
            f"[{index}/{len(records)}] {record.case_id} {record.jaw} <- "
            f"{match.reference_case_id}; rms={match.rms_mm:.6g} mm",
            flush=True,
        )

    if not payloads:
        raise RuntimeError("No exact repeated-CBCT IOS pseudo labels were found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pseudo_labels.json").write_text(
        json.dumps(payloads, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    summary = {
        "matched_jaws": len(payloads),
        "matched_cases": len({row["case_id"] for row in payloads}),
        "upper": sum(row["jaw"] == "upper" for row in payloads),
        "lower": sum(row["jaw"] == "lower" for row in payloads),
        "max_correspondence_rms_mm": max(row["correspondence_rms_mm"] for row in audit),
        "all_cbct_hash_matched": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
