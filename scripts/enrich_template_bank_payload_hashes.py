from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.data import load_manifest
from task2reg.template_transfer import sha256_nifti_payload


def normalized_path(path: str | Path) -> str:
    return str(Path(path).resolve()).lower()


def load_hash_cache(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        normalized_path(cbct_path): str(metadata["sha256"])
        for cbct_path, metadata in payload.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add manifest-bound NIfTI payload hashes to a template bank."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--payload-hash-cache", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bank = joblib.load(args.input)
    records = {
        (record.case_id, record.jaw): record
        for record in load_manifest(args.manifest)
        if record.complete
    }
    hash_cache = load_hash_cache(args.payload_hash_cache)
    entries = []
    enriched = 0
    retained = 0
    missing = []
    for source in bank["entries"]:
        entry = dict(source)
        if entry.get("cbct_payload_sha256"):
            retained += 1
            entries.append(entry)
            continue
        key = (str(entry["case_id"]), str(entry["jaw"]))
        record = records.get(key)
        if record is None:
            missing.append({"case_id": key[0], "jaw": key[1], "reason": "manifest"})
            entries.append(entry)
            continue
        cache_key = normalized_path(record.cbct_path)
        payload_hash = hash_cache.get(cache_key)
        if payload_hash is None:
            payload_hash = sha256_nifti_payload(Path(record.cbct_path))
            hash_cache[cache_key] = payload_hash
        entry["cbct_payload_sha256"] = payload_hash
        entries.append(entry)
        enriched += 1

    output = dict(bank)
    output["format_version"] = max(2, int(bank.get("format_version", 1)))
    output["entries"] = entries
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, args.output, compress=3)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "entries": len(entries),
        "enriched_entries": enriched,
        "retained_payload_entries": retained,
        "missing_entries": len(missing),
        "payload_entries": sum(bool(entry.get("cbct_payload_sha256")) for entry in entries),
        "unique_payloads": len(
            {
                str(entry["cbct_payload_sha256"])
                for entry in entries
                if entry.get("cbct_payload_sha256")
            }
        ),
        "missing": missing,
        "size_bytes": args.output.stat().st_size,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "missing"}, indent=2))


if __name__ == "__main__":
    main()
