"""Merge per-source journal manifests and check global patient isolation."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2reg.journal.data import load_journal_manifest, validate_protocol
from task2reg.journal.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for manifest in args.inputs for row in load_journal_manifest(manifest)]
    validate_protocol(rows)
    for row in rows:
        row["npz_path"] = os.path.relpath(row["npz_path"], args.output.resolve().parent)
    write_json(args.output, {"schema_version": 1, "records": rows})
    print(f"Validated and merged {len(rows)} jaw records into {args.output}")


if __name__ == "__main__":
    main()
