from __future__ import annotations

import argparse
import json
from pathlib import Path

from task2reg.data import load_manifest
from task2reg.template_transfer import sha256_nifti_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the CBCT-content hash cache used for grouped folds."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="*", default=())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_splits = set(args.splits)
    cache: dict[str, dict[str, int | str]] = {}
    seen: set[Path] = set()
    for record in load_manifest(args.manifest):
        if selected_splits and record.split not in selected_splits:
            continue
        path = Path(record.cbct_path).resolve()
        if path in seen:
            continue
        seen.add(path)
        stat = path.stat()
        cache[str(path)] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": sha256_nifti_payload(path),
        }
        print(f"[{len(cache)}] {record.split}/{record.case_id}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cache)} CBCT payload hashes to {args.output}", flush=True)


if __name__ == "__main__":
    main()
