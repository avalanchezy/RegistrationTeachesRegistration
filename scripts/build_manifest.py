from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.data import build_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("manifests/task2.csv"))
    args = parser.parse_args()
    records = build_manifest(args.data_root)
    write_manifest(records, args.output)
    counts = Counter((record.split, record.complete) for record in records)
    print(f"Wrote {len(records)} jaw records to {args.output.resolve()}")
    for key, count in sorted(counts.items()):
        print(f"{key[0]:16s} complete={key[1]}: {count}")


if __name__ == "__main__":
    main()
