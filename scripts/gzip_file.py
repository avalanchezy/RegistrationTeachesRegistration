from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gzip a Docker image tar stream")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    with args.source.open("rb") as source, gzip.open(
        args.destination, "wb", compresslevel=6
    ) as destination:
        shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    print(args.destination)


if __name__ == "__main__":
    main()
