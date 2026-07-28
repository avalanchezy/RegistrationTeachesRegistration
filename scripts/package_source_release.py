from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an organizer-ready source archive from Git-tracked files."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prefix", default="RegistrationTeachesRegistration", help="Archive root folder"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_paths() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    names = [name for name in completed.stdout.decode("utf-8").split("\0") if name]
    paths = [ROOT / name for name in names]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Tracked paths are missing: {missing}")
    return paths


def main() -> None:
    args = parse_args()
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_source_release.py")],
        cwd=ROOT,
        check=True,
    )
    paths = tracked_paths()
    if not paths:
        raise RuntimeError("No Git-tracked source files found")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, f"{args.prefix}/{relative}")
    print(
        f"Wrote {output} ({output.stat().st_size} bytes, {len(paths)} files)\n"
        f"SHA256 {sha256(output)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
