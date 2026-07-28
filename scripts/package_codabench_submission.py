#!/usr/bin/env python3
"""Validate and package an STSR Task 2 Codabench prediction directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np


JAW_FILES = ("lower_gt.npy", "upper_gt.npy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate 50 STSR Task 2 cases and create a root-flat Codabench ZIP."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--case-source-dir",
        type=Path,
        required=True,
        help="Directory whose immediate subdirectories define the expected case IDs.",
    )
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--expected-cases", type=int, default=50)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_transform(path: Path) -> dict[str, float | str | bool]:
    matrix = np.load(path, allow_pickle=False)
    if matrix.shape != (4, 4):
        raise ValueError(f"{path}: expected shape (4, 4), got {matrix.shape}")
    if matrix.dtype != np.float64:
        raise ValueError(f"{path}: expected float64, got {matrix.dtype}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: matrix contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{path}: invalid homogeneous bottom row {matrix[3].tolist()}")

    rotation = matrix[:3, :3]
    orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > 1e-5:
        raise ValueError(
            f"{path}: rotation orthogonality error {orthogonality_error:.6g} exceeds 1e-5"
        )
    if abs(abs(determinant) - 1.0) > 1e-5:
        raise ValueError(f"{path}: linear determinant magnitude {determinant:.6g} is not 1")

    return {
        "dtype": str(matrix.dtype),
        "determinant": determinant,
        "has_reflection": determinant < 0.0,
        "orthogonality_error": orthogonality_error,
        "translation_norm_mm": float(np.linalg.norm(matrix[:3, 3])),
    }


def expected_case_ids(case_source_dir: Path, expected_count: int) -> list[str]:
    case_ids = sorted(path.name for path in case_source_dir.iterdir() if path.is_dir())
    if len(case_ids) != expected_count:
        raise ValueError(
            f"{case_source_dir}: expected {expected_count} case directories, found {len(case_ids)}"
        )
    return case_ids


def validate_input_tree(input_dir: Path, case_ids: list[str]) -> dict[str, object]:
    actual_case_ids = sorted(path.name for path in input_dir.iterdir() if path.is_dir())
    if actual_case_ids != case_ids:
        missing = sorted(set(case_ids) - set(actual_case_ids))
        extra = sorted(set(actual_case_ids) - set(case_ids))
        raise ValueError(f"Case directory mismatch: missing={missing}, extra={extra}")

    cases: dict[str, object] = {}
    for case_id in case_ids:
        case_dir = input_dir / case_id
        actual_files = sorted(path.name for path in case_dir.iterdir() if path.is_file())
        if actual_files != list(JAW_FILES):
            raise ValueError(
                f"{case_dir}: expected only {list(JAW_FILES)}, found {actual_files}"
            )
        cases[case_id] = {
            jaw_file: validate_transform(case_dir / jaw_file) for jaw_file in JAW_FILES
        }
    return cases


def write_submission_zip(input_dir: Path, case_ids: list[str], output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary_zip = output_zip.with_suffix(output_zip.suffix + ".tmp")
    temporary_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(
        temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for case_id in case_ids:
            for jaw_file in JAW_FILES:
                archive.write(input_dir / case_id / jaw_file, f"{case_id}/{jaw_file}")
    temporary_zip.replace(output_zip)


def verify_zip(output_zip: Path, case_ids: list[str]) -> None:
    expected_members = [
        f"{case_id}/{jaw_file}" for case_id in case_ids for jaw_file in JAW_FILES
    ]
    with zipfile.ZipFile(output_zip, "r") as archive:
        if archive.namelist() != expected_members:
            raise ValueError("ZIP member order or root layout does not match the official contract")
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"ZIP CRC validation failed for {corrupt_member}")


def main() -> None:
    args = parse_args()
    case_ids = expected_case_ids(args.case_source_dir, args.expected_cases)
    cases = validate_input_tree(args.input_dir, case_ids)
    write_submission_zip(args.input_dir, case_ids, args.output_zip)
    verify_zip(args.output_zip, case_ids)

    report = {
        "status": "PASS",
        "input_dir": str(args.input_dir.resolve()),
        "output_zip": str(args.output_zip.resolve()),
        "case_count": len(case_ids),
        "matrix_count": len(case_ids) * len(JAW_FILES),
        "zip_member_count": len(case_ids) * len(JAW_FILES),
        "zip_size_bytes": args.output_zip.stat().st_size,
        "zip_sha256": sha256_file(args.output_zip),
        "case_ids": case_ids,
        "cases": cases,
    }
    report_path = args.report_json or args.output_zip.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "cases"}, indent=2))
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
