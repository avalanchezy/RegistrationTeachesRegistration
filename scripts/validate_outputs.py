from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def discover_case_ids(input_dir: Path) -> list[str]:
    case_dirs = {
        path.parent.resolve()
        for pattern in ("CBCT.nii.gz", "CBCT.nii(1).gz")
        for path in input_dir.rglob(pattern)
    }
    if not case_dirs:
        raise FileNotFoundError(f"No CBCT input found under {input_dir}")
    return sorted(path.name for path in case_dirs)


def validate_matrix(path: Path) -> None:
    matrix = np.load(path, allow_pickle=False)
    if matrix.shape != (4, 4):
        raise ValueError(f"{path}: shape {matrix.shape}, expected (4, 4)")
    if matrix.dtype != np.float64:
        raise ValueError(f"{path}: dtype {matrix.dtype}, expected float64")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: contains NaN or infinity")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"{path}: invalid homogeneous bottom row")
    linear = matrix[:3, :3]
    if not np.allclose(linear.T @ linear, np.eye(3), atol=2e-2):
        raise ValueError(f"{path}: non-orthonormal linear block")
    determinant = float(np.linalg.det(linear))
    # The labeled export protocols include reflection-rigid transforms (det=-1).
    if not np.isclose(abs(determinant), 1.0, atol=2e-2):
        raise ValueError(f"{path}: determinant {determinant:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate STSR2026 Task 2 outputs")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    case_ids = discover_case_ids(args.input_dir)
    expected = set(case_ids)
    output_entries = list(args.output_dir.iterdir())
    actual = {path.name for path in output_entries}
    if actual != expected:
        raise ValueError(
            f"Output case directories differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    non_directories = sorted(path.name for path in output_entries if not path.is_dir())
    if non_directories:
        raise ValueError(f"Output root contains non-case files: {non_directories}")
    for case_id in case_ids:
        case_dir = args.output_dir / case_id
        expected_files = {"upper_gt.npy", "lower_gt.npy"}
        actual_files = {path.name for path in case_dir.iterdir()}
        if actual_files != expected_files:
            raise ValueError(
                f"{case_dir}: expected {sorted(expected_files)}, got {sorted(actual_files)}"
            )
        for name in sorted(expected_files):
            validate_matrix(case_dir / name)
    print(f"Validated {len(case_ids)} case(s), {2 * len(case_ids)} rigid matrices")


if __name__ == "__main__":
    main()
