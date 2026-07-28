from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.crown_localizer import fixed_world_grid, resample_hu
from task2reg.data import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample challenge CBCT ROIs onto the crown-localizer inference grid."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--roi-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="Train-Unlabeled")
    parser.add_argument("--case-ids", nargs="*", default=[])
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--spacing-mm", type=float, default=1.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wanted = set(args.case_ids)
    cases: dict[str, Path] = {}
    for record in load_manifest(args.manifest):
        if record.split != args.split or not record.complete:
            continue
        if wanted and record.case_id not in wanted:
            continue
        cases.setdefault(record.case_id, Path(record.cbct_path))
    if not cases:
        raise ValueError(f"No complete cases selected from split {args.split}")

    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shape = (args.grid_size,) * 3
    rows = []
    failures = []
    for index, (case_id, cbct_path) in enumerate(sorted(cases.items())):
        output = data_dir / f"{case_id}.npz"
        print(f"[{index + 1}/{len(cases)}] {case_id}", flush=True)
        try:
            if output.exists() and not args.overwrite:
                with np.load(output, allow_pickle=False) as payload:
                    rows.append(
                        {
                            "case_id": case_id,
                            "status": "existing",
                            "shape": "x".join(map(str, payload["image"].shape)),
                        }
                    )
                continue
            roi_path = args.roi_dir / f"STS2_{case_id}_0000.nii.gz"
            if not roi_path.exists():
                raise FileNotFoundError(f"Missing automatic dental ROI: {roi_path}")
            roi_image = nib.load(str(roi_path))
            grid_affine = fixed_world_grid(roi_image, shape, args.spacing_mm)
            image_hu = resample_hu(roi_image, shape, grid_affine)
            np.savez_compressed(
                output,
                image=np.clip(np.rint(image_hu), -1024, 4095).astype(np.int16),
                affine=grid_affine,
            )
            rows.append(
                {
                    "case_id": case_id,
                    "status": "built",
                    "shape": "x".join(map(str, image_hu.shape)),
                }
            )
        except Exception as error:
            failures.append(
                {
                    "case_id": case_id,
                    "cbct_path": str(cbct_path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"  FAILED: {error}", flush=True)

    if not rows:
        raise RuntimeError("No crown-localizer inference grids were prepared")
    for filename, table in (("manifest.csv", rows), ("failures.csv", failures)):
        path = args.output_dir / filename
        if table:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
        elif path.exists():
            path.unlink()
    print(f"Prepared {len(rows)} cases; failures={len(failures)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
