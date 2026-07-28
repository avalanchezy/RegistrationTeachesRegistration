from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.data import load_manifest
from task2reg.dental_roi import crop_dental_roi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an affine-preserving dental ROI using only HU components."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="Train-Labeled")
    parser.add_argument("--dataset-name", default="ThresholdDentalROI")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=1600.0)
    parser.add_argument("--margin-mm", type=float, default=20.0)
    parser.add_argument("--max-crop-volume-cm3", type=float, default=1000.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-ids", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    unique: dict[str, Path] = {}
    for record in load_manifest(args.manifest):
        if record.split == args.split and record.complete:
            unique.setdefault(record.case_id, Path(record.cbct_path))
    cases = sorted(unique.items())
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [item for item in cases if item[0] in wanted]
    if args.limit:
        cases = cases[: args.limit]
    output_root = args.work_root / "inputs" / args.dataset_name
    rows = []
    failures = []
    for index, (case_id, source) in enumerate(cases, 1):
        output = output_root / "imagesTs" / f"STS2_{case_id}_0000.nii.gz"
        print(f"[{index}/{len(cases)}] {case_id}", flush=True)
        if output.exists() and not args.overwrite:
            image = nib.load(str(output))
            rows.append(
                {
                    "source": str(source),
                    "output": str(output),
                    "source_shape": "existing",
                    "crop_shape": "x".join(map(str, image.shape)),
                    "start_ijk": "",
                    "stop_ijk": "",
                    "threshold": args.threshold,
                    "margin_mm": args.margin_mm,
                    "crop_volume_cm3": float(
                        np.prod(np.asarray(image.shape) * image.header.get_zooms()[:3])
                        / 1000.0
                    ),
                    "volume_fallback": "existing",
                    "anterior_fallback": "existing",
                    "hard_cap": "existing",
                    "component_ids": "",
                    "component_scores": "",
                }
            )
            continue
        try:
            rows.append(
                crop_dental_roi(
                    source,
                    output,
                    args.threshold,
                    args.margin_mm,
                    args.max_crop_volume_cm3,
                )
            )
        except Exception as error:
            failures.append(
                {"case_id": case_id, "source": str(source), "error": str(error)}
            )
            print(f"  FAILED: {error}", flush=True)
    if not rows:
        raise RuntimeError("No threshold dental ROI crops were prepared")
    output_root.mkdir(parents=True, exist_ok=True)
    for filename, table in (("manifest.csv", rows), ("failures.csv", failures)):
        path = output_root / filename
        if table:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
        elif path.exists():
            path.unlink()
    print(f"Prepared {len(rows)} threshold-only crops under {output_root / 'imagesTs'}")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
