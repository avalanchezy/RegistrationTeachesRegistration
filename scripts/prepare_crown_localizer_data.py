from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.crown_localizer import (
    build_crown_labels,
    fixed_world_grid,
    resample_hu,
    save_registration_labels,
    select_aligned_crown_points,
)
from task2reg.data import load_ios_points, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build STS-only pseudo crown masks from labeled IOS transforms."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--roi-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="Train-Labeled")
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--spacing-mm", type=float, default=1.25)
    parser.add_argument("--surface-radius-mm", type=float, default=2.5)
    parser.add_argument("--minimum-hu", type=float, default=150.0)
    parser.add_argument("--crown-fraction", type=float, default=0.35)
    parser.add_argument("--ios-points", type=int, default=120000)
    parser.add_argument("--visualize-cases", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def projection_overlay(image_hu: np.ndarray, labels: np.ndarray, title: str, output: Path) -> None:
    clipped = np.clip(image_hu, -500.0, 2500.0)
    normalized = (clipped + 500.0) / 3000.0
    colors = {1: np.array([0.96, 0.30, 0.20]), 2: np.array([0.12, 0.58, 0.92])}
    figure, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    figure.patch.set_facecolor("white")
    names = ("Sagittal MIP", "Coronal MIP", "Axial MIP")
    for axis, panel, name in zip(range(3), axes, names):
        base = np.max(normalized, axis=axis).T
        rgb = np.repeat(base[..., None], 3, axis=2)
        for label_id, color in colors.items():
            mask = np.max(labels == label_id, axis=axis).T
            rgb[mask] = 0.30 * rgb[mask] + 0.70 * color
        panel.imshow(np.clip(rgb, 0.0, 1.0), origin="lower")
        panel.set_facecolor("white")
        panel.set_title(name)
        panel.set_axis_off()
    figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    records = [
        record
        for record in load_manifest(args.manifest)
        if record.split == args.split and record.complete and record.transform_path
    ]
    if args.case_ids:
        selected = set(map(str, args.case_ids))
        records = [record for record in records if record.case_id in selected]
    grouped: dict[str, dict[str, object]] = {}
    for record in records:
        grouped.setdefault(record.case_id, {})[record.jaw] = record
    grouped = {
        case_id: jaws
        for case_id, jaws in grouped.items()
        if set(jaws) == {"upper", "lower"}
    }
    if not grouped:
        raise ValueError("No complete labeled upper/lower cases were selected")

    data_dir = args.output_dir / "data"
    label_dir = args.output_dir / "registration_labels"
    visualization_dir = args.output_dir / "visualizations"
    data_dir.mkdir(parents=True, exist_ok=True)
    requested_visualizations = set(map(str, args.visualize_cases))
    shape = (args.grid_size,) * 3
    rows: list[dict] = []

    for index, case_id in enumerate(sorted(grouped)):
        output = data_dir / f"{case_id}.npz"
        print(f"[{index + 1}/{len(grouped)}] {case_id}", flush=True)
        if output.exists() and not args.overwrite:
            with np.load(output, allow_pickle=False) as payload:
                labels = payload["label"]
                rows.append(
                    {
                        "case_id": case_id,
                        "status": "existing",
                        "upper_voxels": int(np.sum(labels == 1)),
                        "lower_voxels": int(np.sum(labels == 2)),
                    }
                )
            continue

        roi_path = args.roi_dir / f"STS2_{case_id}_0000.nii.gz"
        if not roi_path.exists():
            raise FileNotFoundError(f"Missing automatic dental ROI: {roi_path}")
        roi_image = nib.load(str(roi_path))
        grid_affine = fixed_world_grid(roi_image, shape, args.spacing_mm)
        image_hu = resample_hu(roi_image, shape, grid_affine)
        cbct_image = nib.load(str(grouped[case_id]["upper"].cbct_path))

        crown_points: dict[str, np.ndarray] = {}
        jaw_metadata: dict[str, dict] = {}
        for jaw in ("upper", "lower"):
            record = grouped[case_id][jaw]
            ios_points, _ = load_ios_points(
                Path(record.ios_path),
                args.ios_points,
                seed=20260715 + int(case_id) * 2 + int(jaw == "lower"),
            )
            transform = np.load(record.transform_path).astype(np.float64)
            crown_points[jaw], jaw_metadata[jaw] = select_aligned_crown_points(
                ios_points,
                transform,
                cbct_image,
                fraction=args.crown_fraction,
            )

        labels, coverage = build_crown_labels(
            image_hu,
            crown_points,
            grid_affine,
            spacing_mm=args.spacing_mm,
            radius_mm=args.surface_radius_mm,
            minimum_hu=args.minimum_hu,
        )
        metadata = {
            "case_id": case_id,
            "source": "STS26 Train-Labeled GT-aligned IOS only",
            "grid_size": args.grid_size,
            "spacing_mm": args.spacing_mm,
            "surface_radius_mm": args.surface_radius_mm,
            "minimum_hu": args.minimum_hu,
            "crown_fraction": args.crown_fraction,
            "grid_coverage": coverage,
            "jaws": jaw_metadata,
        }
        np.savez_compressed(
            output,
            image=np.clip(np.rint(image_hu), -1024, 4095).astype(np.int16),
            label=labels,
            affine=grid_affine,
        )
        (data_dir / f"{case_id}.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        save_registration_labels(
            labels,
            grid_affine,
            label_dir / f"STS2_{case_id}.nii.gz",
        )
        if case_id in requested_visualizations:
            projection_overlay(
                image_hu,
                labels,
                f"STS-only aligned IOS crown pseudo mask | case {case_id}",
                visualization_dir / f"{case_id}_pseudo_crown_overlay.png",
            )
        rows.append(
            {
                "case_id": case_id,
                "status": "built",
                "upper_variant": jaw_metadata["upper"]["selected_variant"],
                "lower_variant": jaw_metadata["lower"]["selected_variant"],
                "upper_grid_coverage": coverage["upper"],
                "lower_grid_coverage": coverage["lower"],
                "upper_voxels": int(np.sum(labels == 1)),
                "lower_voxels": int(np.sum(labels == 2)),
            }
        )

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} crown-localizer cases under {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
