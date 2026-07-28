from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.crown_inference import CrownPostprocessConfig
from task2reg.crown_localizer import save_registration_labels
from task2reg.crown_network import probabilities_to_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize crown registration masks from saved probability volumes."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--probability-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--postprocess-config", type=Path)
    parser.add_argument("--minimum-probability", type=float)
    parser.add_argument("--minimum-component-voxels", type=int)
    parser.add_argument("--maximum-components", type=int)
    parser.add_argument("--minimum-hu", type=float)
    parser.add_argument("--save-label-arrays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.postprocess_config is not None:
        config = CrownPostprocessConfig.load(args.postprocess_config)
    else:
        required = (
            args.minimum_probability,
            args.minimum_component_voxels,
            args.maximum_components,
            args.minimum_hu,
        )
        if any(value is None for value in required):
            raise ValueError(
                "Provide --postprocess-config or all four scalar postprocess arguments"
            )
        config = CrownPostprocessConfig(
            minimum_probability=args.minimum_probability,
            minimum_component_voxels=args.minimum_component_voxels,
            maximum_components=args.maximum_components,
            minimum_hu=args.minimum_hu,
        )
    upper_config = config.resolved_jaw("upper")
    lower_config = config.resolved_jaw("lower")
    registration_dir = args.output_dir / "registration_labels"
    label_array_dir = args.output_dir / "label_arrays"
    rows = []
    for probability_path in sorted(args.probability_dir.glob("*.npz")):
        data_path = args.data_dir / probability_path.name
        if not data_path.exists():
            continue
        with np.load(probability_path, allow_pickle=False) as payload:
            probabilities = payload["probabilities"].astype(np.float32)
        with np.load(data_path, allow_pickle=False) as payload:
            image = payload["image"].astype(np.int16)
            affine = payload["affine"].astype(np.float64)
        labels = probabilities_to_labels(
            probabilities,
            minimum_probability=(
                float(upper_config.minimum_probability),
                float(lower_config.minimum_probability),
            ),
            minimum_component_voxels=(
                int(upper_config.minimum_component_voxels),
                int(lower_config.minimum_component_voxels),
            ),
            maximum_components=(
                int(upper_config.maximum_components),
                int(lower_config.maximum_components),
            ),
            image_hu=image,
            minimum_hu=(float(upper_config.minimum_hu), float(lower_config.minimum_hu)),
        )
        save_registration_labels(
            labels,
            affine,
            registration_dir / f"STS2_{probability_path.stem}.nii.gz",
        )
        if args.save_label_arrays:
            label_array_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                label_array_dir / probability_path.name,
                label=labels.astype(np.uint8),
                affine=affine,
            )
        rows.append(
            {
                "case_id": probability_path.stem,
                "upper_voxels": int(np.sum(labels == 1)),
                "lower_voxels": int(np.sum(labels == 2)),
            }
        )
    if not rows:
        raise RuntimeError("No matching probability and input volumes were found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "cases": len(rows),
        "postprocess": config.to_dict(),
        "mean_upper_voxels": float(np.mean([row["upper_voxels"] for row in rows])),
        "mean_lower_voxels": float(np.mean([row["lower_voxels"] for row in rows])),
        "records": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
