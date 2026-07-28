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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.crown_localizer import save_registration_labels
from task2reg.crown_network import (
    CrownLocalizerUNet,
    apply_inplane_transform,
    inplane_tta_transforms,
    invert_inplane_transform,
    normalize_hu,
    probabilities_to_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouped-OOF crown-localizer prediction.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-probability", type=float, default=0.70)
    parser.add_argument("--minimum-component-voxels", type=int, default=12)
    parser.add_argument("--maximum-components", type=int, default=1)
    parser.add_argument("--minimum-hu", type=float, default=150.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tta-mode", choices=("none", "d4"), default="none")
    parser.add_argument("--fold-indices", nargs="*", type=int)
    parser.add_argument(
        "--ensemble-all-folds",
        action="store_true",
        help="Average every selected fold model for cases without OOF labels.",
    )
    parser.add_argument("--visualize-cases", nargs="*", default=[])
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--save-label-arrays", action="store_true")
    return parser.parse_args()


def overlay(
    image: np.ndarray,
    truth: np.ndarray | None,
    prediction: np.ndarray,
    output: Path,
    title: str,
) -> None:
    image = (np.clip(image, -500.0, 2500.0) + 500.0) / 3000.0
    panels = (
        ((truth, "Pseudo GT"), (prediction, "OOF prediction"))
        if truth is not None
        else ((prediction, "Fold-ensemble prediction"),)
    )
    figure, axes = plt.subplots(
        len(panels), 3, figsize=(14, 4.5 * len(panels)), constrained_layout=True
    )
    axes = np.asarray(axes).reshape(len(panels), 3)
    figure.patch.set_facecolor("white")
    colors = {1: np.array([0.96, 0.30, 0.20]), 2: np.array([0.12, 0.58, 0.92])}
    for row, (labels, name) in enumerate(panels):
        for axis in range(3):
            base = np.max(image, axis=axis).T
            rgb = np.repeat(base[..., None], 3, axis=2)
            for class_id, color in colors.items():
                mask = np.max(labels == class_id, axis=axis).T
                rgb[mask] = 0.30 * rgb[mask] + 0.70 * color
            axes[row, axis].imshow(np.clip(rgb, 0.0, 1.0), origin="lower")
            axes[row, axis].set_facecolor("white")
            axes[row, axis].set_title(f"{name} | axis {axis}")
            axes[row, axis].set_axis_off()
    figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, facecolor="white")
    plt.close(figure)


def dice(prediction: np.ndarray, truth: np.ndarray, class_id: int) -> float:
    predicted = prediction == class_id
    target = truth == class_id
    return float((2 * np.sum(predicted & target) + 1) / (np.sum(predicted) + np.sum(target) + 1))


def main() -> None:
    args = parse_args()
    splits = json.loads((args.model_dir / "splits.json").read_text(encoding="utf-8"))
    case_to_fold = {
        case_id: int(fold)
        for fold, payload in splits.items()
        for case_id in payload["validation"]
    }
    selected_folds = set(args.fold_indices) if args.fold_indices else None
    available_folds = sorted(int(path.name.split("_", 1)[1]) for path in args.model_dir.glob("fold_*") if (path / "best.pt").exists())
    if selected_folds is not None:
        available_folds = [fold for fold in available_folds if fold in selected_folds]
    if not available_folds:
        raise RuntimeError(f"No selected fold checkpoints found under {args.model_dir}")
    requested = set(map(str, args.visualize_cases))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models: dict[int, CrownLocalizerUNet] = {}
    rows = []
    label_dir = args.output_dir / "registration_labels"
    probability_dir = args.output_dir / "probabilities"
    label_array_dir = args.output_dir / "label_arrays"
    shape_report = None

    def get_model(fold: int) -> CrownLocalizerUNet:
        if fold not in models:
            checkpoint = torch.load(
                args.model_dir / f"fold_{fold}" / "best.pt",
                map_location=device,
                weights_only=False,
            )
            model = CrownLocalizerUNet(base_channels=int(checkpoint["base_channels"]))
            model.load_state_dict(checkpoint["state_dict"])
            models[fold] = model.to(device).eval()
        return models[fold]

    for path in sorted(args.data_dir.glob("*.npz")):
        case_id = path.stem
        if args.ensemble_all_folds:
            inference_folds = available_folds
            fold_name = "ensemble_" + "_".join(map(str, inference_folds))
        else:
            if case_id not in case_to_fold:
                continue
            fold = case_to_fold[case_id]
            if selected_folds is not None and fold not in selected_folds:
                continue
            inference_folds = [fold]
            fold_name = str(fold)
        with np.load(path, allow_pickle=False) as payload:
            image = payload["image"].astype(np.float32)
            truth = payload["label"].astype(np.uint8) if "label" in payload.files else None
            affine = payload["affine"].astype(np.float64)
        inputs = torch.from_numpy(normalize_hu(image)[None, None]).to(device)
        fold_probabilities = []
        for fold in inference_folds:
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                view_probabilities = []
                for view_index, (rotation, flip) in enumerate(
                    inplane_tta_transforms(args.tta_mode)
                ):
                    transformed = apply_inplane_transform(inputs, rotation, flip)
                    if view_index == 0:
                        logits, shapes = get_model(fold)(transformed, return_shapes=True)
                    else:
                        logits = get_model(fold)(transformed)
                    probabilities = torch.softmax(logits, dim=1)
                    view_probabilities.append(
                        invert_inplane_transform(probabilities, rotation, flip)
                    )
                fold_probabilities.append(
                    torch.stack(view_probabilities).mean(dim=0)[0].float().cpu().numpy()
                )
        probability_stack = np.stack(fold_probabilities)
        probabilities = np.mean(probability_stack, axis=0)
        foreground = np.max(probabilities[1:], axis=0) >= 0.20
        if len(inference_folds) > 1 and np.any(foreground):
            disagreement = float(
                np.mean(np.std(probability_stack[:, 1:], axis=0)[:, foreground])
            )
            fold_labels = np.argmax(probability_stack, axis=1)
            votes = np.stack(
                [np.sum(fold_labels == class_id, axis=0) for class_id in range(3)]
            )
            consensus = float(np.mean(np.max(votes, axis=0)[foreground] / len(inference_folds)))
        else:
            disagreement = 0.0
            consensus = 1.0
        if shape_report is None:
            shape_report = shapes
        prediction = probabilities_to_labels(
            probabilities,
            minimum_probability=args.minimum_probability,
            minimum_component_voxels=args.minimum_component_voxels,
            maximum_components=args.maximum_components,
            image_hu=image,
            minimum_hu=args.minimum_hu,
        )
        predicted_foreground = prediction > 0
        if np.any(predicted_foreground):
            maximum_probability = np.max(probabilities, axis=0)
            foreground_confidence = float(
                np.mean(maximum_probability[predicted_foreground])
            )
            entropy_map = -np.sum(
                probabilities * np.log(np.clip(probabilities, 1e-7, 1.0)), axis=0
            )
            foreground_entropy = float(np.mean(entropy_map[predicted_foreground]))
        else:
            foreground_confidence = 0.0
            foreground_entropy = float(np.log(3.0))
        save_registration_labels(
            prediction, affine, label_dir / f"STS2_{case_id}.nii.gz"
        )
        if args.save_probabilities:
            probability_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                probability_dir / f"{case_id}.npz",
                probabilities=probabilities.astype(np.float16),
                affine=affine,
            )
        if args.save_label_arrays:
            label_array_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                label_array_dir / f"{case_id}.npz",
                label=prediction.astype(np.uint8),
                affine=affine,
            )
        if case_id in requested:
            overlay(
                image,
                truth,
                prediction,
                args.output_dir / "visualizations" / f"{case_id}_oof_overlay.png",
                f"Challenge-only crown localizer | case {case_id} | {fold_name}",
            )
        row = {
            "case_id": case_id,
            "fold": fold_name,
            "ensemble_models": len(inference_folds),
            "dice_upper": dice(prediction, truth, 1) if truth is not None else "",
            "dice_lower": dice(prediction, truth, 2) if truth is not None else "",
            "predicted_upper_voxels": int(np.sum(prediction == 1)),
            "predicted_lower_voxels": int(np.sum(prediction == 2)),
            "ensemble_disagreement": disagreement,
            "ensemble_consensus": consensus,
            "foreground_confidence": foreground_confidence,
            "foreground_entropy": foreground_entropy,
        }
        rows.append(row)
        if truth is not None:
            print(
                f"{case_id}: upper={row['dice_upper']:.4f} lower={row['dice_lower']:.4f}",
                flush=True,
            )
        else:
            print(
                f"{case_id}: upper={row['predicted_upper_voxels']} "
                f"lower={row['predicted_lower_voxels']} disagreement={disagreement:.4f}",
                flush=True,
            )
    if not rows:
        raise RuntimeError("No OOF cases were predicted")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labeled_rows = [row for row in rows if row["dice_upper"] != ""]
    summary = {
        "cases": len(rows),
        "tta_mode": args.tta_mode,
        "labeled_cases": len(labeled_rows),
        "mean_dice_upper": (
            float(np.mean([row["dice_upper"] for row in labeled_rows]))
            if labeled_rows
            else None
        ),
        "mean_dice_lower": (
            float(np.mean([row["dice_lower"] for row in labeled_rows]))
            if labeled_rows
            else None
        ),
        "mean_ensemble_disagreement": float(
            np.mean([row["ensemble_disagreement"] for row in rows])
        ),
        "mean_ensemble_consensus": float(
            np.mean([row["ensemble_consensus"] for row in rows])
        ),
        "mean_foreground_confidence": float(
            np.mean([row["foreground_confidence"] for row in rows])
        ),
        "mean_foreground_entropy": float(
            np.mean([row["foreground_entropy"] for row in rows])
        ),
        "ensemble_all_folds": args.ensemble_all_folds,
        "minimum_probability": args.minimum_probability,
        "minimum_component_voxels": args.minimum_component_voxels,
        "maximum_components": args.maximum_components,
        "minimum_hu": args.minimum_hu,
        "tensor_shapes": shape_report,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
