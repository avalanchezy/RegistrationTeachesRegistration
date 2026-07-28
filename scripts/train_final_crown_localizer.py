from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_crown_localizer import (
    CrownDataset,
    PseudoWeightedEpochSampler,
    run_epoch,
    set_seed,
)
from task2reg.crown_network import CrownLocalizerUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train final crown-localizer ensemble members on every labeled case."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--unlabeled-data-dir", type=Path)
    parser.add_argument("--pseudo-label-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--epochs-per-member", type=int, nargs="+")
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--background-weight", type=float, default=0.10)
    parser.add_argument("--surface-tolerance-voxels", type=int, default=0)
    parser.add_argument("--pseudo-weight", type=float, default=0.0)
    parser.add_argument("--max-pseudo-cases", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-data", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labeled_paths = sorted(args.data_dir.glob("*.npz"))
    if not labeled_paths:
        raise RuntimeError(f"No labeled crown-localizer arrays under {args.data_dir}")
    if args.epochs_per_member is not None and len(args.epochs_per_member) != len(
        args.seeds
    ):
        raise ValueError("--epochs-per-member must have one value per seed")
    member_epochs = args.epochs_per_member or [args.epochs] * len(args.seeds)
    if any(epochs < 1 for epochs in member_epochs):
        raise ValueError("Every final-training epoch count must be positive")

    semisupervised = args.pseudo_weight > 0.0
    if semisupervised:
        if args.unlabeled_data_dir is None or args.pseudo_label_root is None:
            raise ValueError(
                "Semi-supervised final training requires unlabeled data and pseudo labels"
            )
        if args.max_pseudo_cases < 1:
            raise ValueError("Semi-supervised final training requires pseudo cases")
        pseudo_folds: list[Path | None] = sorted(
            path for path in args.pseudo_label_root.glob("fold_*") if path.is_dir()
        )
        if not pseudo_folds:
            raise RuntimeError(
                f"No fold-specific pseudo labels under {args.pseudo_label_root}"
            )
    else:
        pseudo_folds = [None]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for member, (seed, epochs) in enumerate(zip(args.seeds, member_epochs)):
        member_dir = args.output_dir / f"fold_{member}"
        member_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = member_dir / "best.pt"
        if args.skip_existing and checkpoint.is_file():
            print(f"member={member}: existing all-labeled checkpoint retained", flush=True)
            existing = torch.load(checkpoint, map_location="cpu", weights_only=False)
            summary.append(
                {
                    "member": member,
                    "seed": int(existing.get("seed", seed)),
                    "epochs": int(existing.get("epoch", epochs)),
                    "labeled_cases": len(existing.get("train_cases", labeled_paths)),
                    "pseudo_cases": int(existing.get("pseudo_cases", 0)),
                    "pseudo_source_fold": existing.get("pseudo_source_fold"),
                    "pseudo_weight": float(existing.get("pseudo_weight", 0.0)),
                    "effective_samples_per_epoch": int(
                        existing.get("effective_samples_per_epoch", len(labeled_paths))
                    ),
                    "training_mode": existing.get(
                        "training_mode",
                        "semisupervised"
                        if float(existing.get("pseudo_weight", 0.0)) > 0.0
                        else "supervised",
                    ),
                    "retained_existing": True,
                }
            )
            continue
        set_seed(seed)
        pseudo_fold = pseudo_folds[member % len(pseudo_folds)]
        pseudo_labels = (
            sorted(pseudo_fold.glob("*.npz"))[: args.max_pseudo_cases]
            if pseudo_fold is not None
            else []
        )
        train_paths = list(labeled_paths)
        label_overrides: dict[Path, Path] = {}
        for pseudo_label in pseudo_labels:
            assert args.unlabeled_data_dir is not None
            image_path = args.unlabeled_data_dir / pseudo_label.name
            if not image_path.is_file():
                continue
            train_paths.append(image_path)
            label_overrides[image_path] = pseudo_label
        pseudo_count = len(label_overrides)
        if semisupervised and pseudo_count == 0:
            raise RuntimeError(f"No pseudo labels matched images for {pseudo_fold}")
        dataset = CrownDataset(
            train_paths,
            augment=True,
            cache=args.cache_data,
            seed=seed,
            label_overrides=label_overrides,
        )
        sampler = (
            PseudoWeightedEpochSampler(
                len(labeled_paths), pseudo_count, args.pseudo_weight, seed
            )
            if semisupervised
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=0,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(seed),
        )
        model = CrownLocalizerUNet(base_channels=args.base_channels).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=args.learning_rate * 0.05,
        )
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        history = []
        for epoch in range(1, epochs + 1):
            metrics = run_epoch(
                model,
                loader,
                device,
                args.background_weight,
                args.surface_tolerance_voxels,
                optimizer,
                scaler,
            )
            scheduler.step()
            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{key}": value for key, value in metrics.items()},
            }
            history.append(row)
            print(
                f"member={member} epoch={epoch:03d} train={metrics['loss']:.4f} "
                f"dice={(metrics['hard_dice_upper'] + metrics['hard_dice_lower']) * 0.5:.4f}",
                flush=True,
            )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "base_channels": args.base_channels,
                "fold": member,
                "seed": seed,
                "epoch": epochs,
                "validation_mean_hard_dice": None,
                "surface_tolerance_voxels": args.surface_tolerance_voxels,
                "train_cases": [path.stem for path in labeled_paths],
                "pseudo_cases": pseudo_count,
                "pseudo_source_fold": pseudo_fold.name if pseudo_fold else None,
                "pseudo_weight": args.pseudo_weight,
                "pseudo_weight_mode": (
                    "weighted_resampling" if semisupervised else "none"
                ),
                "effective_samples_per_epoch": len(loader),
                "validation_cases": [],
                "all_labeled_training": True,
                "training_mode": (
                    "semisupervised" if semisupervised else "supervised"
                ),
            },
            checkpoint,
        )
        with (member_dir / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        summary.append(
            {
                "member": member,
                "seed": seed,
                "epochs": epochs,
                "labeled_cases": len(labeled_paths),
                "pseudo_cases": pseudo_count,
                "pseudo_source_fold": pseudo_fold.name if pseudo_fold else None,
                "pseudo_weight": args.pseudo_weight,
                "effective_samples_per_epoch": len(loader),
                "training_mode": (
                    "semisupervised" if semisupervised else "supervised"
                ),
                "retained_existing": False,
            }
        )

    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Final all-labeled crown ensemble: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
