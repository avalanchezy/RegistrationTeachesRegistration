from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.crown_network import (
    CrownLocalizerUNet,
    crown_localizer_loss,
    hard_dice,
    normalize_hu,
)
from task2reg.data import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a challenge-data-only 3D crown localizer.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cbct-hash-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-indices", nargs="*", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--background-weight", type=float, default=0.05)
    parser.add_argument("--surface-tolerance-voxels", type=int, default=0)
    parser.add_argument("--unlabeled-data-dir", type=Path)
    parser.add_argument("--pseudo-label-root", type=Path)
    parser.add_argument("--pseudo-weight", type=float, default=0.10)
    parser.add_argument(
        "--pseudo-weight-mode",
        choices=("weighted_resampling", "loss_scale"),
        default="weighted_resampling",
    )
    parser.add_argument("--max-pseudo-cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--cache-data", action="store_true")
    return parser.parse_args()


class CrownDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        augment: bool,
        cache: bool,
        seed: int,
        label_overrides: dict[Path, Path] | None = None,
        sample_weights: dict[Path, float] | None = None,
    ) -> None:
        self.paths = paths
        self.augment = augment
        self.seed = seed
        self.label_overrides = label_overrides or {}
        self.sample_weights = sample_weights or {}
        self.cache = {}
        if cache:
            for path in paths:
                self.cache[path] = self._load(path)

    def _load(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=False) as payload:
            image = payload["image"].copy()
            if path not in self.label_overrides:
                return image, payload["label"].copy()
        with np.load(self.label_overrides[path], allow_pickle=False) as pseudo_payload:
            return image, pseudo_payload["label"].copy()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, float]:
        path = self.paths[index]
        image, labels = self.cache.get(path, (None, None))
        if image is None:
            image, labels = self._load(path)
        else:
            image, labels = image.copy(), labels.copy()
        if self.augment:
            for axis in (0, 1):
                if random.random() < 0.5:
                    image = np.flip(image, axis=axis)
                    labels = np.flip(labels, axis=axis)
            turns = random.randrange(4)
            if turns:
                image = np.rot90(image, turns, axes=(0, 1))
                labels = np.rot90(labels, turns, axes=(0, 1))
        image = normalize_hu(image.astype(np.float32))
        if self.augment:
            image = image * random.uniform(0.90, 1.10) + random.uniform(-0.10, 0.10)
            image += np.random.normal(0.0, 0.025, size=image.shape).astype(np.float32)
        image = np.ascontiguousarray(image[None])
        labels = np.ascontiguousarray(labels.astype(np.int64))
        return (
            torch.from_numpy(image),
            torch.from_numpy(labels),
            path.stem,
            float(self.sample_weights.get(path, 1.0)),
        )


class PseudoWeightedEpochSampler(Sampler[int]):
    """Visit every labeled case and a weighted subset of pseudo cases per epoch."""

    def __init__(
        self,
        labeled_count: int,
        pseudo_count: int,
        pseudo_weight: float,
        seed: int,
    ) -> None:
        if labeled_count <= 0 or pseudo_count <= 0:
            raise ValueError("Both labeled and pseudo counts must be positive")
        if not 0.0 < pseudo_weight <= 1.0:
            raise ValueError("pseudo_weight must be in (0, 1]")
        self.labeled_count = labeled_count
        self.pseudo_count = pseudo_count
        self.pseudo_samples = max(1, int(round(pseudo_count * pseudo_weight)))
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.labeled_count + self.pseudo_samples

    def __iter__(self):
        labeled = torch.arange(self.labeled_count)
        pseudo_offset = self.labeled_count
        if self.pseudo_samples <= self.pseudo_count:
            pseudo = torch.randperm(self.pseudo_count, generator=self.generator)[
                : self.pseudo_samples
            ]
        else:
            pseudo = torch.randint(
                self.pseudo_count,
                (self.pseudo_samples,),
                generator=self.generator,
            )
        indices = torch.cat((labeled, pseudo + pseudo_offset))
        order = torch.randperm(len(indices), generator=self.generator)
        return iter(indices[order].tolist())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_case_metadata(
    paths: list[Path], manifest_path: Path, cache_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    records = {
        record.case_id: record
        for record in load_manifest(manifest_path)
        if record.split == "Train-Labeled" and record.jaw == "upper"
    }
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    groups = []
    strata = []
    for path in paths:
        record = records[path.stem]
        cache_key = next(
            key for key in cache if Path(key).resolve() == Path(record.cbct_path).resolve()
        )
        groups.append(cache[cache_key]["sha256"])
        transform = np.load(record.transform_path)
        strata.append(int(np.linalg.det(transform[:3, :3]) < 0.0))
    return np.asarray(groups), np.asarray(strata)


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def run_epoch(
    model: CrownLocalizerUNet,
    loader: DataLoader,
    device: torch.device,
    background_weight: float,
    surface_tolerance_voxels: int,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    rows = []
    for images, labels, _, sample_weights in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        sample_weights = sample_weights.to(
            device=device, dtype=images.dtype, non_blocking=True
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss, parts = crown_localizer_loss(
                logits,
                labels,
                background_weight,
                surface_tolerance_voxels,
            )
            loss = loss * torch.mean(sample_weights)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        with torch.no_grad():
            dice_upper, dice_lower = hard_dice(torch.softmax(logits, dim=1), labels)
        rows.append(
            {
                "loss": float(loss.detach().cpu()),
                **parts,
                "hard_dice_upper": dice_upper,
                "hard_dice_lower": dice_lower,
            }
        )
    return average_metrics(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    paths = sorted(args.data_dir.glob("*.npz"))
    if len(paths) < args.folds:
        raise ValueError(f"Only {len(paths)} cases are available for {args.folds} folds")
    groups, strata = load_case_metadata(paths, args.manifest, args.cbct_hash_cache)
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    splits = list(splitter.split(np.zeros(len(paths)), strata, groups))
    split_payload = {
        str(fold): {
            "train": [paths[index].stem for index in train_indices],
            "validation": [paths[index].stem for index in validation_indices],
        }
        for fold, (train_indices, validation_indices) in enumerate(splits)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "splits.json").write_text(
        json.dumps(split_payload, indent=2), encoding="utf-8"
    )
    selected_folds = args.fold_indices if args.fold_indices else list(range(args.folds))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    summary = []

    for fold in selected_folds:
        fold_dir = args.output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = fold_dir / "best.pt"
        if args.skip_existing and checkpoint.exists():
            print(f"Fold {fold}: existing checkpoint retained", flush=True)
            continue
        train_indices, validation_indices = splits[fold]
        labeled_train_paths = [paths[index] for index in train_indices]
        validation_paths = [paths[index] for index in validation_indices]
        train_paths = list(labeled_train_paths)
        label_overrides: dict[Path, Path] = {}
        sample_weights: dict[Path, float] = {}
        pseudo_count = 0
        if args.pseudo_label_root is not None or args.unlabeled_data_dir is not None:
            if args.pseudo_label_root is None or args.unlabeled_data_dir is None:
                raise ValueError(
                    "--pseudo-label-root and --unlabeled-data-dir must be provided together"
                )
            pseudo_labels = sorted((args.pseudo_label_root / f"fold_{fold}").glob("*.npz"))
            if args.max_pseudo_cases > 0:
                pseudo_labels = pseudo_labels[: args.max_pseudo_cases]
            for pseudo_label in pseudo_labels:
                image_path = args.unlabeled_data_dir / pseudo_label.name
                if not image_path.exists():
                    continue
                train_paths.append(image_path)
                label_overrides[image_path] = pseudo_label
                sample_weights[image_path] = args.pseudo_weight
            pseudo_count = len(label_overrides)
        weighted_resampling = (
            pseudo_count > 0 and args.pseudo_weight_mode == "weighted_resampling"
        )
        train_dataset = CrownDataset(
            train_paths,
            augment=True,
            cache=args.cache_data,
            seed=args.seed + fold,
            label_overrides=label_overrides,
            sample_weights={} if weighted_resampling else sample_weights,
        )
        validation_dataset = CrownDataset(
            validation_paths, augment=False, cache=args.cache_data, seed=args.seed + fold
        )
        generator = torch.Generator().manual_seed(args.seed + fold)
        sampler = (
            PseudoWeightedEpochSampler(
                len(labeled_train_paths),
                pseudo_count,
                args.pseudo_weight,
                args.seed + fold,
            )
            if weighted_resampling
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=0,
            pin_memory=device.type == "cuda",
            generator=generator,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        model = CrownLocalizerUNet(base_channels=args.base_channels).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
        )
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        best_score = -np.inf
        stale = 0
        history = []
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                args.background_weight,
                args.surface_tolerance_voxels,
                optimizer,
                scaler,
            )
            with torch.no_grad():
                validation_metrics = run_epoch(
                    model,
                    validation_loader,
                    device,
                    args.background_weight,
                    args.surface_tolerance_voxels,
                    None,
                    scaler,
                )
            scheduler.step()
            score = 0.5 * (
                validation_metrics["hard_dice_upper"]
                + validation_metrics["hard_dice_lower"]
            )
            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                "validation_mean_hard_dice": score,
            }
            history.append(row)
            print(
                f"fold={fold} epoch={epoch:03d} train={train_metrics['loss']:.4f} "
                f"val={validation_metrics['loss']:.4f} dice={score:.4f}",
                flush=True,
            )
            if score > best_score + 1e-4:
                best_score = score
                stale = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "base_channels": args.base_channels,
                        "fold": fold,
                        "epoch": epoch,
                        "validation_mean_hard_dice": score,
                        "surface_tolerance_voxels": args.surface_tolerance_voxels,
                        "train_cases": [path.stem for path in labeled_train_paths],
                        "pseudo_cases": pseudo_count,
                        "pseudo_weight": args.pseudo_weight if pseudo_count else 0.0,
                        "pseudo_weight_mode": args.pseudo_weight_mode,
                        "effective_samples_per_epoch": len(train_loader),
                        "validation_cases": [path.stem for path in validation_paths],
                    },
                    checkpoint,
                )
            else:
                stale += 1
            if stale >= args.patience:
                print(f"fold={fold} early stop after {epoch} epochs", flush=True)
                break
        with (fold_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        summary.append(
            {
                "fold": fold,
                "best_validation_mean_hard_dice": float(best_score),
                "epochs": len(history),
                "train_cases": len(labeled_train_paths),
                "pseudo_cases": pseudo_count,
                "pseudo_weight": args.pseudo_weight if pseudo_count else 0.0,
                "pseudo_weight_mode": args.pseudo_weight_mode,
                "effective_samples_per_epoch": len(train_loader),
                "validation_cases": len(validation_paths),
                "surface_tolerance_voxels": args.surface_tolerance_voxels,
            }
        )
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Training outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
