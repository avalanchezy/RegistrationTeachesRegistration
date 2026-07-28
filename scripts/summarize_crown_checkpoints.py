from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the best epoch and training metadata from crown-localizer checkpoints."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-members", type=int, default=5)
    return parser.parse_args()


def checkpoint_summary(model_dir: Path, expected_members: int) -> dict[str, object]:
    members = []
    for index in range(expected_members):
        checkpoint_path = model_dir / f"fold_{index}" / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing crown-localizer checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        epoch = int(checkpoint.get("epoch", 0))
        if epoch < 1:
            raise ValueError(f"Invalid best epoch {epoch} in {checkpoint_path}")
        members.append(
            {
                "member": index,
                "checkpoint": str(checkpoint_path.resolve()),
                "epoch": epoch,
                "seed": int(checkpoint.get("seed", 0)),
                "pseudo_weight": float(checkpoint.get("pseudo_weight", 0.0)),
                "pseudo_cases": int(checkpoint.get("pseudo_cases", 0)),
                "validation_mean_hard_dice": checkpoint.get(
                    "validation_mean_hard_dice"
                ),
            }
        )
    return {
        "model_dir": str(model_dir.resolve()),
        "expected_members": expected_members,
        "epochs": [member["epoch"] for member in members],
        "members": members,
    }


def main() -> None:
    args = parse_args()
    summary = checkpoint_summary(args.model_dir, args.expected_members)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
