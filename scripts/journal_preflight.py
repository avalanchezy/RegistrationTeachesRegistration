"""Check the target environment and journal protocol before expensive training."""
import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2reg.journal.config import TrainingConfig
from task2reg.journal.data import load_journal_manifest, load_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; install a compatible PyTorch build/driver")
    config = TrainingConfig.from_json(args.config) if args.config else None
    rows = load_journal_manifest(args.manifest) if args.manifest else []
    if args.check_data and not rows:
        raise SystemExit("--check-data requires --manifest")
    warnings, checked = [], 0
    for row in rows if args.check_data else []:
        case = load_case(row)
        if min(case["image"].shape) < 16:
            raise ValueError(f"{row['case_id']}: five-level U-Net requires each dimension >=16")
        if float(np.std(case["image"].astype(np.float32))) < 1.:
            warnings.append(f"{row['case_id']}: almost constant HU volume; inspect preprocessing before training")
        if row.get("surface_definition") == "full_ios":
            warnings.append(f"{row['case_id']} {row['jaw']}: full IOS fallback; supply a fixed crown crop for research")
        if config and row["split"] == "train":
            if config.support_weight and "label" not in case:
                raise ValueError("support loss requires a manual training label volume")
            if config.rank_weight and ("candidates" not in case or len(case["candidates"]) < 2):
                raise ValueError("ranking requires >=2 OOF/geometry-only candidates per training jaw")
        if config and row["split"] == "val" and config.validation_metric == "selected_D_mm" and not len(case.get("candidates", [])):
            raise ValueError("selected_D_mm validation requires cached validation candidates")
        checked += 1
    report = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
              "device": str(device), "cuda_available": torch.cuda.is_available(),
              "cuda_runtime": torch.version.cuda, "checked_records": checked,
              "split_counts": {split: sum(r["split"] == split for r in rows) for split in sorted({r["split"] for r in rows})},
              "patients": len({r["patient_id"] for r in rows}), "warnings": sorted(set(warnings)),
              "real_registration_evidence": False}
    if device.type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
