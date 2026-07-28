from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.crown_inference import CrownLocalizerEnsemble, CrownPostprocessConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure persistent crown-ensemble loading and inference resources."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    config = CrownPostprocessConfig.load(args.postprocess)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    ensemble = CrownLocalizerEnsemble(
        model_dir=args.model_dir,
        config=config,
        device=args.device,
        tta_mode="none",
    )
    synchronize(ensemble.device)
    load_seconds = time.perf_counter() - started

    image = nib.load(str(args.image))
    inference_seconds = []
    statistics: dict[str, float | int] = {}
    for _ in range(args.repeats):
        started = time.perf_counter()
        _, _, _, statistics = ensemble.predict_image(image)
        synchronize(ensemble.device)
        inference_seconds.append(time.perf_counter() - started)

    if ensemble.device.type == "cuda":
        allocated_bytes = int(torch.cuda.max_memory_allocated(ensemble.device))
        reserved_bytes = int(torch.cuda.max_memory_reserved(ensemble.device))
        device_name = torch.cuda.get_device_name(ensemble.device)
    else:
        allocated_bytes = 0
        reserved_bytes = 0
        device_name = str(ensemble.device)

    payload = {
        "model_dir": str(args.model_dir.resolve()),
        "image": str(args.image.resolve()),
        "image_shape": list(image.shape),
        "device": str(ensemble.device),
        "device_name": device_name,
        "models": len(ensemble.models),
        "branches": len(ensemble.branch_metadata),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "mean_inference_seconds": sum(inference_seconds) / len(inference_seconds),
        "peak_allocated_bytes": allocated_bytes,
        "peak_reserved_bytes": reserved_bytes,
        "peak_allocated_gib": allocated_bytes / (1024**3),
        "peak_reserved_gib": reserved_bytes / (1024**3),
        "prediction_statistics": statistics,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
