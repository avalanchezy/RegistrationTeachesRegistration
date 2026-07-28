from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend two aligned crown-localizer probability directories."
    )
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--first-weight", type=float, required=True)
    parser.add_argument("--first-weight-upper", type=float)
    parser.add_argument("--first-weight-lower", type=float)
    parser.add_argument(
        "--mode", choices=("arithmetic", "geometric"), default="arithmetic"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def blend_probability_arrays(
    first: np.ndarray,
    second: np.ndarray,
    first_weight: float,
    mode: str = "arithmetic",
    first_weight_upper: float | None = None,
    first_weight_lower: float | None = None,
) -> np.ndarray:
    if first.shape != second.shape:
        raise ValueError(f"Probability shape mismatch: {first.shape} vs {second.shape}")
    class_specific = first_weight_upper is not None or first_weight_lower is not None
    if class_specific and (first_weight_upper is None or first_weight_lower is None):
        raise ValueError("Both upper and lower weights are required for class-specific fusion")
    if class_specific:
        assert first_weight_upper is not None and first_weight_lower is not None
        class_weights = np.asarray(
            [
                0.5 * (first_weight_upper + first_weight_lower),
                first_weight_upper,
                first_weight_lower,
            ],
            dtype=np.float32,
        ).reshape((3,) + (1,) * (first.ndim - 1))
    else:
        class_weights = np.asarray(first_weight, dtype=np.float32)
    if mode == "arithmetic":
        probabilities = class_weights * first + (1.0 - class_weights) * second
        if not class_specific:
            return probabilities
        epsilon = np.finfo(np.float32).tiny
        return probabilities / np.maximum(
            np.sum(probabilities, axis=0, keepdims=True), epsilon
        )
    if mode == "geometric":
        epsilon = np.finfo(np.float32).tiny
        log_probabilities = class_weights * np.log(np.clip(first, epsilon, 1.0))
        log_probabilities += (1.0 - class_weights) * np.log(
            np.clip(second, epsilon, 1.0)
        )
        log_probabilities -= np.max(log_probabilities, axis=0, keepdims=True)
        probabilities = np.exp(log_probabilities)
        return probabilities / np.maximum(
            np.sum(probabilities, axis=0, keepdims=True), epsilon
        )
    raise ValueError(f"Unknown probability blend mode: {mode}")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.first_weight <= 1.0:
        raise ValueError("--first-weight must be between zero and one")
    jaw_weights = (args.first_weight_upper, args.first_weight_lower)
    if any(weight is not None for weight in jaw_weights):
        if any(weight is None or not 0.0 <= weight <= 1.0 for weight in jaw_weights):
            raise ValueError("Both jaw-specific weights must be between zero and one")
    first_paths = {path.name: path for path in args.first.glob("*.npz")}
    second_paths = {path.name: path for path in args.second.glob("*.npz")}
    if not first_paths or first_paths.keys() != second_paths.keys():
        missing_first = sorted(second_paths.keys() - first_paths.keys())
        missing_second = sorted(first_paths.keys() - second_paths.keys())
        raise RuntimeError(
            "Probability directories must contain the same non-empty case set; "
            f"missing_first={missing_first}, missing_second={missing_second}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(first_paths):
        with np.load(first_paths[name], allow_pickle=False) as first_payload:
            first = first_payload["probabilities"].astype(np.float32)
            first_affine = first_payload["affine"].astype(np.float64)
        with np.load(second_paths[name], allow_pickle=False) as second_payload:
            second = second_payload["probabilities"].astype(np.float32)
            second_affine = second_payload["affine"].astype(np.float64)
        if not np.allclose(first_affine, second_affine, atol=1e-6):
            raise ValueError(f"Affine mismatch for {name}")
        try:
            blended = blend_probability_arrays(
                first,
                second,
                args.first_weight,
                args.mode,
                args.first_weight_upper,
                args.first_weight_lower,
            )
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
        np.savez_compressed(
            args.output_dir / name,
            probabilities=blended.astype(np.float16),
            affine=first_affine,
        )
    summary = {
        "cases": len(first_paths),
        "first": str(args.first.resolve()),
        "second": str(args.second.resolve()),
        "first_weight": args.first_weight,
        "second_weight": 1.0 - args.first_weight,
        "first_weight_upper": args.first_weight_upper,
        "first_weight_lower": args.first_weight_lower,
        "mode": args.mode,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
