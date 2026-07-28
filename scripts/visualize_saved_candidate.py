from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.data import apply_transform, ios_pca_side_variants, load_ios_points, load_manifest
from task2reg.surfaces import threshold_aggregate_surface_candidates, threshold_surface_candidates
from task2reg.visualize import save_overlay_ply, save_registration_figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild and visualize a non-adaptive candidate saved in candidates.json."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--jaw", choices=("upper", "lower"), required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-points", type=int, default=12000)
    parser.add_argument("--thresholds", type=float, nargs="+", default=(1200, 1400, 1600, 1800, 2000))
    parser.add_argument("--aggregate-components", type=int, nargs="+", default=(2, 4))
    parser.add_argument("--ios-crop-fractions", type=float, nargs="+", default=(0.25, 0.35))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    records = {
        (record.case_id, record.jaw): record
        for record in load_manifest(args.manifest)
        if record.complete
    }
    record = records[(args.case_id, args.jaw)]
    candidate_path = args.run_dir / f"{args.case_id}_{args.jaw}" / "candidates.json"
    rows = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate = next(row for row in rows if int(row["unsupervised_rank"]) == args.rank)
    if "adaptive" in str(candidate["target_metadata"].get("mode", "")):
        raise ValueError("Adaptive local targets cannot be reconstructed from metadata alone")

    source, bounds = load_ios_points(Path(record.ios_path), args.source_points, args.seed)
    source_variants = dict(
        ios_pca_side_variants(
            source, fractions=tuple(args.ios_crop_fractions), include_full=False
        )
    )
    source_points = source_variants[candidate["source_variant"]]
    extent = bounds[1] - bounds[0]
    targets = threshold_surface_candidates(
        Path(record.cbct_path),
        args.jaw,
        extent,
        tuple(args.thresholds),
        seed=args.seed,
    )
    targets.extend(
        threshold_aggregate_surface_candidates(
            Path(record.cbct_path),
            extent,
            tuple(args.thresholds),
            tuple(args.aggregate_components),
            seed=args.seed + 17,
        )
    )
    target_map = {target.name: target.points for target in targets}
    if candidate["target"] not in target_map:
        raise KeyError(f"Could not reconstruct target {candidate['target']}")
    target_points = target_map[candidate["target"]]
    initial = np.asarray(candidate["transform_initial"], dtype=np.float64)
    prediction = np.asarray(candidate["transform"], dtype=np.float64)
    ground_truth = np.load(record.transform_path) if record.transform_path else None
    quality = (
        f"TRE={float(candidate['mean_tre_mm']):.3f} mm"
        if "mean_tre_mm" in candidate
        else f"selection score={float(candidate['selection_score_mm']):.3f} mm"
    )
    title = (
        f"{args.case_id} {args.jaw} | saved rank {args.rank} | "
        f"{quality} | {candidate['target']}"
    )
    save_registration_figure(
        args.output, source_points, target_points, initial, prediction, ground_truth, title
    )
    prediction_points = apply_transform(source, prediction)
    truth_points = apply_transform(source, ground_truth) if ground_truth is not None else None
    save_overlay_ply(args.output.with_suffix(".ply"), target_points, prediction_points, truth_points)
    print(f"Saved {args.output} and {args.output.with_suffix('.ply')}")


if __name__ == "__main__":
    main()
