from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.data import apply_transform, ios_pca_side_variants, load_ios_points, load_manifest
from task2reg.geometry import (
    RegistrationResult,
    bidirectional_fit_score,
    fit_score,
    register_geometry,
    robust_trimmed_icp,
    stochastic_basin_refinement,
    transform_points as geometry_transform_points,
)
from task2reg.metrics import evaluate_transform
from task2reg.priors import fit_rotation_prior, load_rotation_priors
from task2reg.surfaces import (
    SurfaceCandidate,
    crown_guided_cbct_surface_candidates,
    crown_mask_surface_candidates,
    crown_probability_surface_candidates,
    threshold_aggregate_surface_candidates,
    threshold_roi_surface_candidates,
    threshold_surface_candidates,
    toothseg_surface_candidates,
)


def _toothseg_path(root: Path, case_id: str) -> Path:
    return root / f"STS2_{case_id}.nii.gz"


def _threshold_volume_path(root: Path, case_id: str) -> Path:
    return root / f"STS2_{case_id}_0000.nii.gz"


def _crown_probability_path(root: Path, case_id: str) -> Path:
    return root / f"{case_id}.npz"


def metadata_chirality(cbct_path: Path) -> int:
    image = nib.load(str(cbct_path))
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    reflection_protocol = tuple(image.shape) == (266, 266, 200) and np.allclose(spacing, 0.3, atol=1e-4)
    return -1 if reflection_protocol else 1


def select_basin_candidates(candidates: list, top_k: int, strategy: str) -> list:
    ranked = sorted(candidates, key=lambda item: item[0])
    if strategy == "global" or top_k <= 0:
        return ranked[:top_k]

    selected = []
    selected_ids: set[int] = set()
    seen_groups: set[tuple[str, ...]] = set()
    for item in ranked:
        source_name = item[3]
        target_name = item[1].name
        target_family = next(
            (family for family in ("adaptive", "tracked", "aggregate") if family in target_name),
            "other",
        )
        group = (
            (source_name, target_family)
            if strategy == "source-target-diverse"
            else (source_name,)
        )
        if group in seen_groups:
            continue
        selected.append(item)
        selected_ids.add(id(item))
        seen_groups.add(group)
        if len(selected) == top_k:
            return selected
    for item in ranked:
        if id(item) in selected_ids:
            continue
        selected.append(item)
        if len(selected) == top_k:
            break
    return selected


def _summary_row_from_result_payload(
    payload: dict[str, object],
    *,
    target_mode: str,
    chirality_mode: str,
    expected_case_id: str,
    expected_jaw: str,
) -> dict[str, object]:
    record = payload.get("record")
    target = payload.get("target")
    registration = payload.get("registration")
    selection = payload.get("selection")
    oracle = payload.get("oracle_diagnostic")
    metrics = payload.get("metrics", {})
    if not all(isinstance(item, dict) for item in (record, target, registration, selection, oracle, metrics)):
        raise ValueError("Incomplete registration result payload")
    if record.get("case_id") != expected_case_id or record.get("jaw") != expected_jaw:
        raise ValueError("Registration result belongs to a different case or jaw")
    return {
        "case_id": expected_case_id,
        "jaw": expected_jaw,
        "target_mode": target_mode,
        "target": target["name"],
        "source_variant": payload["source_variant"],
        "method": registration["method"],
        "chirality": registration["chirality"],
        "chirality_mode": chirality_mode,
        "selection_score_mm": selection["score"],
        "rank_score_mm": selection["rank_score"],
        "prior_angle_deg": selection["prior_angle_deg"],
        "fit_score_mm": registration["score"],
        "fit_median_mm": registration["median_distance"],
        "fit_p90_mm": registration["p90_distance"],
        "overlap_2mm": registration["overlap_2mm"],
        "oracle_mean_tre_mm": oracle.get("mean_tre_mm", ""),
        "oracle_unsupervised_rank": oracle["unsupervised_rank"],
        "oracle_method": oracle["method"],
        "oracle_target": oracle["target"],
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="Train-Labeled")
    parser.add_argument("--case-ids", nargs="*", default=[])
    parser.add_argument("--jaws", nargs="+", choices=("upper", "lower"), default=("upper", "lower"))
    parser.add_argument(
        "--target-mode",
        choices=(
            "threshold",
            "toothseg",
            "crown",
            "crown-probability",
            "crown-guided",
        ),
        default="threshold",
    )
    parser.add_argument(
        "--tracked-jaw-mode",
        choices=("requested", "opposite", "both"),
        default="requested",
        help="Track the requested, opposite, or both z-ordered arch assignments.",
    )
    parser.add_argument("--toothseg-dir", type=Path)
    parser.add_argument("--crown-mask-dir", type=Path)
    parser.add_argument("--crown-probability-dir", type=Path)
    parser.add_argument(
        "--crown-guided-thresholds",
        nargs="+",
        type=float,
        default=(500.0, 800.0, 1100.0, 1400.0, 1700.0),
    )
    parser.add_argument(
        "--crown-guidance-radii-mm",
        nargs="+",
        type=float,
        default=(2.5, 4.0, 6.0),
    )
    parser.add_argument(
        "--crown-probability-thresholds",
        nargs="+",
        type=float,
        default=(0.25, 0.35, 0.50, 0.70),
    )
    parser.add_argument(
        "--crown-probability-voxel-counts",
        nargs="*",
        type=int,
        default=(1500, 2500, 4000),
    )
    parser.add_argument(
        "--threshold-volume-dir",
        type=Path,
        help="Optional directory of affine-preserving unsupervised dental ROI NIfTIs.",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=(1400, 1600, 1800, 2000))
    parser.add_argument("--adaptive-thresholds", nargs="*", type=float, default=())
    parser.add_argument("--aggregate-components", nargs="*", type=int, default=())
    parser.add_argument("--adaptive-top-k", type=int, default=6)
    parser.add_argument("--adaptive-radius-mm", type=float, default=6.0)
    parser.add_argument("--methods", nargs="+", choices=("pca", "fgr", "ransac"), default=("pca", "fgr", "ransac"))
    parser.add_argument("--source-points", type=int, default=12000)
    parser.add_argument("--ios-source-mode", choices=("full", "sides", "all"), default="all")
    parser.add_argument("--ios-crop-fractions", nargs="+", type=float, default=(0.25, 0.35))
    parser.add_argument("--max-target-candidates", type=int, default=6)
    parser.add_argument("--pca-refine-top-k", type=int, default=24)
    parser.add_argument("--basin-refine-top-k", type=int, default=3)
    parser.add_argument("--basin-samples", type=int, default=256)
    parser.add_argument(
        "--basin-selection",
        choices=("global", "source-diverse", "source-target-diverse"),
        default="global",
        help="Allocate stochastic searches globally or first across distinct IOS source crops.",
    )
    parser.add_argument("--transform-prior", type=Path)
    parser.add_argument(
        "--leave-one-cbct-group-out-prior",
        action="store_true",
        help=(
            "For labeled evaluation, fit each case prior after excluding every "
            "labeled case with the same cached CBCT hash."
        ),
    )
    parser.add_argument(
        "--cbct-hash-cache",
        type=Path,
        help="Required with --leave-one-cbct-group-out-prior.",
    )
    parser.add_argument(
        "--prior-max-angle-deg",
        type=float,
        default=90.0,
        help="Reject 180-degree arch-symmetry solutions outside the labeled protocol family.",
    )
    parser.add_argument(
        "--prior-angle-weight",
        type=float,
        default=0.002,
        help="Registration-score penalty in mm per degree from the learned protocol mean.",
    )
    parser.add_argument("--ransac-iterations", type=int, default=50000)
    parser.add_argument(
        "--chirality-mode",
        choices=("metadata", "both"),
        default="metadata",
        help="Use the acquisition-protocol coordinate prior or search both determinant families.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--stable-record-seeds",
        action="store_true",
        help="Derive each case/jaw seed independently so chunking cannot change candidates.",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip PNG and PLY diagnostics during production inference.",
    )
    parser.add_argument(
        "--resume-completed-records",
        action="store_true",
        help="Reuse complete per-case result.json files already present in the output directory.",
    )
    args = parser.parse_args()
    rotation_priors = load_rotation_priors(args.transform_prior) if args.transform_prior else {}
    manifest_records = load_manifest(args.manifest)
    grouped_rotation_priors: dict[str, dict[str, object]] = {}
    grouped_prior_audit: dict[str, dict[str, object]] = {}
    if args.leave_one_cbct_group_out_prior:
        if args.split != "Train-Labeled":
            raise ValueError(
                "--leave-one-cbct-group-out-prior is only valid for Train-Labeled"
            )
        if args.cbct_hash_cache is None:
            raise ValueError(
                "--cbct-hash-cache is required with --leave-one-cbct-group-out-prior"
            )
        hash_cache = json.loads(args.cbct_hash_cache.read_text(encoding="utf-8-sig"))
        case_hashes: dict[str, str] = {}
        for manifest_record in manifest_records:
            if manifest_record.split != "Train-Labeled":
                continue
            cached = hash_cache.get(str(Path(manifest_record.cbct_path)))
            if not isinstance(cached, dict) or not cached.get("sha256"):
                raise KeyError(
                    f"Missing cached CBCT hash for {manifest_record.cbct_path}"
                )
            case_hashes[manifest_record.case_id] = str(cached["sha256"]).lower()
        for case_id, cbct_hash in case_hashes.items():
            excluded = {
                other_case
                for other_case, other_hash in case_hashes.items()
                if other_hash == cbct_hash
            }
            case_priors = {
                jaw: fit_rotation_prior(manifest_records, jaw, excluded_cases=excluded)
                for jaw in ("upper", "lower")
            }
            grouped_rotation_priors[case_id] = case_priors
            grouped_prior_audit[case_id] = {
                "cbct_sha256": cbct_hash,
                "excluded_case_ids": sorted(excluded),
                "training_counts": {
                    jaw: int(len(prior.training_angles_deg))
                    for jaw, prior in case_priors.items()
                },
            }

    selected = []
    wanted = set(args.case_ids)
    for record in manifest_records:
        if record.split != args.split or record.jaw not in args.jaws or not record.complete:
            continue
        if wanted and record.case_id not in wanted:
            continue
        selected.append(record)
    if not selected:
        raise RuntimeError("No records matched the requested filters")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    resumed_records = 0
    for index, record in enumerate(selected):
        print(f"[{index + 1}/{len(selected)}] {record.case_id} {record.jaw}", flush=True)
        case_dir = args.output_dir / f"{record.case_id}_{record.jaw}"
        result_path = case_dir / "result.json"
        if args.resume_completed_records and result_path.is_file():
            try:
                resumed_row = _summary_row_from_result_payload(
                    json.loads(result_path.read_text(encoding="utf-8")),
                    target_mode=args.target_mode,
                    chirality_mode=args.chirality_mode,
                    expected_case_id=record.case_id,
                    expected_jaw=record.jaw,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                print(f"  incomplete resume artifact; recomputing: {error}", flush=True)
            else:
                rows.append(resumed_row)
                resumed_records += 1
                print("  resumed complete result", flush=True)
                continue
        if args.stable_record_seeds:
            seed_payload = f"{args.seed}|{record.case_id}|{record.jaw}".encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:4], "little")
        else:
            seed = args.seed + index * 101
        active_rotation_priors = grouped_rotation_priors.get(
            record.case_id, rotation_priors
        )
        source, bounds = load_ios_points(Path(record.ios_path), args.source_points, seed)
        extent = bounds[1] - bounds[0]
        chirality_hint = metadata_chirality(Path(record.cbct_path)) if args.chirality_mode == "metadata" else None
        threshold_volume = Path(record.cbct_path)
        if args.target_mode == "threshold" and args.threshold_volume_dir is not None:
            threshold_volume = _threshold_volume_path(args.threshold_volume_dir, record.case_id)
            if not threshold_volume.exists():
                failures.append(
                    {
                        "case_id": record.case_id,
                        "jaw": record.jaw,
                        "stage": "target_extraction",
                        "error": f"Missing threshold ROI: {threshold_volume}",
                    }
                )
                print(f"  missing threshold ROI: {threshold_volume}", file=sys.stderr, flush=True)
                continue
        if args.ios_source_mode == "full":
            source_variants = [("full", source)]
        else:
            source_variants = ios_pca_side_variants(
                source,
                fractions=tuple(args.ios_crop_fractions),
                include_full=args.ios_source_mode == "all",
            )
        try:
            if args.target_mode == "threshold":
                targets = []
                target_errors = []
                opposite_jaw = "lower" if record.jaw == "upper" else "upper"
                if args.tracked_jaw_mode == "requested":
                    tracked_jaws = (record.jaw,)
                elif args.tracked_jaw_mode == "opposite":
                    tracked_jaws = (opposite_jaw,)
                else:
                    tracked_jaws = ("upper", "lower")
                for tracked_jaw in tracked_jaws:
                    try:
                        tracked_targets = threshold_surface_candidates(
                            threshold_volume,
                            tracked_jaw,
                            extent,
                            tuple(args.thresholds),
                            seed=seed,
                        )
                        if args.tracked_jaw_mode != "requested":
                            for target in tracked_targets:
                                target.name = f"axial_{tracked_jaw}_{target.name}"
                                target.metadata["axial_assignment"] = tracked_jaw
                        targets.extend(tracked_targets)
                    except (RuntimeError, ValueError, IndexError) as error:
                        target_errors.append(f"tracked-{tracked_jaw}: {error}")
                        print(
                            f"  tracked {tracked_jaw} targets unavailable: {error}", flush=True
                        )
                if args.aggregate_components:
                    try:
                        targets.extend(
                            threshold_aggregate_surface_candidates(
                                threshold_volume,
                                extent,
                                tuple(args.thresholds),
                                tuple(args.aggregate_components),
                                seed=seed + 17,
                            )
                        )
                    except (RuntimeError, ValueError, IndexError) as error:
                        target_errors.append(f"aggregate: {error}")
                        print(f"  aggregate targets unavailable: {error}", flush=True)
                if not targets:
                    raise RuntimeError("; ".join(target_errors) or "No threshold targets")
            elif args.target_mode == "toothseg":
                if args.toothseg_dir is None:
                    raise ValueError("--toothseg-dir is required for target-mode toothseg")
                targets = toothseg_surface_candidates(
                    _toothseg_path(args.toothseg_dir, record.case_id), record.jaw, seed=seed
                )
            elif args.target_mode == "crown":
                if args.crown_mask_dir is None:
                    raise ValueError("--crown-mask-dir is required for target-mode crown")
                targets = crown_mask_surface_candidates(
                    _toothseg_path(args.crown_mask_dir, record.case_id), record.jaw, seed=seed
                )
            elif args.target_mode == "crown-probability":
                if args.crown_probability_dir is None:
                    raise ValueError(
                        "--crown-probability-dir is required for target-mode crown-probability"
                    )
                targets = crown_probability_surface_candidates(
                    _crown_probability_path(args.crown_probability_dir, record.case_id),
                    record.jaw,
                    probability_thresholds=tuple(args.crown_probability_thresholds),
                    voxel_counts=tuple(args.crown_probability_voxel_counts),
                    seed=seed,
                )
            else:
                if args.crown_mask_dir is None:
                    raise ValueError(
                        "--crown-mask-dir is required for target-mode crown-guided"
                    )
                targets = crown_guided_cbct_surface_candidates(
                    threshold_volume,
                    _toothseg_path(args.crown_mask_dir, record.case_id),
                    record.jaw,
                    thresholds=tuple(args.crown_guided_thresholds),
                    guidance_radii_mm=tuple(args.crown_guidance_radii_mm),
                    seed=seed,
                )
            for target in targets:
                target.metadata["volume_path"] = str(threshold_volume)
        except (RuntimeError, ValueError, IndexError) as error:
            failures.append(
                {
                    "case_id": record.case_id,
                    "jaw": record.jaw,
                    "stage": "target_extraction",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"  target extraction failed: {error}", file=sys.stderr, flush=True)
            continue
        targets = sorted(targets, key=lambda item: float(item.metadata.get("rank_score", 0.0)))[: args.max_target_candidates]
        jaw_reference_center = targets[0].points.mean(axis=0)
        if args.target_mode in (
            "toothseg",
            "crown",
            "crown-probability",
            "crown-guided",
        ):
            try:
                arch_reference = threshold_surface_candidates(
                    Path(record.cbct_path),
                    record.jaw,
                    extent,
                    (1600.0,),
                    points_per_candidate=1000,
                    seed=seed,
                )[0]
                arch_min = np.asarray(arch_reference.metadata["anchor_bbox_min"], dtype=np.float64)
                arch_max = np.asarray(arch_reference.metadata["anchor_bbox_max"], dtype=np.float64)
                jaw_reference_center = (arch_min + arch_max) * 0.5
            except (RuntimeError, ValueError, IndexError, KeyError):
                print("  warning: threshold arch reference unavailable; using ToothSeg centroid", flush=True)
        source_full_centroid = source.mean(axis=0)

        case_results = []
        for target_index, target in enumerate(targets):
            print(f"  target {target_index + 1}/{len(targets)}: {target.name}", flush=True)
            for source_index, (source_name, source_points) in enumerate(source_variants):
                print(f"    source: {source_name} ({len(source_points)} points)", flush=True)
                results = register_geometry(
                    source_points,
                    target.points,
                    methods=tuple(args.methods),
                    pca_refine_top_k=args.pca_refine_top_k,
                    ransac_iterations=args.ransac_iterations,
                    seed=seed + target_index * 31 + source_index,
                    chirality_hint=chirality_hint,
                )
                if record.jaw in active_rotation_priors and chirality_hint is not None:
                    initial = active_rotation_priors[record.jaw].centered_initialization(
                        source_points, target.points, chirality_hint
                    )
                    target_tree = cKDTree(target.points)
                    initial_metrics = fit_score(
                        geometry_transform_points(source_points, initial), target_tree
                    )
                    refined, correspondences = robust_trimmed_icp(
                        source_points, target.points, initial
                    )
                    refined_metrics = fit_score(
                        geometry_transform_points(source_points, refined), target_tree
                    )
                    results.append(
                        RegistrationResult(
                            method="protocol-prior",
                            chirality=chirality_hint,
                            transform_initial=initial,
                            transform=refined,
                            score_initial=initial_metrics["score"],
                            score=refined_metrics["score"],
                            median_distance=refined_metrics["median"],
                            p90_distance=refined_metrics["p90"],
                            overlap_2mm=refined_metrics["overlap_2mm"],
                            correspondences=correspondences,
                        )
                    )
                for result in results:
                    case_results.append((result.score, target, result, source_name, source_points))
        if not case_results:
            print("  no valid candidates", file=sys.stderr, flush=True)
            failures.append(
                {
                    "case_id": record.case_id,
                    "jaw": record.jaw,
                    "stage": "registration",
                    "error": "No valid registration candidates",
                }
            )
            continue
        if args.target_mode == "threshold" and args.adaptive_thresholds:
            anchor_target = next(
                (target for target in targets if "anchor_bbox_min" in target.metadata),
                targets[0],
            )
            anchor_min = np.asarray(
                anchor_target.metadata.get("anchor_bbox_min", anchor_target.metadata["bbox_min"]),
                dtype=np.float64,
            )
            anchor_max = np.asarray(
                anchor_target.metadata.get("anchor_bbox_max", anchor_target.metadata["bbox_max"]),
                dtype=np.float64,
            )
            adaptive_targets = threshold_roi_surface_candidates(
                threshold_volume,
                anchor_min,
                anchor_max,
                thresholds=tuple(args.adaptive_thresholds),
                seed=seed,
            )
            adaptive_pool = case_results
            if record.jaw in active_rotation_priors:
                prior = active_rotation_priors[record.jaw]
                accepted = [
                    item for item in adaptive_pool
                    if prior.angle_deg(item[2].transform[:3, :3]) <= args.prior_max_angle_deg
                ]
                adaptive_pool = accepted or adaptive_pool
            for adaptive_index, (_, coarse_target, coarse, source_name, source_points) in enumerate(
                sorted(adaptive_pool, key=lambda item: item[0])[: args.adaptive_top_k]
            ):
                moved = geometry_transform_points(source_points, coarse.transform)
                moved_tree = cKDTree(moved)
                for roi_target in adaptive_targets:
                    distances, _ = moved_tree.query(roi_target.points, workers=-1)
                    local_points = roi_target.points[distances <= args.adaptive_radius_mm]
                    if len(local_points) < 128:
                        continue
                    adaptive_target = SurfaceCandidate(
                        name=f"{roi_target.name}_from_{coarse_target.name}",
                        points=local_points,
                        metadata={
                            **roi_target.metadata,
                            "coarse_target": coarse_target.name,
                            "adaptive_radius_mm": args.adaptive_radius_mm,
                            "local_surface_points": int(len(local_points)),
                        },
                    )
                    refined, correspondences = robust_trimmed_icp(
                        source_points,
                        adaptive_target.points,
                        coarse.transform,
                        distance_schedule=(4.0, 2.5, 1.5, 1.0),
                        iterations_per_level=8,
                    )
                    adaptive_tree = cKDTree(adaptive_target.points)
                    metrics = fit_score(
                        geometry_transform_points(source_points, refined), adaptive_tree
                    )
                    result = RegistrationResult(
                        method=f"{coarse.method}+adaptive",
                        chirality=coarse.chirality,
                        transform_initial=coarse.transform,
                        transform=refined,
                        score_initial=coarse.score,
                        score=metrics["score"],
                        median_distance=metrics["median"],
                        p90_distance=metrics["p90"],
                        overlap_2mm=metrics["overlap_2mm"],
                        correspondences=correspondences,
                    )
                    case_results.append(
                        (result.score, adaptive_target, result, source_name, source_points)
                    )
                print(
                    f"  adaptive refine {adaptive_index + 1}/{min(args.adaptive_top_k, len(adaptive_pool))}: "
                    f"{source_name} from {coarse_target.name}",
                    flush=True,
                )
        if args.basin_refine_top_k > 0:
            basin_pool = case_results
            if record.jaw in active_rotation_priors:
                prior = active_rotation_priors[record.jaw]
                accepted = [
                    item
                    for item in basin_pool
                    if prior.angle_deg(item[2].transform[:3, :3]) <= args.prior_max_angle_deg
                ]
                basin_pool = accepted or basin_pool
            base_top = select_basin_candidates(
                basin_pool,
                args.basin_refine_top_k,
                args.basin_selection,
            )
            for basin_index, (_, target, result, source_name, source_points) in enumerate(base_top):
                print(
                    f"  basin refine {basin_index + 1}/{len(base_top)}: "
                    f"{source_name} -> {target.name}",
                    flush=True,
                )
                refined_transform, correspondence_count = stochastic_basin_refinement(
                    source_points,
                    target.points,
                    result.transform,
                    samples_per_stage=args.basin_samples,
                    seed=seed + 10000 + basin_index,
                )
                target_tree = cKDTree(target.points)
                refined_metrics = fit_score(
                    geometry_transform_points(source_points, refined_transform), target_tree
                )
                refined_result = RegistrationResult(
                    method=f"{result.method}+basin",
                    chirality=result.chirality,
                    transform_initial=result.transform,
                    transform=refined_transform,
                    score_initial=result.score,
                    score=refined_metrics["score"],
                    median_distance=refined_metrics["median"],
                    p90_distance=refined_metrics["p90"],
                    overlap_2mm=refined_metrics["overlap_2mm"],
                    correspondences=correspondence_count,
                )
                case_results.append(
                    (refined_result.score, target, refined_result, source_name, source_points)
                )
        selection_metrics: dict[int, dict[str, float]] = {}
        rescored_results = []
        for _, target, result, source_name, source_points in case_results:
            selection = bidirectional_fit_score(
                geometry_transform_points(source_points, result.transform), target.points
            )
            selection_metrics[id(result)] = selection
            prior_angle = (
                active_rotation_priors[record.jaw].angle_deg(result.transform[:3, :3])
                if record.jaw in active_rotation_priors
                else 0.0
            )
            selection["prior_angle_deg"] = prior_angle
            selection["prior_accepted"] = float(
                not active_rotation_priors or prior_angle <= args.prior_max_angle_deg
            )
            prior_penalty = args.prior_angle_weight * prior_angle
            selection["rank_score"] = selection["score"] + prior_penalty
            rescored_results.append(
                (selection["rank_score"], target, result, source_name, source_points)
            )
        case_results = rescored_results
        ground_truth = np.load(record.transform_path) if record.transform_path else None
        accepted_results = [
            item for item in case_results if selection_metrics[id(item[2])]["prior_accepted"] > 0.5
        ]
        ranked_results = sorted(accepted_results or case_results, key=lambda item: item[0])
        _, best_target, best, best_source_name, best_source = ranked_results[0]
        metrics = evaluate_transform(source, best.transform, ground_truth) if ground_truth is not None else {}
        case_dir.mkdir(parents=True, exist_ok=True)
        candidate_rows = []
        candidate_payloads = []
        for rank, (_, candidate_target, candidate_result, source_name, source_points) in enumerate(ranked_results, start=1):
            diagnostic = (
                evaluate_transform(source, candidate_result.transform, ground_truth)
                if ground_truth is not None
                else {}
            )
            candidate_rows.append(
                {
                    "unsupervised_rank": rank,
                    "target": candidate_target.name,
                    "source_variant": source_name,
                    "method": candidate_result.method,
                    "chirality": candidate_result.chirality,
                    "selection_score_mm": selection_metrics[id(candidate_result)]["score"],
                    "rank_score_mm": selection_metrics[id(candidate_result)]["rank_score"],
                    "prior_angle_deg": selection_metrics[id(candidate_result)]["prior_angle_deg"],
                    "prior_accepted": int(selection_metrics[id(candidate_result)]["prior_accepted"]),
                    "fit_score_mm": candidate_result.score,
                    "fit_score_initial_mm": candidate_result.score_initial,
                    "fit_median_mm": candidate_result.median_distance,
                    "fit_p90_mm": candidate_result.p90_distance,
                    "overlap_2mm": candidate_result.overlap_2mm,
                    "target_trimmed_score_mm": selection_metrics[id(candidate_result)]["target_score"],
                    "target_coverage_1mm": selection_metrics[id(candidate_result)]["target_coverage_1mm"],
                    "target_coverage_2mm": selection_metrics[id(candidate_result)]["target_coverage_2mm"],
                    "correspondences": candidate_result.correspondences,
                    **diagnostic,
                }
            )
            candidate_payloads.append(
                {
                    **candidate_rows[-1],
                    "transform_initial": candidate_result.transform_initial.tolist(),
                    "transform": candidate_result.transform.tolist(),
                    "source_full_centroid": source_full_centroid.tolist(),
                    "predicted_full_centroid": apply_transform(
                        source_full_centroid[None], candidate_result.transform
                    )[0].tolist(),
                    "jaw_reference_center": jaw_reference_center.tolist(),
                    "target_metadata": candidate_target.metadata,
                }
            )
        with (case_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(candidate_rows)
        (case_dir / "candidates.json").write_text(
            json.dumps(candidate_payloads, indent=2), encoding="utf-8"
        )
        oracle_row = min(candidate_rows, key=lambda item: item.get("mean_tre_mm", float("inf")))
        payload = {
            "record": record.__dict__,
            "target": {"name": best_target.name, **best_target.metadata},
            "source_variant": best_source_name,
            "registration": best.json_dict(),
            "selection": selection_metrics[id(best)],
            "metrics": metrics,
            "oracle_diagnostic": oracle_row,
        }
        (case_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not args.no_visualizations:
            from task2reg.visualize import save_overlay_ply, save_registration_figure

            truth_points = apply_transform(source, ground_truth) if ground_truth is not None else None
            prediction_points = apply_transform(source, best.transform)
            save_registration_figure(
                case_dir / "stages.png",
                best_source,
                best_target.points,
                best.transform_initial,
                best.transform,
                ground_truth,
                title=(
                    f"{record.case_id} {record.jaw} | {best_source_name} -> {best_target.name} | "
                    f"{best.method} det={best.chirality:+d}"
                ),
            )
            save_overlay_ply(
                case_dir / "overlay.ply", best_target.points, prediction_points, truth_points
            )
            for rank, (_, candidate_target, candidate_result, source_name, source_points) in enumerate(
                ranked_results[:3], start=1
            ):
                save_registration_figure(
                    case_dir
                    / f"candidate_{rank:02d}_{source_name}_{candidate_result.method}_{candidate_target.name}.png",
                    source_points,
                    candidate_target.points,
                    candidate_result.transform_initial,
                    candidate_result.transform,
                    ground_truth,
                    title=(
                        f"Unsupervised rank {rank} | {source_name} | {candidate_result.method} | "
                        f"{candidate_target.name} | det={candidate_result.chirality:+d}"
                    ),
                )
        row = {
            "case_id": record.case_id,
            "jaw": record.jaw,
            "target_mode": args.target_mode,
            "target": best_target.name,
            "source_variant": best_source_name,
            "method": best.method,
            "chirality": best.chirality,
            "chirality_mode": args.chirality_mode,
            "selection_score_mm": selection_metrics[id(best)]["score"],
            "rank_score_mm": selection_metrics[id(best)]["rank_score"],
            "prior_angle_deg": selection_metrics[id(best)]["prior_angle_deg"],
            "fit_score_mm": best.score,
            "fit_median_mm": best.median_distance,
            "fit_p90_mm": best.p90_distance,
            "overlap_2mm": best.overlap_2mm,
            "oracle_mean_tre_mm": oracle_row.get("mean_tre_mm", ""),
            "oracle_unsupervised_rank": oracle_row["unsupervised_rank"],
            "oracle_method": oracle_row["method"],
            "oracle_target": oracle_row["target"],
            **metrics,
        }
        rows.append(row)
        print(
            f"  best={best.method}/{best_target.name} rank={selection_metrics[id(best)]['rank_score']:.3f} "
            f"selection={selection_metrics[id(best)]['score']:.3f} "
            f"source={best.score:.3f} "
            f"TRE={metrics.get('mean_tre_mm', float('nan')):.3f} mm",
            flush=True,
        )

    if rows:
        output_csv = args.output_dir / "summary.csv"
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        numeric = [row["mean_tre_mm"] for row in rows if "mean_tre_mm" in row]
        print(f"Summary: {output_csv}")
        if numeric:
            print(f"Mean TRE: {np.mean(numeric):.3f} mm; median TRE: {np.median(numeric):.3f} mm")
    if failures:
        failure_csv = args.output_dir / "failures.csv"
        with failure_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(failures[0].keys()))
            writer.writeheader()
            writer.writerows(failures)
        print(f"Failures: {failure_csv} ({len(failures)})")
    run_metadata = {
        "split": args.split,
        "target_mode": args.target_mode,
        "case_ids": sorted(wanted),
        "jaws": list(args.jaws),
        "transform_prior": str(args.transform_prior) if args.transform_prior else None,
        "leave_one_cbct_group_out_prior": args.leave_one_cbct_group_out_prior,
        "cbct_hash_cache": str(args.cbct_hash_cache) if args.cbct_hash_cache else None,
        "grouped_prior_audit": {
            case_id: grouped_prior_audit[case_id]
            for case_id in sorted({record.case_id for record in selected})
            if case_id in grouped_prior_audit
        },
        "stable_record_seeds": args.stable_record_seeds,
        "seed": args.seed,
        "completed_jaws": len(rows),
        "resumed_jaws": resumed_records,
        "failed_jaws": len(failures),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
