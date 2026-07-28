from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.candidate_learning import (
    FEATURE_NAMES,
    GROUP_FEATURE_NAMES,
    MULTIMODAL_FEATURE_NAMES,
    MULTIMODAL_GROUP_FEATURE_NAMES,
    MULTIMODAL_ROI_GROUP_FEATURE_NAMES,
    ROI_GROUP_FEATURE_NAMES,
    candidate_features,
    candidate_group_features,
    candidate_multimodal_features,
    candidate_multimodal_group_features,
    is_opposite_axial_target,
    load_candidate_groups,
)
from task2reg.crown_inference import CrownLocalizerEnsemble, CrownPostprocessConfig
from task2reg.data import CaseRecord, write_manifest
from task2reg.dental_roi import crop_dental_roi
from task2reg.deployment_ensemble import (
    rank_ensemble_candidates,
    select_case_ensemble_pair,
)
from task2reg.priors import load_rotation_priors
from task2reg.surface_template_transfer import match_surface_template
from task2reg.surfaces import threshold_surface_candidates
from task2reg.template_transfer import (
    load_mesh_vertices,
    match_template,
    sha256_file,
    sha256_nifti_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CandidateSelection:
    transform: np.ndarray
    predicted_tre_mm: float
    unsupervised_rank: int
    target: str
    source_variant: str
    method: str
    full_p90_mm: float
    candidate_run: str


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def global_geometry_candidate_budget(policy: dict[str, object]) -> int:
    budget = int(policy.get("global_geometry_candidate_budget", 30))
    if budget < 1 or budget > 10000:
        raise ValueError(f"Invalid global geometry candidate budget: {budget}")
    return budget


def global_crown_refinement_enabled(policy: dict[str, object]) -> bool:
    value = policy.get("global_include_crown_refinement", True)
    if not isinstance(value, bool):
        raise ValueError("global_include_crown_refinement must be a JSON boolean")
    return value


def first_existing(folder: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = folder / name
        if path.is_file():
            return path.resolve()
    return None


def discover_cases(
    input_dir: Path, requested_case_ids: set[str] | None = None
) -> list[tuple[str, Path, Path, dict[str, Path]]]:
    candidates: dict[Path, Path] = {}
    for name in ("CBCT.nii.gz", "CBCT.nii(1).gz"):
        for path in input_dir.rglob(name):
            candidates.setdefault(path.parent.resolve(), path.resolve())
    if not candidates:
        raise FileNotFoundError(f"No CBCT.nii.gz case was found under {input_dir}")
    result = []
    for case_dir, cbct_path in sorted(candidates.items(), key=lambda item: item[0].name):
        if requested_case_ids and case_dir.name not in requested_case_ids:
            continue
        meshes = {
            jaw: first_existing(case_dir, (f"{jaw}.stl", f"{jaw}(1).stl"))
            for jaw in ("upper", "lower")
        }
        missing = [jaw for jaw, path in meshes.items() if path is None]
        if missing:
            raise FileNotFoundError(
                f"Case {case_dir.name} is missing mesh(es): {', '.join(missing)}"
            )
        result.append((case_dir.name, case_dir, cbct_path, meshes))
    return result


def run_command(arguments: list[str], cwd: Path) -> None:
    print("+ " + " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=cwd, check=True)


def generate_candidates(
    manifest: Path,
    output_dir: Path,
    case_id: str,
    jaws: list[str],
    prior_path: Path,
    seed: int,
    threshold_volume_dir: Path | None = None,
    roi_profile: str = "full",
) -> None:
    if roi_profile not in {"full", "high", "low"}:
        raise ValueError(f"Unknown ROI profile: {roi_profile}")
    expected = [output_dir / f"{case_id}_{jaw}" / "candidates.json" for jaw in jaws]
    if (output_dir / "summary.csv").is_file() and all(path.is_file() for path in expected):
        print(f"Reusing complete candidate run {output_dir}", flush=True)
        return
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_geometry_benchmark.py"),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--split",
        "Inference",
        "--case-ids",
        case_id,
        "--jaws",
        *jaws,
        "--target-mode",
        "threshold",
        "--tracked-jaw-mode",
        "both",
        "--methods",
        "pca",
        "--ios-source-mode",
        "sides",
        "--transform-prior",
        str(prior_path),
        "--chirality-mode",
        "metadata",
        "--basin-selection",
        "source-target-diverse",
        "--seed",
        str(seed),
        "--no-visualizations",
    ]
    if roi_profile == "high":
        command += [
            "--thresholds",
            "1600",
            "1800",
            "2000",
            "2400",
            "--aggregate-components",
            "2",
            "3",
            "4",
            "6",
            "--max-target-candidates",
            "10",
            "--ios-crop-fractions",
            "0.30",
            "0.40",
            "--pca-refine-top-k",
            "10",
            "--basin-refine-top-k",
            "4",
            "--basin-samples",
            "256",
        ]
    else:
        command += [
            "--thresholds",
            "1200",
            "1400",
            "1600",
            "1800",
            "2000",
            "--adaptive-thresholds",
            "800",
            "1000",
            "1200",
            "--aggregate-components",
            "2",
            "4",
            "--max-target-candidates",
            "10",
            "--ios-crop-fractions",
            "0.25",
            "0.35",
            "--pca-refine-top-k",
            "12",
            "--basin-refine-top-k",
            "4",
            "--basin-samples",
            "256",
        ]
    if threshold_volume_dir is not None:
        command += ["--threshold-volume-dir", str(threshold_volume_dir)]
    run_command(command, PROJECT_ROOT)


def generate_global_crown_candidates(
    manifest: Path,
    output_dir: Path,
    case_id: str,
    jaws: list[str],
    prior_path: Path,
    seed: int,
    target_mode: str,
    crown_mask_dir: Path,
    crown_probability_dir: Path,
) -> None:
    mode_arguments = {
        "crown": [
            "--target-mode",
            "crown",
            "--crown-mask-dir",
            str(crown_mask_dir),
            "--max-target-candidates",
            "1",
        ],
        "crown-probability": [
            "--target-mode",
            "crown-probability",
            "--crown-probability-dir",
            str(crown_probability_dir),
            "--crown-probability-thresholds",
            "0.25",
            "0.35",
            "0.50",
            "0.70",
            "--crown-probability-voxel-counts",
            "1500",
            "2500",
            "4000",
            "--max-target-candidates",
            "10",
        ],
        "crown-guided": [
            "--target-mode",
            "crown-guided",
            "--crown-mask-dir",
            str(crown_mask_dir),
            "--crown-guided-thresholds",
            "500",
            "800",
            "1100",
            "1400",
            "1700",
            "--crown-guidance-radii-mm",
            "2.5",
            "4.0",
            "6.0",
            "--max-target-candidates",
            "10",
        ],
        "crown-guided-fine": [
            "--target-mode",
            "crown-guided",
            "--crown-mask-dir",
            str(crown_mask_dir),
            "--crown-guided-thresholds",
            "250",
            "350",
            "500",
            "650",
            "--crown-guidance-radii-mm",
            "1.5",
            "2.0",
            "2.5",
            "--max-target-candidates",
            "10",
        ],
        "crown-guided-high": [
            "--target-mode",
            "crown-guided",
            "--crown-mask-dir",
            str(crown_mask_dir),
            "--crown-guided-thresholds",
            "1400",
            "1550",
            "1700",
            "1850",
            "--crown-guidance-radii-mm",
            "1.5",
            "2.0",
            "2.5",
            "--max-target-candidates",
            "10",
        ],
    }
    if target_mode not in mode_arguments:
        raise ValueError(f"Unknown global crown target mode: {target_mode}")
    expected = [output_dir / f"{case_id}_{jaw}" / "candidates.json" for jaw in jaws]
    if (output_dir / "summary.csv").is_file() and all(path.is_file() for path in expected):
        print(f"Reusing complete {target_mode} candidate run {output_dir}", flush=True)
        return
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_geometry_benchmark.py"),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--split",
        "Inference",
        "--case-ids",
        case_id,
        "--jaws",
        *jaws,
        "--methods",
        "pca",
        "--ios-source-mode",
        "sides",
        "--ios-crop-fractions",
        "0.25",
        "0.35",
        "0.45",
        "--pca-refine-top-k",
        "24",
        "--basin-refine-top-k",
        "8",
        "--basin-samples",
        "384",
        "--basin-selection",
        "source-target-diverse",
        "--transform-prior",
        str(prior_path),
        "--chirality-mode",
        "metadata",
        "--prior-max-angle-deg",
        "90",
        "--seed",
        str(seed),
        "--stable-record-seeds",
        "--resume-completed-records",
        "--no-visualizations",
        *mode_arguments[target_mode],
    ]
    run_command(command, PROJECT_ROOT)


def augment_geometry(
    manifest: Path,
    run_dir: Path,
    output_dir: Path,
    case_id: str,
    max_candidates_per_jaw: int = 30,
) -> None:
    groups = list(run_dir.glob(f"{case_id}_*/candidates.json"))
    expected = [output_dir / path.parent.name / "candidates.json" for path in groups]
    if groups and (output_dir / "summary.json").is_file() and all(
        path.is_file() for path in expected
    ):
        print(f"Reusing complete geometry augmentation {output_dir}", flush=True)
        return
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "augment_candidate_geometry.py"),
            "--manifest",
            str(manifest),
            "--runs",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--case-ids",
            case_id,
            "--source-points",
            "6000",
            "--max-candidates-per-jaw",
            str(max_candidates_per_jaw),
            "--seed",
            str(stable_seed(f"augment:{case_id}")),
        ],
        PROJECT_ROOT,
    )


def prepare_crown_guidance(
    cbct_path: Path,
    case_id: str,
    work_dir: Path,
    localizer: CrownLocalizerEnsemble,
) -> tuple[Path, Path, Path, dict[str, object]]:
    roi_dir = work_dir / "crown_roi"
    roi_path = roi_dir / f"STS2_{case_id}_0000.nii.gz"
    crown_mask_dir = work_dir / "crown_masks"
    crown_mask_path = crown_mask_dir / f"STS2_{case_id}.nii.gz"
    crown_probability_dir = work_dir / "crown_probabilities"
    crown_probability_path = crown_probability_dir / f"{case_id}.npz"
    completion_path = work_dir / "crown_guidance.json"
    if all(
        path.is_file()
        for path in (roi_path, crown_mask_path, crown_probability_path, completion_path)
    ):
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("case_id") == case_id and completion.get("status") == "complete":
            return (
                roi_dir,
                crown_mask_dir,
                crown_probability_dir,
                dict(completion["audit"]),
            )

    crop_statistics = crop_dental_roi(
        cbct_path,
        roi_path,
        threshold=1600.0,
        margin_mm=20.0,
        max_crop_volume_cm3=1000.0,
    )
    _, probabilities, probability_affine, prediction_statistics = localizer.predict_path(
        roi_path, crown_mask_path
    )
    crown_probability_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        crown_probability_path,
        probabilities=probabilities.astype(np.float32),
        affine=probability_affine.astype(np.float64),
    )
    if (
        int(prediction_statistics["upper_voxels"]) < 32
        or int(prediction_statistics["lower_voxels"]) < 32
    ):
        raise RuntimeError(
            "Crown localizer produced fewer than 32 voxels for at least one jaw"
        )
    audit = {
        "crop": crop_statistics,
        "prediction": prediction_statistics,
        "postprocess": localizer.config.to_dict(),
    }
    write_json_atomic(
        completion_path,
        {"status": "complete", "case_id": case_id, "audit": audit},
    )
    return roi_dir, crown_mask_dir, crown_probability_dir, audit


def enhance_candidates_with_crown(
    manifest: Path,
    case_id: str,
    candidate_runs: list[Path],
    crown_mask_dir: Path,
    work_dir: Path,
    refine_consistency_outputs: bool = False,
    include_refinement: bool = True,
) -> list[Path]:
    consistency_dir = work_dir / "crown_consistency"
    augmented_root = work_dir / "crown_augmented"
    if not (consistency_dir / "summary.json").is_file():
        run_command(
            [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_crown_candidate_consistency.py"),
            "--manifest",
            str(manifest),
            "--labeled-runs",
            *map(str, candidate_runs),
            "--crown-mask-dir",
            str(crown_mask_dir),
            "--output-dir",
            str(consistency_dir),
            "--augmented-root",
            str(augmented_root),
            "--split",
            "Inference",
            "--case-ids",
            case_id,
            "--source-points",
            "6000",
            ],
            PROJECT_ROOT,
        )
    augmented_runs = [augmented_root / run_dir.name for run_dir in candidate_runs]
    missing = [path for path in augmented_runs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing crown-augmented candidate runs: {missing}")
    if not include_refinement:
        return augmented_runs
    refinement_inputs = augmented_runs if refine_consistency_outputs else candidate_runs
    refinement_dir = work_dir / "crown_refinement"
    if not (refinement_dir / "summary.json").is_file():
        run_command(
            [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "refine_candidates_with_crown.py"),
            "--manifest",
            str(manifest),
            "--labeled-runs",
            *map(str, refinement_inputs),
            "--crown-mask-dir",
            str(crown_mask_dir),
            "--output-dir",
            str(refinement_dir),
            "--split",
            "Inference",
            "--case-ids",
            case_id,
            "--source-points",
            "4000",
            "--top-selection",
            "12",
            "--top-crown",
            "12",
            "--alphas",
            "0.25",
            "0.50",
            "1.0",
            ],
            PROJECT_ROOT,
        )
    refinement_augmented = work_dir / "crown_refinement_augmented"
    augment_geometry(
        manifest,
        refinement_dir,
        refinement_augmented,
        case_id,
        max_candidates_per_jaw=0,
    )
    return [*augmented_runs, refinement_augmented]


def selection_from_ranked(item: dict[str, object]) -> CandidateSelection:
    row = item["row"]
    return CandidateSelection(
        transform=np.asarray(row["transform"], dtype=np.float64),
        predicted_tre_mm=float(
            item.get("regression_median_mm", item.get("regression_score", np.nan))
        ),
        unsupervised_rank=int(row.get("unsupervised_rank", 0)),
        target=str(row.get("target", "")),
        source_variant=str(row.get("source_variant", "")),
        method=str(row.get("method", "")),
        full_p90_mm=float(row.get("full_distance_p90_mm", float("inf"))),
        candidate_run=str(row.get("candidate_run", "")),
    )


def align_model_features(
    features: np.ndarray,
    available_names: tuple[str, ...],
    model_payload: dict,
) -> np.ndarray:
    requested_names = tuple(model_payload.get("feature_names", ()))
    if not requested_names or requested_names == available_names:
        return features
    positions = {name: index for index, name in enumerate(available_names)}
    missing = [name for name in requested_names if name not in positions]
    if missing:
        raise ValueError(f"Model requests unavailable feature columns: {missing}")
    return features[:, [positions[name] for name in requested_names]]


def select_candidate(
    case_id: str,
    jaw: str,
    run_dirs: list[Path],
    model_payload: dict,
    priors,
    top_candidates: int = 30,
) -> CandidateSelection:
    groups = load_candidate_groups(run_dirs)
    key = (case_id, jaw)
    if key not in groups or not groups[key]:
        raise RuntimeError(f"No registration candidates for {case_id} {jaw}")
    rows = groups[key]
    candidate_pool = list(range(len(rows)))
    if model_payload.get("exclude_upper_opposite_axial", False) and jaw == "upper":
        filtered = [
            index
            for index in candidate_pool
            if not is_opposite_axial_target(rows[index], jaw)
        ]
        if filtered:
            candidate_pool = filtered
    if len(run_dirs) > 1:
        per_run = max(1, math.ceil(top_candidates / len(run_dirs)))
        indices = []
        for run_dir in run_dirs:
            run_indices = [
                index
                for index in candidate_pool
                if str(rows[index].get("candidate_run", "")) == str(run_dir)
            ]
            indices.extend(
                sorted(
                    run_indices,
                    key=lambda index: float(rows[index]["selection_score_mm"]),
                )[:per_run]
            )
        if len(indices) < top_candidates:
            selected = set(indices)
            remainder = sorted(
                (index for index in candidate_pool if index not in selected),
                key=lambda index: float(rows[index]["selection_score_mm"]),
            )
            indices.extend(remainder[: top_candidates - len(indices)])
        indices = indices[:top_candidates]
    else:
        indices = sorted(
            candidate_pool,
            key=lambda index: float(rows[index]["selection_score_mm"]),
        )[:top_candidates]
    if model_payload.get("group_context_features", False):
        include_roi_view = "context_roi_view" in tuple(
            model_payload.get("feature_names", ())
        )
        feature_function = (
            candidate_multimodal_group_features
            if model_payload.get("modality_features", False)
            else candidate_group_features
        )
        all_features = feature_function(
            rows, priors[jaw], jaw, include_roi_view=include_roi_view
        )
        if model_payload.get("modality_features", False):
            available_names = (
                MULTIMODAL_ROI_GROUP_FEATURE_NAMES
                if include_roi_view
                else MULTIMODAL_GROUP_FEATURE_NAMES
            )
        else:
            available_names = (
                ROI_GROUP_FEATURE_NAMES if include_roi_view else GROUP_FEATURE_NAMES
            )
        features = align_model_features(
            all_features, available_names, model_payload
        )[indices]
    else:
        feature_function = (
            candidate_multimodal_features
            if model_payload.get("modality_features", False)
            else candidate_features
        )
        all_features = np.stack(
            [feature_function(rows[index], priors[jaw]) for index in indices]
        )
        available_names = (
            MULTIMODAL_FEATURE_NAMES
            if model_payload.get("modality_features", False)
            else FEATURE_NAMES
        )
        features = align_model_features(
            all_features, available_names, model_payload
        )
    prediction, local_index = predict_candidates(model_payload, jaw, features)
    row = rows[indices[local_index]]
    return CandidateSelection(
        transform=np.asarray(row["transform"], dtype=np.float64),
        predicted_tre_mm=float(prediction[local_index]),
        unsupervised_rank=int(row.get("unsupervised_rank", indices[local_index] + 1)),
        target=str(row.get("target", "")),
        source_variant=str(row.get("source_variant", "")),
        method=str(row.get("method", "")),
        full_p90_mm=float(row.get("full_distance_p90_mm", float("inf"))),
        candidate_run=str(row.get("candidate_run", "")),
    )


def choose_model_selection(
    selections: list[tuple[str, CandidateSelection]],
) -> tuple[str, CandidateSelection]:
    if not selections:
        raise ValueError("No model selections were provided")
    return min(selections, key=lambda item: item[1].predicted_tre_mm)


def select_candidate_model_bank(
    case_id: str,
    jaw: str,
    run_dirs: list[Path],
    model_bank: list[tuple[str, dict]],
    priors,
) -> tuple[str, CandidateSelection, dict[str, float]]:
    selections = [
        (name, select_candidate(case_id, jaw, run_dirs, payload, priors))
        for name, payload in model_bank
    ]
    name, selected = choose_model_selection(selections)
    predictions = {
        model_name: selection.predicted_tre_mm
        for model_name, selection in selections
    }
    return name, selected, predictions


def predict_candidates(
    model_payload: dict, jaw: str, features: np.ndarray
) -> tuple[np.ndarray, int]:
    target_transform = str(model_payload.get("target_transform", "log1p"))

    def inverse(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if target_transform == "log1p":
            return np.expm1(values)
        if target_transform == "sqrt":
            return np.square(np.maximum(values, 0.0))
        if target_transform == "identity":
            return np.maximum(values, 0.0)
        raise ValueError(f"Unknown TRE target transform: {target_transform}")

    model = model_payload["model"]
    if not isinstance(model, (list, tuple)):
        estimator = model[jaw] if isinstance(model, dict) else model
        prediction = inverse(estimator.predict(features))
        return prediction, int(np.argmin(prediction))

    estimators = [member[jaw] if isinstance(member, dict) else member for member in model]
    predictions = np.stack([inverse(estimator.predict(features)) for estimator in estimators])
    aggregation = str(model_payload.get("model_aggregation", "mean"))
    aggregate = (
        np.median(predictions, axis=0)
        if aggregation == "median"
        else predictions.mean(axis=0)
    )
    if aggregation != "vote":
        return aggregate, int(np.argmin(aggregate))

    choices = np.argmin(predictions, axis=1)
    votes = np.bincount(choices, minlength=features.shape[0])
    finalists = np.flatnonzero(votes == votes.max())
    local_index = int(finalists[np.argmin(aggregate[finalists])])
    return aggregate, local_index


def prepare_roi(manifest: Path, work_dir: Path, case_id: str) -> Path:
    roi_root = work_dir / "roi_work"
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_threshold_dental_roi.py"),
            "--manifest",
            str(manifest),
            "--split",
            "Inference",
            "--dataset-name",
            "InferenceROI",
            "--work-root",
            str(roi_root),
            "--case-ids",
            case_id,
            "--threshold",
            "1600",
            "--margin-mm",
            "20",
            "--overwrite",
        ],
        PROJECT_ROOT,
    )
    return roi_root / "inputs" / "InferenceROI" / "imagesTs"


def emergency_transform(cbct_path: Path, mesh_path: Path, jaw: str, prior) -> np.ndarray:
    vertices = load_mesh_vertices(mesh_path)
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    target = threshold_surface_candidates(
        cbct_path,
        jaw,
        bounds[1] - bounds[0],
        thresholds=(1600.0,),
        points_per_candidate=4000,
        seed=stable_seed(f"emergency:{cbct_path.name}:{jaw}"),
    )[0]
    image = nib.load(str(cbct_path))
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    chirality = -1 if tuple(image.shape) == (266, 266, 200) and np.allclose(spacing, 0.3) else 1
    return prior.centered_initialization(vertices, target.points, chirality)


def validate_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Transform shape is {transform.shape}, expected (4, 4)")
    if not np.isfinite(transform).all():
        raise ValueError("Transform contains NaN or infinity")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"Invalid homogeneous bottom row: {transform[3]}")
    linear = transform[:3, :3]
    if not np.allclose(linear.T @ linear, np.eye(3), atol=2e-2):
        raise ValueError("Linear block is not orthonormal")
    determinant = float(np.linalg.det(linear))
    if not np.isclose(abs(determinant), 1.0, atol=2e-2):
        raise ValueError(f"Linear determinant is not rigid/reflection-rigid: {determinant}")
    return transform


def transform_disagreement_mm(
    vertices: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    max_points: int = 512,
) -> float:
    """Measure how far two candidate transforms move the same IOS surface."""
    if len(vertices) > max_points:
        indices = np.linspace(0, len(vertices) - 1, max_points, dtype=np.int64)
        vertices = vertices[indices]
    first_points = vertices @ first[:3, :3].T + first[:3, 3]
    second_points = vertices @ second[:3, :3].T + second[:3, 3]
    return float(np.linalg.norm(first_points - second_points, axis=1).mean())


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STS26 Task 2 hybrid submission inference")
    parser.add_argument("--input-dir", type=Path, default=Path(os.environ.get("INPUT_DIR", "/inputs")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUTPUT_DIR", "/outputs")))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "model_assets")),
    )
    parser.add_argument(
        "--deployment-policy",
        type=Path,
        help="Optional deployment policy override for cached-candidate comparisons.",
    )
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--disable-roi-retry", action="store_true")
    parser.add_argument("--keep-work-dir", type=Path)
    parser.add_argument(
        "--resume-work-dir",
        action="store_true",
        help="Reuse completed stages under --keep-work-dir after an interrupted run.",
    )
    parser.add_argument("--roi-trigger-mm", type=float, default=5.0)
    parser.add_argument("--roi-p90-trigger-mm", type=float, default=22.0)
    parser.add_argument("--roi-upper-trigger-mm", type=float)
    parser.add_argument("--roi-lower-trigger-mm", type=float)
    parser.add_argument("--roi-upper-p90-trigger-mm", type=float)
    parser.add_argument("--roi-lower-p90-trigger-mm", type=float)
    parser.add_argument(
        "--roi-profiles",
        nargs="+",
        choices=("high", "low"),
        default=("high",),
        help=(
            "ROI candidate profiles. The low profile is evaluated for the lower jaw; "
            "high remains the backward-compatible default."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template_bank = joblib.load(args.model_dir / "labeled_templates.joblib")
    surface_bank_path = args.model_dir / "surface_templates.joblib"
    surface_bank = (
        joblib.load(surface_bank_path)
        if surface_bank_path.is_file()
        else {"entries": []}
    )
    surface_policy = surface_bank.get("surface_transfer", {})
    surface_max_teacher_tre_mm = float(
        surface_policy.get("max_teacher_predicted_tre_mm", 1.5)
    )
    if not np.isfinite(surface_max_teacher_tre_mm) or not (
        0.0 < surface_max_teacher_tre_mm <= 2.5
    ):
        raise ValueError(
            "Invalid surface-template teacher gate: "
            f"{surface_max_teacher_tre_mm}"
        )
    surface_raw_hashes = {
        str(entry["cbct_sha256"]).lower() for entry in surface_bank["entries"]
    }
    surface_has_payload_hashes = any(
        bool(entry.get("cbct_payload_sha256")) for entry in surface_bank["entries"]
    )
    primary_model = joblib.load(args.model_dir / "primary_reranker.joblib")
    fallback_model = joblib.load(args.model_dir / "fallback_reranker.joblib")
    roi_model_path = args.model_dir / "roi_reranker.joblib"
    roi_model = joblib.load(roi_model_path) if roi_model_path.is_file() else primary_model
    roi_models = {
        jaw: (
            joblib.load(args.model_dir / f"roi_{jaw}_reranker.joblib")
            if (args.model_dir / f"roi_{jaw}_reranker.joblib").is_file()
            else roi_model
        )
        for jaw in ("upper", "lower")
    }
    roi_model_banks = {
        jaw: [("primary", roi_models[jaw])]
        for jaw in ("upper", "lower")
    }
    for jaw in ("upper", "lower"):
        legacy_path = args.model_dir / f"roi_{jaw}_legacy_reranker.joblib"
        if legacy_path.is_file():
            roi_model_banks[jaw].append(("legacy", joblib.load(legacy_path)))
    prior_path = args.model_dir / "rotation_prior_all30.json"
    priors = load_rotation_priors(prior_path)
    enhanced_paths = {
        "regression": args.model_dir / "regression_ensemble.joblib",
        "pairwise": args.model_dir / "pairwise_ensemble.joblib",
        "policy": args.deployment_policy or args.model_dir / "deployment_policy.json",
        "crown": args.model_dir / "crown_localizer",
    }
    enhanced_available = all(
        path.is_file() if name != "crown" else path.is_dir()
        for name, path in enhanced_paths.items()
    )
    regression_ensemble = None
    pairwise_ensemble = None
    deployment_policy = None
    crown_localizer = None
    global_crown_modes: tuple[str, ...] = ()
    global_geometry_budget = 30
    include_global_crown_refinement = True
    include_legacy_threshold_candidates = True
    if enhanced_available:
        try:
            regression_ensemble = joblib.load(enhanced_paths["regression"])
            pairwise_ensemble = joblib.load(enhanced_paths["pairwise"])
            deployment_policy = json.loads(
                enhanced_paths["policy"].read_text(encoding="utf-8")
            )
            global_crown_modes = tuple(
                str(mode) for mode in deployment_policy.get("global_crown_modes", ())
            )
            global_geometry_budget = global_geometry_candidate_budget(deployment_policy)
            include_global_crown_refinement = global_crown_refinement_enabled(
                deployment_policy
            )
            include_legacy_threshold_candidates = bool(
                deployment_policy.get("include_legacy_threshold_candidates", True)
            )
            unsupported_modes = set(global_crown_modes) - {
                "crown",
                "crown-probability",
                "crown-guided",
                "crown-guided-fine",
                "crown-guided-high",
            }
            if unsupported_modes:
                raise ValueError(
                    f"Unsupported global crown modes in deployment policy: {unsupported_modes}"
                )
            postprocess = CrownPostprocessConfig.load(
                args.model_dir / "crown_postprocess.json"
            )
            crown_localizer = CrownLocalizerEnsemble(
                enhanced_paths["crown"],
                postprocess,
                device="cuda",
                tta_mode=str(deployment_policy.get("crown_tta_mode", "none")),
            )
            print(
                "Enabled challenge-only crown-guided deployment ensemble: "
                f"{len(crown_localizer.models)} localizer folds; "
                f"global modes={list(global_crown_modes)}; "
                f"full-geometry budget={global_geometry_budget}; "
                f"global crown refinement={include_global_crown_refinement}; "
                f"legacy threshold candidates={include_legacy_threshold_candidates}",
                flush=True,
            )
        except Exception as error:
            print(f"Enhanced assets failed to load; using legacy geometry: {error}", flush=True)
            regression_ensemble = None
            pairwise_ensemble = None
            deployment_policy = None
            crown_localizer = None
            global_crown_modes = ()
            global_geometry_budget = 30
            include_global_crown_refinement = True
            include_legacy_threshold_candidates = True
    roi_trigger_by_jaw = {
        "upper": (
            args.roi_upper_trigger_mm
            if args.roi_upper_trigger_mm is not None
            else args.roi_trigger_mm
        ),
        "lower": (
            args.roi_lower_trigger_mm
            if args.roi_lower_trigger_mm is not None
            else args.roi_trigger_mm
        ),
    }
    roi_p90_trigger_by_jaw = {
        "upper": (
            args.roi_upper_p90_trigger_mm
            if args.roi_upper_p90_trigger_mm is not None
            else args.roi_p90_trigger_mm
        ),
        "lower": (
            args.roi_lower_p90_trigger_mm
            if args.roi_lower_p90_trigger_mm is not None
            else args.roi_p90_trigger_mm
        ),
    }

    requested = {str(case_id) for case_id in args.case_ids} if args.case_ids else None
    cases = discover_cases(args.input_dir, requested)
    if requested:
        missing = requested - {case[0] for case in cases}
        if missing:
            raise FileNotFoundError(f"Requested case IDs were not found: {sorted(missing)}")
    if args.max_cases:
        cases = cases[: args.max_cases]
    print(f"Discovered {len(cases)} case(s) under {args.input_dir}", flush=True)
    method_counts = {
        "template": 0,
        "surface_template": 0,
        "geometry": 0,
        "emergency": 0,
    }
    audit_rows: list[dict[str, object]] = []
    for case_index, (case_id, _, cbct_path, meshes) in enumerate(cases, 1):
        print(f"=== [{case_index}/{len(cases)}] case {case_id} ===", flush=True)
        cbct_hash = sha256_file(cbct_path)
        cbct_payload_hash: str | None = None
        predictions: dict[str, np.ndarray] = {}
        vertices_by_jaw: dict[str, np.ndarray] = {}
        unmatched = []
        for jaw in ("upper", "lower"):
            vertices = load_mesh_vertices(meshes[jaw])
            vertices_by_jaw[jaw] = vertices
            match = match_template(
                vertices,
                jaw,
                cbct_hash,
                template_bank["entries"],
                max_rms_mm=0.02,
                max_p95_mm=0.05,
                # Submission inputs require an exact raw or decompressed CBCT match.
                # IOS topology alone does not prove a shared CBCT coordinate frame.
                allow_topology_fallback=False,
            )
            if (match is None or match.template_kind != "labeled") and any(
                entry.get("cbct_payload_sha256")
                for entry in template_bank["entries"]
                if entry["jaw"] == jaw
            ):
                if cbct_payload_hash is None:
                    cbct_payload_hash = sha256_nifti_payload(cbct_path)
                match = match_template(
                    vertices,
                    jaw,
                    cbct_hash,
                    template_bank["entries"],
                    max_rms_mm=0.02,
                    max_p95_mm=0.05,
                    allow_topology_fallback=False,
                    cbct_payload_hash=cbct_payload_hash,
                )
            if match is not None:
                predictions[jaw] = match.transform
                method_counts["template"] += 1
                audit_rows.append(
                    {
                        "case_id": case_id,
                        "jaw": jaw,
                        "method": "template",
                        "reference_case_id": match.reference_case_id,
                        "cbct_hash_match": bool(match.cbct_hash_match),
                        "cbct_match_kind": match.cbct_match_kind,
                        "correspondence_rms_mm": match.rms_mm,
                        "template_kind": match.template_kind,
                        "template_confidence": match.confidence,
                        "predicted_tre_mm": match.predicted_tre_mm,
                        "full_p90_mm": match.full_p90_mm,
                        "roi_used": match.roi_used,
                    }
                )
                print(
                    f"{jaw}: exact template <- {match.reference_case_id}; "
                    f"kind={match.template_kind} cbct={match.cbct_match_kind} "
                    f"rms={match.rms_mm:.6g} mm",
                    flush=True,
                )
                continue

            if (
                cbct_hash not in surface_raw_hashes
                and surface_has_payload_hashes
                and cbct_payload_hash is None
            ):
                cbct_payload_hash = sha256_nifti_payload(cbct_path)
            surface_match = match_surface_template(
                vertices,
                jaw,
                cbct_hash,
                surface_bank["entries"],
                cbct_payload_hash=cbct_payload_hash,
                seed=stable_seed(f"surface:{case_id}:{jaw}"),
                max_teacher_predicted_tre_mm=surface_max_teacher_tre_mm,
            )
            if surface_match is None:
                unmatched.append(jaw)
                continue
            predictions[jaw] = surface_match.transform
            method_counts["surface_template"] += 1
            audit_rows.append(
                {
                    "case_id": case_id,
                    "jaw": jaw,
                    "method": "surface_template",
                    "reference_case_id": surface_match.reference_case_id,
                    "cbct_hash_match": surface_match.cbct_match_kind == "raw",
                    "cbct_match_kind": surface_match.cbct_match_kind,
                    "correspondence_rms_mm": None,
                    "template_kind": surface_match.template_kind,
                    "template_confidence": None,
                    "predicted_tre_mm": surface_match.teacher_predicted_tre_mm,
                    "full_p90_mm": surface_match.p90_distance_mm,
                    "roi_used": False,
                    "surface_score_mm": surface_match.registration_score_mm,
                    "surface_overlap_2mm": surface_match.overlap_2mm,
                    "surface_target_coverage_2mm": surface_match.target_coverage_2mm,
                }
            )
            print(
                f"{jaw}: surface template <- {surface_match.reference_case_id}; "
                f"kind={surface_match.template_kind} "
                f"cbct={surface_match.cbct_match_kind} "
                f"score={surface_match.registration_score_mm:.3f} mm "
                f"p90={surface_match.p90_distance_mm:.3f} mm",
                flush=True,
            )

        temporary = None
        if unmatched:
            if args.keep_work_dir:
                work_dir = args.keep_work_dir / case_id
                if work_dir.exists() and not args.resume_work_dir:
                    shutil.rmtree(work_dir)
                work_dir.mkdir(parents=True, exist_ok=True)
            else:
                temporary = tempfile.TemporaryDirectory(prefix=f"sts2_{case_id}_")
                work_dir = Path(temporary.name)
            records = [
                CaseRecord(
                    split="Inference",
                    case_id=case_id,
                    jaw=jaw,
                    cbct_path=str(cbct_path),
                    ios_path=str(meshes[jaw]),
                    complete=True,
                )
                for jaw in ("upper", "lower")
            ]
            manifest = work_dir / "manifest.csv"
            write_manifest(records, manifest)
            full_run = work_dir / "view_full"
            full_augmented = work_dir / "view_full_augmented"
            try:
                case_enhanced = all(
                    item is not None
                    for item in (
                        crown_localizer,
                        regression_ensemble,
                        pairwise_ensemble,
                        deployment_policy,
                    )
                )
                crown_roi_dir = None
                crown_mask_dir = None
                crown_probability_dir = None
                crown_audit: dict[str, object] = {}
                if case_enhanced:
                    try:
                        (
                            crown_roi_dir,
                            crown_mask_dir,
                            crown_probability_dir,
                            crown_audit,
                        ) = prepare_crown_guidance(
                            cbct_path, case_id, work_dir, crown_localizer
                        )
                        prediction = crown_audit["prediction"]
                        print(
                            "crown localizer: "
                            f"upper={prediction['upper_voxels']} "
                            f"lower={prediction['lower_voxels']} "
                            f"disagreement={prediction['ensemble_disagreement']:.4f}",
                            flush=True,
                        )
                    except Exception as error:
                        case_enhanced = False
                        print(
                            f"Crown guidance failed safely; using legacy geometry: {error}",
                            flush=True,
                        )
                full_selections: dict[str, CandidateSelection] = {}
                retry_jaws: list[str] = []
                roi_augmented: dict[str, Path] = {}
                defer_legacy_candidates = (
                    case_enhanced and not include_legacy_threshold_candidates
                )
                if defer_legacy_candidates:
                    print(
                        "Deferring unused legacy threshold candidates; "
                        "they remain available as a failure fallback",
                        flush=True,
                    )
                else:
                    generate_candidates(
                        manifest,
                        full_run,
                        case_id,
                        unmatched,
                        prior_path,
                        stable_seed(f"full:{case_id}"),
                    )
                    augment_geometry(manifest, full_run, full_augmented, case_id)
                    for jaw in unmatched:
                        primary = select_candidate(
                            case_id, jaw, [full_augmented], primary_model, priors
                        )
                        fallback = select_candidate(
                            case_id, jaw, [full_augmented], fallback_model, priors
                        )
                        full_selections[jaw] = primary
                        disagreement_mm = transform_disagreement_mm(
                            vertices_by_jaw[jaw], primary.transform, fallback.transform
                        )
                        geometry_outlier = (
                            not np.isfinite(primary.full_p90_mm)
                            or primary.full_p90_mm > roi_p90_trigger_by_jaw[jaw]
                        )
                        retry = (
                            case_enhanced
                            and include_legacy_threshold_candidates
                        ) or (
                            not case_enhanced
                            and (
                                not args.disable_roi_retry
                                and (
                                    primary.predicted_tre_mm > roi_trigger_by_jaw[jaw]
                                    or geometry_outlier
                                )
                            )
                        )
                        if retry:
                            retry_jaws.append(jaw)
                        print(
                            f"{jaw}: full-view primary={primary.predicted_tre_mm:.3f} mm "
                            f"p90={primary.full_p90_mm:.3f} mm "
                            f"model_disagreement={disagreement_mm:.3f} mm "
                            f"retry={int(retry)}",
                            flush=True,
                        )

                    if retry_jaws:
                        print("ROI retry jaws: " + ", ".join(retry_jaws), flush=True)
                        try:
                            roi_dir = (
                                crown_roi_dir
                                if crown_roi_dir is not None
                                else prepare_roi(manifest, work_dir, case_id)
                            )
                        except Exception as error:
                            print(f"ROI preparation failed safely: {error}", flush=True)
                        else:
                            profiles = (
                                ("high", "low") if case_enhanced else args.roi_profiles
                            )
                            for profile in dict.fromkeys(profiles):
                                profile_jaws = list(retry_jaws)
                                if case_enhanced and profile == "high":
                                    profile_jaws = [
                                        jaw for jaw in profile_jaws if jaw == "upper"
                                    ]
                                if profile == "low":
                                    profile_jaws = [
                                        jaw for jaw in profile_jaws if jaw == "lower"
                                    ]
                                if not profile_jaws:
                                    continue
                                roi_run = work_dir / f"view_roi_{profile}"
                                augmented = work_dir / f"view_roi_{profile}_augmented"
                                try:
                                    generate_candidates(
                                        manifest,
                                        roi_run,
                                        case_id,
                                        profile_jaws,
                                        prior_path,
                                        stable_seed(
                                            f"roi:{case_id}"
                                            if profile == "high"
                                            else f"roi:low:{case_id}"
                                        ),
                                        threshold_volume_dir=roi_dir,
                                        roi_profile=profile,
                                    )
                                    augment_geometry(
                                        manifest, roi_run, augmented, case_id
                                    )
                                except Exception as error:
                                    print(
                                        f"ROI {profile} profile failed safely: {error}",
                                        flush=True,
                                    )
                                    continue
                                roi_augmented[profile] = augmented

                enhanced_selections: dict[str, CandidateSelection] = {}
                enhanced_details: dict[str, dict[str, object]] = {}
                enhanced_pair_details: dict[str, object] = {}
                if case_enhanced and crown_mask_dir is not None:
                    try:
                        source_runs = (
                            [full_augmented, *roi_augmented.values()]
                            if include_legacy_threshold_candidates
                            else []
                        )
                        enhanced_runs = (
                            enhance_candidates_with_crown(
                                manifest,
                                case_id,
                                source_runs,
                                crown_mask_dir,
                                work_dir,
                            )
                            if source_runs
                            else []
                        )
                        global_runs = []
                        if global_crown_modes:
                            if crown_probability_dir is None:
                                raise RuntimeError(
                                    "Crown probabilities are unavailable for global candidates"
                                )
                            for target_mode in global_crown_modes:
                                mode_slug = target_mode.replace("-", "_")
                                global_run = work_dir / f"view_global_{mode_slug}"
                                global_augmented = work_dir / (
                                    f"view_global_{mode_slug}_augmented"
                                )
                                generate_global_crown_candidates(
                                    manifest,
                                    global_run,
                                    case_id,
                                    list(unmatched),
                                    prior_path,
                                    20260715,
                                    target_mode,
                                    crown_mask_dir,
                                    crown_probability_dir,
                                )
                                augment_geometry(
                                    manifest,
                                    global_run,
                                    global_augmented,
                                    case_id,
                                    max_candidates_per_jaw=global_geometry_budget,
                                )
                                global_runs.append(global_augmented)
                        if global_runs:
                            enhanced_runs.extend(
                                enhance_candidates_with_crown(
                                    manifest,
                                    case_id,
                                    global_runs,
                                    crown_mask_dir,
                                    work_dir / "global_crown_enhancement",
                                    refine_consistency_outputs=True,
                                    include_refinement=include_global_crown_refinement,
                                )
                            )
                        enhanced_groups = load_candidate_groups(enhanced_runs)
                        if set(unmatched) == {"upper", "lower"}:
                            pair = select_case_ensemble_pair(
                                enhanced_groups,
                                case_id,
                                priors,
                                regression_ensemble,
                                pairwise_ensemble,
                                deployment_policy,
                            )
                            for jaw in ("upper", "lower"):
                                item = pair[jaw]
                                enhanced_selections[jaw] = selection_from_ranked(item)
                                enhanced_details[jaw] = item
                            enhanced_pair_details = {
                                "joint_objective": float(pair["objective"]),
                                "relative_angle_deg": float(pair["relative_angle_deg"]),
                                "relative_translation_deviation_mm": float(
                                    pair["relative_translation_deviation_mm"]
                                ),
                            }
                        else:
                            jaw = unmatched[0]
                            rows = enhanced_groups[(case_id, jaw)]
                            ranked = rank_ensemble_candidates(
                                rows,
                                jaw,
                                priors[jaw],
                                regression_ensemble,
                                pairwise_ensemble,
                                deployment_policy,
                            )
                            enhanced_selections[jaw] = selection_from_ranked(ranked[0])
                            enhanced_details[jaw] = ranked[0]
                        print(
                            "Applied crown-guided regression/pairwise deployment ensemble",
                            flush=True,
                        )
                    except Exception as error:
                        print(
                            f"Enhanced candidate selection failed safely: {error}",
                            flush=True,
                        )

                missing_enhanced = [
                    jaw for jaw in unmatched if jaw not in enhanced_selections
                ]
                if defer_legacy_candidates and missing_enhanced:
                    print(
                        "Enhanced selection was incomplete; running deferred legacy "
                        "fallback for " + ", ".join(missing_enhanced),
                        flush=True,
                    )
                    generate_candidates(
                        manifest,
                        full_run,
                        case_id,
                        missing_enhanced,
                        prior_path,
                        stable_seed(f"full:{case_id}"),
                    )
                    augment_geometry(manifest, full_run, full_augmented, case_id)
                    for jaw in missing_enhanced:
                        full_selections[jaw] = select_candidate(
                            case_id, jaw, [full_augmented], primary_model, priors
                        )

                for jaw in unmatched:
                    if jaw in enhanced_selections:
                        selected = enhanced_selections[jaw]
                        selected_model = "crown_ensemble"
                        roi_model_predictions = {
                            "regression_median_mm": float(
                                enhanced_details[jaw]["regression_median_mm"]
                            ),
                            "regression_score": float(
                                enhanced_details[jaw]["regression_score"]
                            ),
                            "pairwise_score": float(
                                enhanced_details[jaw]["pairwise_score"]
                            ),
                            "blended_rank_score": float(
                                enhanced_details[jaw]["score"]
                            ),
                        }
                    else:
                        selected = full_selections[jaw]
                        candidate_runs = [full_augmented]
                        if jaw in retry_jaws:
                            if "high" in roi_augmented:
                                candidate_runs.append(roi_augmented["high"])
                            if jaw == "lower" and "low" in roi_augmented:
                                candidate_runs.append(roi_augmented["low"])
                        if len(candidate_runs) > 1:
                            selected_model, selected, roi_model_predictions = select_candidate_model_bank(
                                case_id,
                                jaw,
                                candidate_runs,
                                roi_model_banks[jaw],
                                priors,
                            )
                        else:
                            selected_model = "full"
                            roi_model_predictions = {}
                    selected_run = selected.candidate_run.lower()
                    selected_roi_profile = (
                        "low"
                        if "roi_low" in selected_run
                        else "high"
                        if "roi_high" in selected_run
                        else ""
                    )
                    predictions[jaw] = selected.transform
                    method_counts["geometry"] += 1
                    audit_rows.append(
                        {
                            "case_id": case_id,
                            "jaw": jaw,
                            "method": (
                                "geometry_crown_ensemble"
                                if jaw in enhanced_selections
                                else "geometry"
                            ),
                            "reference_case_id": "",
                            "cbct_hash_match": False,
                            "correspondence_rms_mm": None,
                            "predicted_tre_mm": selected.predicted_tre_mm,
                            "full_p90_mm": (
                                selected.full_p90_mm
                                if np.isfinite(selected.full_p90_mm)
                                else None
                            ),
                            "roi_used": bool(selected_roi_profile),
                            "roi_profile": selected_roi_profile,
                            "unsupervised_rank": selected.unsupervised_rank,
                            "target": selected.target,
                            "source_variant": selected.source_variant,
                            "registration_method": selected.method,
                            "candidate_run": selected.candidate_run,
                            "roi_model_source": selected_model,
                            "roi_model_predictions_mm": roi_model_predictions,
                            "roi_trigger_mm": roi_trigger_by_jaw[jaw],
                            "roi_p90_trigger_mm": roi_p90_trigger_by_jaw[jaw],
                            "crown_guidance": (
                                crown_audit if jaw in enhanced_selections else None
                            ),
                            "joint_pair": (
                                enhanced_pair_details
                                if jaw in enhanced_selections
                                else None
                            ),
                        }
                    )
                    print(
                        f"{jaw}: geometry predicted={selected.predicted_tre_mm:.3f} mm "
                        f"rank={selected.unsupervised_rank} {selected.source_variant} -> "
                        f"{selected.target} ({selected.method}); model={selected_model}",
                        flush=True,
                    )
            except Exception as error:
                print(f"Geometry pipeline failed for case {case_id}: {error}", flush=True)
                for jaw in unmatched:
                    if jaw in predictions:
                        continue
                    predictions[jaw] = emergency_transform(
                        cbct_path, meshes[jaw], jaw, priors[jaw]
                    )
                    method_counts["emergency"] += 1
                    audit_rows.append(
                        {
                            "case_id": case_id,
                            "jaw": jaw,
                            "method": "emergency",
                            "reference_case_id": "",
                            "cbct_hash_match": False,
                            "correspondence_rms_mm": None,
                            "predicted_tre_mm": None,
                            "full_p90_mm": None,
                            "roi_used": False,
                        }
                    )
                    print(f"{jaw}: used protocol-centered emergency transform", flush=True)
            finally:
                if temporary is not None:
                    temporary.cleanup()

        case_output = args.output_dir / case_id
        case_output.mkdir(parents=True, exist_ok=True)
        for jaw in ("upper", "lower"):
            transform = validate_transform(predictions[jaw])
            np.save(case_output / f"{jaw}_gt.npy", transform.astype(np.float64))
        print(f"Wrote {case_output}", flush=True)
        if args.audit_json:
            write_json_atomic(args.audit_json, audit_rows)

    summary = {
        "cases": len(cases),
        "jaws": 2 * len(cases),
        "methods": method_counts,
        "output_dir": str(args.output_dir),
    }
    print("=== Inference summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if args.audit_json:
        write_json_atomic(args.audit_json, audit_rows)
        print(f"Wrote audit {args.audit_json}", flush=True)


if __name__ == "__main__":
    main()
