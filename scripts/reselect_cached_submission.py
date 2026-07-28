#!/usr/bin/env python3
"""Create a submission variant by reselecting already generated crown candidates."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.candidate_learning import load_candidate_groups
from task2reg.deployment_ensemble import (
    rank_ensemble_candidates,
    select_case_ensemble_pair,
)
from task2reg.priors import load_rotation_priors


JAW_FILES = {"upper": "upper_gt.npy", "lower": "lower_gt.npy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reselect cached STSR Task 2 candidates under another deployment policy."
    )
    parser.add_argument("--base-output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--deployment-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    return parser.parse_args()


def validate_transform(transform: np.ndarray, source: str) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"Invalid transform from {source}: shape={transform.shape}")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"Invalid homogeneous row from {source}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-2):
        raise ValueError(f"Non-rigid rotation from {source}")
    determinant = float(np.linalg.det(rotation))
    if abs(abs(determinant) - 1.0) > 2e-2:
        raise ValueError(f"Invalid determinant from {source}: {determinant}")
    return transform


def candidate_runs(case_work_dir: Path) -> list[Path]:
    enhancement = case_work_dir / "global_crown_enhancement"
    augmented_root = enhancement / "crown_augmented"
    runs = (
        sorted(path for path in augmented_root.iterdir() if path.is_dir())
        if augmented_root.is_dir()
        else []
    )
    refinement = enhancement / "crown_refinement_augmented"
    if refinement.is_dir():
        runs.append(refinement)
    return runs


def copy_base_case(base_case: Path, output_case: Path) -> dict[str, np.ndarray]:
    output_case.mkdir(parents=True, exist_ok=True)
    transforms: dict[str, np.ndarray] = {}
    for jaw, file_name in JAW_FILES.items():
        source = base_case / file_name
        if not source.is_file():
            raise FileNotFoundError(source)
        transforms[jaw] = validate_transform(np.load(source, allow_pickle=False), str(source))
        shutil.copy2(source, output_case / file_name)
    return transforms


def ranked_audit(item: dict[str, object]) -> dict[str, object]:
    row = item["row"]
    return {
        "candidate_index": int(item["candidate_index"]),
        "score": float(item["score"]),
        "source_variant": str(row.get("source_variant", "")),
        "target": str(row.get("target", "")),
        "method": str(row.get("method", "")),
        "candidate_run": str(row.get("candidate_run", "")),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    policy = json.loads(args.deployment_policy.read_text(encoding="utf-8"))
    regression = joblib.load(args.model_dir / "regression_ensemble.joblib")
    pairwise = joblib.load(args.model_dir / "pairwise_ensemble.joblib")
    priors = load_rotation_priors(args.model_dir / "rotation_prior_all30.json")
    case_dirs = sorted(path for path in args.base_output_dir.iterdir() if path.is_dir())
    if not case_dirs:
        raise ValueError(f"No base cases found under {args.base_output_dir}")

    audit: list[dict[str, object]] = []
    reselected_jaws = 0
    for index, base_case in enumerate(case_dirs, 1):
        case_id = base_case.name
        output_case = args.output_dir / case_id
        transforms = copy_base_case(base_case, output_case)
        runs = candidate_runs(args.work_dir / case_id)
        case_audit: dict[str, object] = {
            "case_id": case_id,
            "candidate_runs": [str(path) for path in runs],
            "jaws": {},
        }
        if runs:
            groups = load_candidate_groups(runs)
            available_jaws = [jaw for jaw in JAW_FILES if (case_id, jaw) in groups]
            selected: dict[str, dict[str, object]] = {}
            if set(available_jaws) == set(JAW_FILES):
                pair = select_case_ensemble_pair(
                    groups,
                    case_id,
                    priors,
                    regression,
                    pairwise,
                    policy,
                )
                selected = {jaw: pair[jaw] for jaw in JAW_FILES}
                case_audit["joint"] = {
                    "objective": float(pair["objective"]),
                    "relative_angle_deg": float(pair["relative_angle_deg"]),
                    "relative_translation_deviation_mm": float(
                        pair["relative_translation_deviation_mm"]
                    ),
                }
            else:
                for jaw in available_jaws:
                    selected[jaw] = rank_ensemble_candidates(
                        groups[(case_id, jaw)],
                        jaw,
                        priors[jaw],
                        regression,
                        pairwise,
                        policy,
                    )[0]

            for jaw, item in selected.items():
                transform = validate_transform(
                    np.asarray(item["row"]["transform"], dtype=np.float64),
                    f"{case_id}/{jaw}",
                )
                np.save(output_case / JAW_FILES[jaw], transform.astype(np.float64))
                delta = float(np.linalg.norm(transform - transforms[jaw]))
                case_audit["jaws"][jaw] = {**ranked_audit(item), "base_matrix_delta": delta}
                reselected_jaws += 1
        audit.append(case_audit)
        print(
            f"[{index}/{len(case_dirs)}] {case_id}: "
            f"reselected={len(case_audit['jaws'])}",
            flush=True,
        )

    payload = {
        "status": "PASS",
        "policy": str(args.deployment_policy.resolve()),
        "base_output_dir": str(args.base_output_dir.resolve()),
        "work_dir": str(args.work_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "cases": len(case_dirs),
        "reselected_jaws": reselected_jaws,
        "audit": audit,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "audit"}, indent=2))


if __name__ == "__main__":
    main()
