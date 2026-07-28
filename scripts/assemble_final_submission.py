from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE_ASSETS = (
    "fallback_reranker.joblib",
    "primary_reranker.joblib",
    "rotation_prior_all30.json",
)
OPTIONAL_LEGACY_ASSETS = (
    "roi_reranker.joblib",
    "roi_upper_reranker.joblib",
    "roi_lower_reranker.joblib",
    "roi_upper_legacy_reranker.joblib",
    "roi_lower_legacy_reranker.joblib",
)
TEMPLATE_ASSETS = ("labeled_templates.joblib", "surface_templates.joblib")
ENHANCED_ASSETS = (
    "regression_ensemble.joblib",
    "pairwise_ensemble.joblib",
    "deployment_policy.json",
)
SOURCE_PROVENANCE_ASSETS = (
    "model_asset_manifest.json",
    "model_asset_manifest.sha256",
    "selection_provenance.json",
    "strict_fair_selection.json",
    "strict_official_mode_selection.json",
)
RUNTIME_SCRIPTS = (
    "augment_candidate_geometry.py",
    "evaluate_crown_candidate_consistency.py",
    "prepare_threshold_dental_roi.py",
    "refine_candidates_with_crown.py",
    "run_geometry_benchmark.py",
    "run_submission_inference.py",
)
SCAFFOLD_FILES = (
    "Dockerfile",
    "predict.sh",
    "requirements.txt",
    "scripts/gzip_file.py",
    "scripts/validate_outputs.py",
    "scripts/verify_assets.py",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Atomically assemble the selected STSR2026 Task 2 Docker assets."
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--legacy-assets", type=Path, required=True)
    parser.add_argument("--template-assets", type=Path, required=True)
    parser.add_argument("--enhanced-assets", type=Path, required=True)
    parser.add_argument("--crown-supervised", type=Path, required=True)
    parser.add_argument("--crown-semisupervised", type=Path, required=True)
    parser.add_argument("--fusion-summary", type=Path, required=True)
    parser.add_argument("--fusion-selection-evidence", type=Path)
    parser.add_argument("--supervised-weight", type=float)
    parser.add_argument("--fusion-mode", choices=("arithmetic", "geometric"))
    parser.add_argument("--crown-tta-mode", choices=("none", "d4"))
    parser.add_argument("--postprocess-config", type=Path)
    parser.add_argument("--expected-members", type=int, default=5)
    parser.add_argument("--expected-labeled-cases", type=int, default=30)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.name


def require_file(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def safe_remove_tree(path: Path, allowed_parent: Path) -> None:
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    if resolved == parent or parent not in resolved.parents:
        raise RuntimeError(f"Refusing to remove directory outside {parent}: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def validate_final_training(
    root: Path, expected_members: int, expected_labeled_cases: int
) -> dict[str, object]:
    summary_path = require_file(root, "training_summary.json")
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != expected_members:
        raise RuntimeError(
            f"Expected {expected_members} final members under {root}, found {len(rows)}"
        )
    expected_indices = list(range(expected_members))
    members = sorted(int(row["member"]) for row in rows)
    if members != expected_indices:
        raise RuntimeError(f"Unexpected member indices under {root}: {members}")
    for row in rows:
        if int(row.get("labeled_cases", -1)) != expected_labeled_cases:
            raise RuntimeError(
                f"Member {row.get('member')} under {root} was not trained on "
                f"all {expected_labeled_cases} labeled cases"
            )
        if int(row.get("epochs", 0)) < 1:
            raise RuntimeError(f"Invalid epoch count in {summary_path}: {row}")
        require_file(root / f"fold_{int(row['member'])}", "best.pt")
    return {
        "path": str(root.resolve()),
        "summary_sha256": sha256(summary_path),
        "members": rows,
    }


def select_fusion(summary_path: Path, override: float | None) -> tuple[float, dict]:
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    best = payload.get("best", payload)
    if not isinstance(best, dict):
        raise RuntimeError(f"Fusion summary has no usable best record: {summary_path}")
    best = dict(best)
    if "fusion_mode" not in best:
        method = str(best.get("method", "")).lower()
        best["fusion_mode"] = "geometric" if "geometric" in method else "arithmetic"
    weight = float(best.get("supervised_weight")) if override is None else override
    if not 0.0 < weight < 1.0:
        raise ValueError(f"Supervised fusion weight must be between zero and one: {weight}")
    return weight, best


def crown_branch_weights(
    best: dict, supervised_weight: float
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    upper = best.get("upper_supervised_weight")
    lower = best.get("lower_supervised_weight")
    if upper is None and lower is None:
        return None, None
    if upper is None or lower is None:
        raise ValueError("Both upper and lower fusion weights must be provided")
    supervised = {
        "background": supervised_weight,
        "upper": float(upper),
        "lower": float(lower),
    }
    if any(not 0.0 < value < 1.0 for value in supervised.values()):
        raise ValueError(f"Class-specific fusion weights must be between zero and one: {supervised}")
    semisupervised = {name: 1.0 - value for name, value in supervised.items()}
    return supervised, semisupervised


def clean_postprocess(payload: dict) -> dict[str, object]:
    if "best_geometry" in payload:
        payload = payload["best_geometry"]
    result: dict[str, object] = {
        "threshold": float(payload["threshold"]),
        "minimum_component_voxels": int(payload["minimum_component_voxels"]),
        "maximum_components": int(payload["maximum_components"]),
        "minimum_hu": float(payload["minimum_hu"]),
        "grid_size": int(payload.get("grid_size", 128)),
        "spacing_mm": float(payload.get("spacing_mm", 1.25)),
    }
    jaws = payload.get("jaws")
    if isinstance(jaws, dict):
        clean_jaws = {}
        for jaw in ("upper", "lower"):
            if jaw not in jaws:
                continue
            source = jaws[jaw]
            if not isinstance(source, dict):
                raise RuntimeError(f"Invalid {jaw} postprocess override")
            clean_jaws[jaw] = {
                key: source[key]
                for key in (
                    "threshold",
                    "minimum_component_voxels",
                    "maximum_components",
                    "minimum_hu",
                )
                if key in source
            }
        if clean_jaws:
            result["jaws"] = clean_jaws
    return result


def write_hash_manifest(root: Path) -> None:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")


def sync_runtime(project_root: Path, destination: Path) -> None:
    for relative in SCAFFOLD_FILES:
        require_file(destination, relative)
    script_destination = destination / "scripts"
    allowed_scripts = set(RUNTIME_SCRIPTS) | {
        "__init__.py",
        "gzip_file.py",
        "validate_outputs.py",
        "verify_assets.py",
    }
    for stale in script_destination.glob("*.py"):
        if stale.name not in allowed_scripts:
            stale.unlink()
    task2reg_source = project_root / "task2reg"
    task2reg_destination = destination / "task2reg"
    task2reg_destination.mkdir(parents=True, exist_ok=True)
    source_names = {path.name for path in task2reg_source.glob("*.py")}
    for stale in task2reg_destination.glob("*.py"):
        if stale.name not in source_names:
            stale.unlink()
    for source in sorted(task2reg_source.glob("*.py")):
        copy_file(source, task2reg_destination / source.name)
    for name in RUNTIME_SCRIPTS:
        copy_file(require_file(project_root / "scripts", name), destination / "scripts" / name)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    destination = args.destination.resolve()
    legacy_assets = args.legacy_assets.resolve()
    template_assets = args.template_assets.resolve()
    enhanced_assets = args.enhanced_assets.resolve()
    crown_supervised = args.crown_supervised.resolve()
    crown_semisupervised = args.crown_semisupervised.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    supervised_training = validate_final_training(
        crown_supervised, args.expected_members, args.expected_labeled_cases
    )
    semisupervised_training = validate_final_training(
        crown_semisupervised, args.expected_members, args.expected_labeled_cases
    )
    supervised_training["path"] = portable_path(crown_supervised, project_root)
    semisupervised_training["path"] = portable_path(
        crown_semisupervised, project_root
    )
    supervised_weight, fusion_best = select_fusion(
        args.fusion_summary.resolve(), args.supervised_weight
    )
    fusion_selection_evidence = None
    if args.fusion_selection_evidence is not None:
        evidence_path = args.fusion_selection_evidence.resolve()
        if not evidence_path.is_file():
            raise FileNotFoundError(evidence_path)
        evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8-sig"))
        fusion_selection_evidence = {
            "path": portable_path(evidence_path, project_root),
            "sha256": sha256(evidence_path),
            "robust_selection": evidence_payload.get("robust_selection"),
        }
    fusion_mode = args.fusion_mode or str(fusion_best.get("fusion_mode", "arithmetic"))
    if fusion_mode not in {"arithmetic", "geometric"}:
        raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

    postprocess_source = (
        args.postprocess_config.resolve()
        if args.postprocess_config is not None
        else args.fusion_summary.resolve()
    )
    postprocess_payload = json.loads(postprocess_source.read_text(encoding="utf-8-sig"))
    if args.postprocess_config is None:
        postprocess_payload = fusion_best
    postprocess = clean_postprocess(postprocess_payload)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage = destination / f"model_assets.next_{timestamp}"
    backup = destination / f"model_assets.previous_{timestamp}"
    safe_remove_tree(stage, destination)
    stage.mkdir(parents=True)

    for name in BASE_ASSETS:
        copy_file(require_file(legacy_assets, name), stage / name)
    for name in OPTIONAL_LEGACY_ASSETS:
        source = legacy_assets / name
        if source.is_file():
            copy_file(source, stage / name)
    for name in TEMPLATE_ASSETS:
        copy_file(require_file(template_assets, name), stage / name)
    for name in ENHANCED_ASSETS:
        copy_file(require_file(enhanced_assets, name), stage / name)
    for name in SOURCE_PROVENANCE_ASSETS:
        source = enhanced_assets / name
        if source.is_file():
            copy_file(source, stage / name)

    policy_path = stage / "deployment_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if args.crown_tta_mode is not None:
        policy["crown_tta_mode"] = args.crown_tta_mode
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    (stage / "crown_postprocess.json").write_text(
        json.dumps(postprocess, indent=2) + "\n", encoding="utf-8"
    )

    crown_root = stage / "crown_localizer"
    for branch_name, source_root in (
        ("supervised", crown_supervised),
        ("semisupervised", crown_semisupervised),
    ):
        for member in range(args.expected_members):
            source = require_file(source_root / f"fold_{member}", "best.pt")
            copy_file(source, crown_root / branch_name / f"fold_{member}" / "best.pt")
    supervised_class_weights, semisupervised_class_weights = crown_branch_weights(
        fusion_best, supervised_weight
    )
    supervised_branch = {
        "name": "supervised",
        "path": "supervised",
        "weight": supervised_weight,
        "fold_indices": list(range(args.expected_members)),
    }
    semisupervised_branch = {
        "name": "semisupervised",
        "path": "semisupervised",
        "weight": 1.0 - supervised_weight,
        "fold_indices": list(range(args.expected_members)),
    }
    if supervised_class_weights is not None:
        supervised_branch["weights"] = supervised_class_weights
        semisupervised_branch["weights"] = semisupervised_class_weights
    ensemble = {
        "mode": fusion_mode,
        "branches": [supervised_branch, semisupervised_branch],
    }
    (crown_root / "ensemble.json").write_text(
        json.dumps(ensemble, indent=2) + "\n", encoding="utf-8"
    )

    assembly = {
        "assembled_utc": timestamp,
        "fusion_summary": portable_path(args.fusion_summary, project_root),
        "fusion_summary_sha256": sha256(args.fusion_summary.resolve()),
        "fusion_best": fusion_best,
        "fusion_selection_evidence": fusion_selection_evidence,
        "crown_ensemble": ensemble,
        "crown_postprocess": postprocess,
        "legacy_assets": portable_path(legacy_assets, project_root),
        "template_assets": portable_path(template_assets, project_root),
        "enhanced_assets": portable_path(enhanced_assets, project_root),
        "supervised_training": supervised_training,
        "semisupervised_training": semisupervised_training,
    }
    (stage / "ASSEMBLY_MANIFEST.json").write_text(
        json.dumps(assembly, indent=2) + "\n", encoding="utf-8"
    )
    write_hash_manifest(stage)

    sync_runtime(project_root, destination)
    current = destination / "model_assets"
    if backup.exists():
        safe_remove_tree(backup, destination)
    if current.exists():
        current.rename(backup)
    try:
        stage.rename(current)
        verifier = require_file(destination / "scripts", "verify_assets.py")
        subprocess.run([sys.executable, str(verifier)], cwd=destination, check=True)
        for path in sorted((destination / "task2reg").glob("*.py")):
            py_compile.compile(str(path), doraise=True)
        for name in RUNTIME_SCRIPTS:
            py_compile.compile(str(destination / "scripts" / name), doraise=True)
    except Exception:
        if current.exists():
            safe_remove_tree(current, destination)
        if backup.exists():
            backup.rename(current)
        raise
    if backup.exists():
        safe_remove_tree(backup, destination)

    assembly["asset_manifest_sha256"] = sha256(current / "SHA256SUMS")
    assembly["asset_bytes"] = sum(path.stat().st_size for path in current.rglob("*") if path.is_file())
    report = destination / "ASSEMBLY_REPORT.json"
    report.write_text(json.dumps(assembly, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(assembly, indent=2), flush=True)


if __name__ == "__main__":
    main()
