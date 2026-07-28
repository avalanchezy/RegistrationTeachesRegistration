from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import joblib


REQUIRED_BASE_ASSETS = {
    "fallback_reranker.joblib",
    "labeled_templates.joblib",
    "primary_reranker.joblib",
    "rotation_prior_all30.json",
    "surface_templates.joblib",
}
REQUIRED_RUNTIME_SCRIPTS = {
    "augment_candidate_geometry.py",
    "evaluate_crown_candidate_consistency.py",
    "prepare_threshold_dental_roi.py",
    "refine_candidates_with_crown.py",
    "run_geometry_benchmark.py",
    "run_submission_inference.py",
    "validate_outputs.py",
    "verify_assets.py",
}
REQUIRED_RUNTIME_MODULES = {
    "candidate_learning.py",
    "crown_inference.py",
    "crown_localizer.py",
    "crown_network.py",
    "data.py",
    "dental_roi.py",
    "deployment_ensemble.py",
    "geometry.py",
    "metrics.py",
    "priors.py",
    "surfaces.py",
    "surface_template_transfer.py",
    "template_transfer.py",
    "threshold_consensus.py",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def relative_asset(root: Path, value: str) -> Path:
    relative = Path(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe asset path in SHA256SUMS: {value}")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"Asset path escapes model directory: {value}")
    return path


def verify_manifest(root: Path) -> None:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing asset manifest: {manifest}")
    expected_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    declared_files = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="ascii").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeError(f"Malformed SHA256SUMS line {line_number}: {line}")
        expected, filename = parts[0].lower(), parts[1].strip()
        path = relative_asset(root, filename)
        relative = path.relative_to(root).as_posix()
        if relative in declared_files:
            raise RuntimeError(f"Duplicate SHA256 entry: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Manifest asset is missing: {relative}")
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {relative}: {actual} != {expected}")
        declared_files.add(relative)
        print(f"OK {relative} {actual}")
    missing_hashes = sorted(expected_files - declared_files)
    extra_hashes = sorted(declared_files - expected_files)
    if missing_hashes or extra_hashes:
        raise RuntimeError(
            f"SHA256 coverage mismatch: missing={missing_hashes}, extra={extra_hashes}"
        )


def verify_source_provenance(root: Path) -> None:
    manifest = root / "model_asset_manifest.json"
    binding = root / "model_asset_manifest.sha256"
    if not manifest.exists() and not binding.exists():
        return
    if not manifest.is_file() or not binding.is_file():
        raise FileNotFoundError("Incomplete source model-asset provenance pair")
    expected = binding.read_text(encoding="ascii").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError("Malformed source model-asset manifest binding")
    observed = digest(manifest)
    if observed != expected:
        raise RuntimeError(
            f"Source model-asset manifest binding mismatch: {observed} != {expected}"
        )
    payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    fair_binding = str(payload.get("fair_selection_summary_sha256", "")).lower()
    if not payload.get("fair_selection_summary") or not re.fullmatch(
        r"[0-9a-f]{64}", fair_binding
    ):
        raise RuntimeError("Source manifest has no strict fairness binding")
    fair_path = root / "strict_fair_selection.json"
    if not fair_path.is_file() or digest(fair_path) != fair_binding:
        raise RuntimeError("Embedded strict fairness binding is missing or changed")
    fair = json.loads(fair_path.read_text(encoding="utf-8-sig"))
    if fair.get("status") != "PASS" or fair.get("baseline_mode") != "direct":
        raise RuntimeError("Embedded strict fairness selection is invalid")
    official_binding = str(
        payload.get("official_mode_selection_summary_sha256", "")
    ).lower()
    if not payload.get("official_mode_selection_summary") or not re.fullmatch(
        r"[0-9a-f]{64}", official_binding
    ):
        raise RuntimeError("Source manifest has no official-metric mode binding")
    official_path = root / "strict_official_mode_selection.json"
    if not official_path.is_file() or digest(official_path) != official_binding:
        raise RuntimeError("Embedded official-metric mode selection is missing or changed")
    official = json.loads(official_path.read_text(encoding="utf-8-sig"))
    selected_mode = str(payload.get("official_selected_mode", ""))
    if not selected_mode or str(official.get("recommended_mode", "")) != selected_mode:
        raise RuntimeError("Embedded official-metric mode selection disagrees with assets")
    policy = json.loads((root / "deployment_policy.json").read_text(encoding="utf-8"))
    score_mode = str(payload.get("candidate_score_mode", ""))
    if score_mode != str(policy.get("candidate_score_mode", "")):
        raise RuntimeError("Source manifest candidate score mode disagrees with assets")
    if score_mode == "unsupervised" and str(
        payload.get("unsupervised_score_key", "")
    ) != str(policy.get("unsupervised_score_key", "")):
        raise RuntimeError("Source manifest unsupervised score key disagrees with assets")
    sources = payload.get("formal_all30_reranker_sources")
    if not isinstance(sources, list):
        raise RuntimeError("Source manifest has no formal all-30 reranker provenance")
    expected_names = {
        "fallback_reranker.joblib",
        "primary_reranker.joblib",
        "roi_reranker.joblib",
    }
    observed_names = {str(source.get("destination", "")) for source in sources}
    if observed_names != expected_names:
        raise RuntimeError(
            "Unexpected formal all-30 reranker provenance: "
            f"{sorted(observed_names)}"
        )
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", "")).lower())
        for source in sources
    ):
        raise RuntimeError("Formal all-30 reranker provenance has an invalid hash")
    for source in sources:
        destination = root / str(source["destination"])
        if not destination.is_file():
            raise FileNotFoundError(
                f"Formal all-30 reranker destination is missing: {destination.name}"
            )
        expected_hash = str(source["sha256"]).lower()
        observed_hash = digest(destination)
        if observed_hash != expected_hash:
            raise RuntimeError(
                "Formal all-30 reranker does not match its declared source: "
                f"{destination.name} {observed_hash} != {expected_hash}"
            )
    print(f"OK source model provenance ({len(sources)} formal all-30 rerankers)")


def verify_crown_localizer(root: Path) -> int:
    crown_dir = (root / "crown_localizer").resolve()
    ensemble_path = crown_dir / "ensemble.json"
    if ensemble_path.is_file():
        ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
        if ensemble.get("mode", "arithmetic") not in {"arithmetic", "geometric"}:
            raise RuntimeError("Unsupported crown-localizer blend mode")
        branches = ensemble.get("branches")
        if not isinstance(branches, list) or not branches:
            raise RuntimeError("crown_localizer/ensemble.json has no branches")
    else:
        branches = [{"name": "default", "path": "."}]

    checkpoints = []
    for index, branch in enumerate(branches):
        if not isinstance(branch, dict):
            raise RuntimeError(f"Invalid crown-localizer branch {index}")
        branch_name = str(branch.get("name", f"branch_{index}"))
        branch_dir = (crown_dir / str(branch.get("path", "."))).resolve()
        if branch_dir != crown_dir and crown_dir not in branch_dir.parents:
            raise RuntimeError(f"Crown branch escapes model directory: {branch_name}")
        class_weights = branch.get("weights")
        if class_weights is not None:
            if not isinstance(class_weights, dict):
                raise RuntimeError(f"Invalid class weights for crown branch {branch_name}")
            for jaw in ("background", "upper", "lower"):
                value = float(class_weights.get(jaw, branch.get("weight", 1.0)))
                if value <= 0.0:
                    raise RuntimeError(
                        f"Non-positive {jaw} weight for crown branch {branch_name}"
                    )
        configured_folds = branch.get("fold_indices")
        folds = (
            [int(value) for value in configured_folds]
            if configured_folds is not None
            else sorted(
                int(path.name.split("_", 1)[1])
                for path in branch_dir.glob("fold_*")
                if path.is_dir() and path.name.split("_", 1)[1].isdigit()
            )
        )
        if not folds:
            raise RuntimeError(f"No folds configured for crown branch {branch_name}")
        for fold in folds:
            checkpoint = branch_dir / f"fold_{fold}" / "best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Missing crown-localizer checkpoint: {checkpoint.relative_to(root)}"
                )
            if checkpoint.stat().st_size < 10_000_000:
                raise RuntimeError(f"Truncated crown-localizer checkpoint: {checkpoint}")
            checkpoints.append(checkpoint.resolve())
    if len(set(checkpoints)) != len(checkpoints):
        raise RuntimeError("A crown-localizer checkpoint is configured more than once")
    return len(checkpoints)


def verify_assembly_manifest(root: Path) -> None:
    path = root / "ASSEMBLY_MANIFEST.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing assembly manifest: {path}")
    text = path.read_text(encoding="utf-8")
    if re.search(r"[A-Za-z]:[\\/]", text):
        raise RuntimeError("Assembly manifest contains a host-absolute Windows path")
    json.loads(text)
    print("OK portable assembly manifest")


def verify_enhanced_assets(root: Path) -> None:
    ensemble_files = {
        "regression_ensemble.joblib",
        "pairwise_ensemble.joblib",
        "deployment_policy.json",
        "crown_postprocess.json",
    }
    present = {name for name in ensemble_files if (root / name).is_file()}
    if present and present != ensemble_files:
        raise RuntimeError(
            "Incomplete deployment ensemble assets: "
            + ", ".join(sorted(ensemble_files - present))
        )
    if not present:
        return
    for name in ("regression_ensemble.joblib", "pairwise_ensemble.joblib"):
        if (root / name).stat().st_size < 100_000:
            raise RuntimeError(f"Truncated deployment model: {name}")
    policy = json.loads((root / "deployment_policy.json").read_text(encoding="utf-8"))
    required_policy = {
        "blend_alpha",
        "candidate_score_mode",
        "crown_tta_mode",
        "global_crown_modes",
        "include_legacy_threshold_candidates",
        "joint_angle_weight_mm_per_deg",
        "joint_pair_top_k",
        "joint_translation_weight",
        "pair_prior_relative_rotation",
        "pair_prior_relative_translation",
        "pairwise_aggregation",
        "regression_aggregation",
    }
    missing_policy = required_policy - set(policy)
    if missing_policy:
        raise RuntimeError(
            "Deployment policy is missing: " + ", ".join(sorted(missing_policy))
        )
    score_mode = str(policy["candidate_score_mode"])
    if score_mode not in {"learned_blend", "unsupervised"}:
        raise RuntimeError(f"Unsupported deployment candidate score mode: {score_mode}")
    if score_mode == "unsupervised" and policy.get("unsupervised_score_key") not in {
        "rank_score_mm",
        "selection_score_mm",
    }:
        raise RuntimeError("Unsupported deployment unsupervised score key")
    provenance_path = root / "selection_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
        if str(provenance.get("candidate_score_mode", "")) != score_mode:
            raise RuntimeError(
                "Deployment candidate score mode disagrees with selection provenance"
            )
        if score_mode == "unsupervised" and str(
            provenance.get("unsupervised_score_key", "")
        ) != str(policy["unsupervised_score_key"]):
            raise RuntimeError(
                "Deployment unsupervised score key disagrees with selection provenance"
            )
    postprocess = json.loads(
        (root / "crown_postprocess.json").read_text(encoding="utf-8")
    )
    jaws = postprocess.get("jaws", {})
    if jaws and set(jaws) != {"upper", "lower"}:
        raise RuntimeError("Jaw-specific postprocess must configure upper and lower")
    checkpoint_count = verify_crown_localizer(root)
    print(f"OK challenge-only crown localizer ({checkpoint_count} checkpoints)")


def verify_template_banks(root: Path) -> None:
    exact = joblib.load(root / "labeled_templates.joblib")
    exact_entries = exact.get("entries")
    if not isinstance(exact_entries, list) or not exact_entries:
        raise RuntimeError("Exact-template bank is empty or malformed")

    surface = joblib.load(root / "surface_templates.joblib")
    surface_entries = surface.get("entries")
    if not isinstance(surface_entries, list) or not surface_entries:
        raise RuntimeError("Surface-template bank is empty or malformed")
    gate = float(
        surface.get("surface_transfer", {}).get(
            "max_teacher_predicted_tre_mm", 1.5
        )
    )
    if not math.isfinite(gate) or not 0.0 < gate <= 2.5:
        raise RuntimeError(f"Invalid surface-template teacher gate: {gate}")
    violations = [
        (str(entry.get("case_id", "")), str(entry.get("jaw", "")))
        for entry in surface_entries
        if float(entry.get("predicted_tre_mm", 0.0)) > gate + 1e-8
    ]
    if violations:
        raise RuntimeError(
            f"Surface-template teachers exceed the frozen {gate} mm gate: "
            f"{violations[:5]}"
        )
    print(
        f"OK template banks (exact={len(exact_entries)}, "
        f"surface={len(surface_entries)}, teacher_gate={gate:.3f} mm)"
    )


def verify_runtime(algorithm_root: Path) -> None:
    script_root = algorithm_root / "scripts"
    module_root = algorithm_root / "task2reg"
    missing_scripts = sorted(
        name for name in REQUIRED_RUNTIME_SCRIPTS if not (script_root / name).is_file()
    )
    missing_modules = sorted(
        name for name in REQUIRED_RUNTIME_MODULES if not (module_root / name).is_file()
    )
    if missing_scripts or missing_modules:
        raise FileNotFoundError(
            f"Incomplete runtime closure: scripts={missing_scripts}, modules={missing_modules}"
        )
    print(
        f"OK runtime closure ({len(REQUIRED_RUNTIME_SCRIPTS)} scripts, "
        f"{len(REQUIRED_RUNTIME_MODULES)} modules)"
    )


def main() -> None:
    algorithm_root = Path(__file__).resolve().parents[1]
    verify_runtime(algorithm_root)
    root = (algorithm_root / "model_assets").resolve()
    missing_base = sorted(name for name in REQUIRED_BASE_ASSETS if not (root / name).is_file())
    if missing_base:
        raise FileNotFoundError("Missing baseline assets: " + ", ".join(missing_base))
    verify_manifest(root)
    verify_source_provenance(root)
    verify_assembly_manifest(root)
    verify_template_banks(root)
    verify_enhanced_assets(root)


if __name__ == "__main__":
    main()
