from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
    "Dockerfile",
    "predict.sh",
    "task2reg/crown_network.py",
    "scripts/train_final_crown_localizer.py",
    "scripts/fit_final_multiseed_ensemble.py",
    "scripts/run_submission_inference.py",
    "configs/training/final_method.json",
    "configs/submission/deployment_policy.json",
    "configs/submission/runtime_source.sha256",
)
FORBIDDEN_SUFFIXES = (
    ".nii",
    ".nii.gz",
    ".stl",
    ".ply",
    ".obj",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".joblib",
    ".pkl",
    ".pickle",
    ".tar",
    ".tar.gz",
    ".zip",
)
TEXT_SUFFIXES = {
    ".cff",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PRIVATE_PATTERNS = (
    re.compile(r"C:\\Research", re.IGNORECASE),
    re.compile(r"Users\\zhuyi", re.IGNORECASE),
    re.compile(r"filesender", re.IGNORECASE),
    re.compile(r"SemiTeethSegChallenge@outlook\.com", re.IGNORECASE),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_files() -> list[Path]:
    ignored_roots = {".git", ".pytest_cache", "__pycache__"}
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in ignored_roots for part in path.parts)
    )


def audit_required(errors: list[str]) -> None:
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")


def audit_payload(files: list[Path], errors: list[str]) -> None:
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        lowered = relative.lower()
        if any(lowered.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            errors.append(f"restricted data or binary artifact: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"unexpected file larger than 10 MiB: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 text file: {relative}")
            continue
        if relative == "scripts/audit_source_release.py":
            continue
        for pattern in PRIVATE_PATTERNS:
            if pattern.search(content):
                errors.append(f"private path or submission detail in {relative}")


def audit_runtime_hashes(errors: list[str]) -> None:
    manifest = ROOT / "configs/submission/runtime_source.sha256"
    if not manifest.is_file():
        return
    for line in manifest.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative.strip()
        if not path.is_file():
            errors.append(f"runtime hash target missing: {relative}")
        elif sha256(path) != expected:
            errors.append(f"runtime hash mismatch: {relative}")


def audit_configuration(errors: list[str]) -> None:
    path = ROOT / "configs/training/final_method.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    weak = payload["weak_support"]
    fusion = payload["crown_ensemble"]["branch_weights"]
    ranking = payload["candidate_ranking"]
    expected = {
        "weak_support.surface_radius_mm": (weak["surface_radius_mm"], 0.7),
        "weak_support.minimum_hu": (weak["minimum_hu"], -1000.0),
        "crown_ensemble.supervised_weight": (fusion["supervised"], 0.5),
        "crown_ensemble.semisupervised_weight": (fusion["semisupervised"], 0.5),
        "candidate_ranking.features": (ranking["features"], 97),
        "candidate_ranking.blend_alpha": (ranking["blend_alpha"], 0.575),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            errors.append(f"configuration drift: {name}={actual!r}, expected {wanted!r}")

    policy_path = ROOT / "configs/submission/deployment_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("global_crown_modes") != ["crown"]:
        errors.append("configuration drift: submitted target mode is not crown-only")
    if policy.get("include_legacy_threshold_candidates") is not False:
        errors.append("configuration drift: legacy threshold candidates are enabled")


def audit_json(files: list[Path], errors: list[str]) -> None:
    for path in files:
        if path.suffix.lower() != ".json":
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid JSON in {path.relative_to(ROOT).as_posix()}: {exc}")


def audit_model_assets(errors: list[str]) -> None:
    asset_root = ROOT / "model_assets"
    if not asset_root.is_dir():
        errors.append("missing model_assets policy directory")
        return
    unexpected = [
        path.relative_to(ROOT).as_posix()
        for path in asset_root.rglob("*")
        if path.is_file() and path.name != "README.md"
    ]
    if unexpected:
        errors.append(f"generated model assets present: {unexpected}")


def main() -> None:
    errors: list[str] = []
    files = tracked_files()
    audit_required(errors)
    audit_payload(files, errors)
    audit_json(files, errors)
    audit_model_assets(errors)
    audit_runtime_hashes(errors)
    audit_configuration(errors)
    if errors:
        raise SystemExit("Source release audit failed:\n- " + "\n- ".join(errors))
    print(f"Source release audit passed: {len(files)} files", flush=True)


if __name__ == "__main__":
    main()
