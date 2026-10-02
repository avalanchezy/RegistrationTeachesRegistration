"""Synthetic conversion of the old selector's actual audit.csv format."""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from task2reg.journal.data import load_case, load_journal_manifest


def inputs(tmp_path, *, duplicate_source=False):
    data = tmp_path / "original"
    data.mkdir()
    records = []
    points = np.array([[1., 1., 1.], [2., 1., 1.], [1., 2., 1.]])
    definitions = [("site", "train", "t", "upper"), ("site", "val", "v", "upper"),
                   ("site", "unlabeled", "001", "upper"), ("site", "unlabeled", "001", "lower"),
                   ("site", "unlabeled", "002", "upper"), ("site", "unlabeled", "002", "lower")]
    if duplicate_source:
        definitions.append(("other", "unlabeled", "001", "upper"))
    for source, split, old_id, jaw in definitions:
        path = data / f"{source}_{old_id}_{jaw}.npz"
        payload = dict(image=np.zeros((16, 16, 16), dtype=np.int16), affine=np.eye(4),
                       points=points, anchors=points, candidates=np.eye(4)[None])
        if split != "unlabeled":
            payload["transform"] = np.eye(4)
        np.savez_compressed(path, **payload)
        records.append(dict(case_id=f"{source}:{old_id}", original_case_id=old_id, source=source,
                            patient_id=f"{source}-{old_id}", split=split, jaw=jaw,
                            npz_path=f"original/{path.name}",
                            reference_kind="none" if split == "unlabeled" else "manual",
                            candidate_provenance={"kind": "geometry_only"}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "records": records}))
    label_dir = tmp_path / "top_80" / "fold_0"
    label_dir.mkdir(parents=True)
    labels = np.zeros((16, 16, 16), dtype=np.uint8)
    labels[1:3] = 1
    labels[9:11] = 2
    for old_id in ("001", "002"):
        np.savez_compressed(label_dir / f"{old_id}.npz", label=labels, affine=np.eye(4))
    audit = tmp_path / "audit.csv"
    audit.write_text("fold,case_id,upper_voxels,lower_voxels,confidence,entropy,selection_score,accepted_filter\n"
                     "0,001,512,512,0.9,0.1,0.8,1\n0,002,512,512,0.8,0.2,0.6,1\n"
                     "1,001,512,512,0.9,0.1,0.8,1\n")
    teacher = tmp_path / "teacher.json"
    teacher.write_text(json.dumps({"teacher_id": "old-fold0", "excluded_patient_ids": ["site-v"]}))
    return manifest, label_dir, audit, teacher


def run_import(manifest, label_dir, audit, teacher, output, *options):
    module = importlib.import_module("scripts.import_legacy_support_pseudolabels")
    args = module.parse_args(["--manifest", str(manifest), "--pseudo-label-dir", str(label_dir),
                              "--selection-csv", str(audit), "--teacher-provenance", str(teacher),
                              "--output-dir", str(output), *options])
    return module.import_pseudo_labels(args)


def test_legacy_import_preserves_case_budget_jaw_pairs_and_relative_paths(tmp_path):
    manifest, label_dir, audit, teacher = inputs(tmp_path)
    result = run_import(manifest, label_dir, audit, teacher, tmp_path / "imported", "--max-cases", "1")
    rows = load_journal_manifest(result)
    pseudo = [row for row in rows if row["split"] == "pseudo"]
    assert len(pseudo) == 2
    assert {row["jaw"] for row in pseudo} == {"upper", "lower"}
    assert {row["original_case_id"] for row in pseudo} == {"001"}
    assert sum(row["split"] == "unlabeled" for row in rows) == 2
    for row in pseudo:
        case = load_case(row)
        assert row["pseudo_kind"] == "support" and "label" in case
        assert "transform" not in case and "candidates" not in case
    raw = json.loads(result.read_text())
    assert all(not Path(row["npz_path"]).is_absolute() for row in raw["records"])
    unchanged = next(row for row in rows if row["split"] == "train")
    assert Path(unchanged["npz_path"]).parent == tmp_path / "original"


@pytest.mark.parametrize("fault", ["shape", "affine", "label", "transform"])
def test_legacy_import_rejects_bad_prediction_arrays_without_publishing_manifest(tmp_path, fault):
    manifest, label_dir, audit, teacher = inputs(tmp_path)
    with np.load(label_dir / "001.npz") as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    if fault == "shape":
        payload["label"] = payload["label"][:8]
    elif fault == "affine":
        payload["affine"][0, 3] = 2
    elif fault == "label":
        payload["label"][0, 0, 0] = 17
    else:
        payload["transform"] = np.eye(4)
    np.savez_compressed(label_dir / "001.npz", **payload)
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match=fault):
        run_import(manifest, label_dir, audit, teacher, output)
    assert not (output / "manifest.json").exists()


def test_legacy_import_rejects_teacher_leakage(tmp_path):
    manifest, label_dir, audit, teacher = inputs(tmp_path)
    teacher.write_text(json.dumps({"teacher_id": "leaky", "excluded_patient_ids": []}))
    with pytest.raises(ValueError, match="exclu"):
        run_import(manifest, label_dir, audit, teacher, tmp_path / "bad")


def test_legacy_import_requires_source_for_old_id_collision(tmp_path):
    manifest, label_dir, audit, teacher = inputs(tmp_path, duplicate_source=True)
    with pytest.raises(ValueError, match="source"):
        run_import(manifest, label_dir, audit, teacher, tmp_path / "bad")
    result = run_import(manifest, label_dir, audit, teacher, tmp_path / "good", "--source", "site")
    rows = load_journal_manifest(result)
    assert all(row["split"] == "unlabeled" for row in rows if row["source"] == "other")


def test_legacy_import_rejects_inconsistent_original_ids_between_jaws(tmp_path):
    manifest, label_dir, audit, teacher = inputs(tmp_path)
    payload = json.loads(manifest.read_text())
    row = next(r for r in payload["records"] if r["case_id"] == "site:001" and r["jaw"] == "lower")
    row["original_case_id"] = "003"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="original_case_id"):
        run_import(manifest, label_dir, audit, teacher, tmp_path / "bad")


def test_legacy_import_uses_copied_membership_not_just_accepted_filter(tmp_path):
    manifest, label_dir, audit, teacher = inputs(tmp_path)
    (label_dir / "001.npz").unlink()
    result = run_import(manifest, label_dir, audit, teacher, tmp_path / "good")
    rows = load_journal_manifest(result)
    assert {r["original_case_id"] for r in rows if r["split"] == "pseudo"} == {"002"}


def test_legacy_import_cli_runs_real_conversion(tmp_path):
    manifest, label_dir, audit, teacher = inputs(tmp_path)
    command = [sys.executable, "scripts/import_legacy_support_pseudolabels.py", "--manifest", str(manifest),
               "--pseudo-label-dir", str(label_dir), "--selection-csv", str(audit),
               "--teacher-provenance", str(teacher), "--output-dir", str(tmp_path / "cli"), "--source", "site"]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    assert (tmp_path / "cli" / "manifest.json").is_file()
