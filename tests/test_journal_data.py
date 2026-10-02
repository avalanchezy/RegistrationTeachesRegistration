"""Protocol and physical supervision tests; all fixtures are synthetic."""
import csv
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def api():
    return importlib.import_module("task2reg.journal.data")


def record(case="s:1", patient="p1", split="train", **changes):
    row = dict(case_id=case, patient_id=patient, source="s", split=split,
               jaw="upper", npz_path="data/1.npz", reference_kind="manual",
               candidate_provenance={"kind": "none"})
    row.update(changes)
    return row


def payload():
    affine = np.array([[0., -2., 0., 10.], [1., 0., 0., -3.], [0., 0., 3., 2.], [0., 0., 0., 1.]])
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    transform = np.eye(4)
    transform[:3, 3] = (4., 3., 8.)
    return dict(image=np.zeros((12, 10, 8), dtype=np.int16), affine=affine,
                points=points, anchors=points.copy(), transform=transform)


def test_manifest_relocates_relative_npz_paths(tmp_path):
    manifest = tmp_path / "config" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"schema_version": 1, "records": [record()]}))
    rows = api().load_journal_manifest(manifest, data_root=tmp_path / "moved")
    assert Path(rows[0]["npz_path"]) == tmp_path / "moved" / "data" / "1.npz"


@pytest.mark.parametrize("change", [
    dict(patient_id="p1", source="other"),
    dict(patient_id="p2", content_hash="duplicate"),
])
def test_protocol_rejects_patient_and_content_leakage_across_sources(change):
    first = record(content_hash="duplicate")
    second = record(case="other:2", split="val", **change)
    with pytest.raises(ValueError, match="(patient|content|overlap)"):
        api().validate_protocol([first, second])


def test_protocol_rejects_external_source_training_and_silver_supervision():
    with pytest.raises(ValueError, match="external"):
        api().validate_protocol([record(), record(case="s:2", patient="p2", split="external_test")])
    with pytest.raises(ValueError, match="silver"):
        api().validate_protocol([record(reference_kind="silver")])


@pytest.mark.parametrize("shared", [{"patient_id": "shared"}, {"content_hash": "shared-content"}])
def test_pseudo_and_unlabeled_scans_share_training_pool_without_eval_leakage(shared):
    unlabeled = record(case="s:u1", patient="u1", split="unlabeled", reference_kind="none", **shared)
    pseudo = record(case="s:u2", patient="u2", split="pseudo", reference_kind="pseudo",
                    pseudo_provenance={"teacher_id": "round0", "excluded_patient_ids": ["shared", "u1"]}, **shared)
    api().validate_protocol([unlabeled, pseudo])
    for split in ("train", "val", "test"):
        protected = {**unlabeled, "split": split, "reference_kind": "manual"}
        with pytest.raises(ValueError, match="overlap"):
            api().validate_protocol([protected, pseudo])


def test_unlabeled_and_pseudo_jaws_of_same_case_still_cannot_split():
    unlabeled = record(case="s:u", split="unlabeled", reference_kind="none")
    pseudo = record(case="s:u", split="pseudo", reference_kind="pseudo", jaw="lower",
                    pseudo_provenance={"teacher_id": "round0", "excluded_patient_ids": []})
    with pytest.raises(ValueError, match="two jaws"):
        api().validate_protocol([unlabeled, pseudo])


def test_protocol_requires_complete_learned_candidate_exclusions():
    training = record(candidate_provenance={"kind": "learned", "model_id": "fold0", "excluded_patient_ids": ["p1"]})
    validation = record(case="s:2", patient="p2", split="val")
    with pytest.raises(ValueError, match="exclu"):
        api().validate_protocol([training, validation])
    training["candidate_provenance"]["excluded_patient_ids"].append("p2")
    api().validate_protocol([training, validation])


def test_pseudo_requires_teacher_provenance_and_eval_exclusions():
    pseudo = record(case="s:u", patient="pu", split="pseudo", reference_kind="pseudo")
    validation = record(case="s:v", patient="pv", split="val")
    with pytest.raises(ValueError, match="pseudo.*provenance"):
        api().validate_protocol([pseudo, validation])
    pseudo["pseudo_provenance"] = {"teacher_id": "teacher0", "excluded_patient_ids": []}
    with pytest.raises(ValueError, match="exclu"):
        api().validate_protocol([pseudo, validation])
    pseudo["pseudo_provenance"]["excluded_patient_ids"] = ["pv"]
    api().validate_protocol([pseudo, validation])


def test_load_case_checks_physical_arrays_and_unlabeled_gt(tmp_path):
    path = tmp_path / "case.npz"
    np.savez_compressed(path, **payload())
    result = api().load_case(record(npz_path=str(path)))
    np.testing.assert_array_equal(result["affine"], payload()["affine"])
    with pytest.raises(ValueError, match="(unlabeled|reference|transform)"):
        api().load_case(record(split="unlabeled", reference_kind="none", npz_path=str(path)))
    broken = payload()
    broken["affine"][2] = 0
    np.savez_compressed(path, **broken)
    with pytest.raises(ValueError, match="affine"):
        api().load_case(record(npz_path=str(path)))


def test_support_pseudo_case_needs_label_but_not_transform(tmp_path):
    case = payload()
    del case["transform"]
    case["label"] = np.zeros_like(case["image"], dtype=np.uint8)
    path = tmp_path / "pseudo.npz"
    np.savez_compressed(path, **case)
    row = record(split="pseudo", reference_kind="pseudo", pseudo_kind="support", npz_path=str(path),
                 pseudo_provenance={"teacher_id": "old-crown", "excluded_patient_ids": []})
    loaded = api().load_case(row)
    assert "label" in loaded and "transform" not in loaded
    with pytest.raises(ValueError, match="transform"):
        api().sample_queries(loaded, 20, np.random.default_rng(4))


def test_queries_are_reproducible_physical_distances_and_uncertainty_weights():
    case = payload()
    case["point_weights"] = np.array([0.2, 0.4, 0.6, 0.8])
    a = api().sample_queries(case, 101, np.random.default_rng(42), truncation_mm=5)
    b = api().sample_queries(case, 101, np.random.default_rng(42), truncation_mm=5)
    for key in ("points_world", "targets", "weights"):
        np.testing.assert_array_equal(a[key], b[key])
    assert a["points_world"].shape == (101, 3)
    surface = case["points"] + case["transform"][:3, 3]
    all_distances = np.linalg.norm(a["points_world"][:, None] - surface[None], axis=2)
    nearest = all_distances.argmin(axis=1)
    np.testing.assert_allclose(a["targets"], np.minimum(all_distances.min(axis=1), 5), atol=1e-6)
    np.testing.assert_allclose(a["weights"], case["point_weights"][nearest] * a["in_roi"])
    assert np.sum(a["targets"] < 1e-5) >= 10


def test_query_candidates_are_sampled_in_world_coordinates():
    case = payload()
    candidate = case["transform"].copy()
    candidate[0, 3] += 2
    case["candidates"] = candidate[None]
    result = api().sample_queries(case, 100, np.random.default_rng(2))
    moved = case["points"] + candidate[:3, 3]
    distance = np.linalg.norm(result["points_world"][:, None] - moved[None], axis=2)
    assert np.sum(distance.min(axis=1) < 1e-6) >= 10


def test_queries_outside_affine_roi_have_no_reconstruction_supervision():
    case = payload()
    case["affine"] = np.array([[0., -2., .2, 10.], [1., 0., 0., -5.],
                               [0., 0., 3., 12.], [0., 0., 0., 1.]])
    case["transform"] = np.eye(4)
    # Surface on the boundary forces near/outer sampling to visit both sides.
    ijk = np.array([[0., 2., 2.], [0., 3., 2.], [0., 2., 3.], [0., 3., 3.]])
    case["points"] = ijk @ case["affine"][:3, :3].T + case["affine"][:3, 3]
    samples = api().sample_queries(case, 1000, np.random.default_rng(11))
    inverse = np.linalg.inv(case["affine"])
    voxel = samples["points_world"] @ inverse[:3, :3].T + inverse[:3, 3]
    inside = ((voxel >= -1e-5) & (voxel <= np.array(case["image"].shape) - 1 + 1e-5)).all(axis=1)
    assert np.any(~inside) and np.any(inside)
    assert np.all(samples["weights"][~inside] == 0)
    assert np.any(samples["weights"][inside] > 0)


def test_pseudo_queries_ignore_unknown_space(tmp_path):
    case = payload()
    case["point_weights"] = np.full(len(case["points"]), 0.7)
    path = tmp_path / "pseudo.npz"
    np.savez_compressed(path, **case)
    row = record(split="pseudo", reference_kind="pseudo", npz_path=str(path),
                 supervision_radius_mm=1.,
                 pseudo_provenance={"teacher_id": "round0", "excluded_patient_ids": []})
    samples = api().sample_queries(api().load_case(row), 100, np.random.default_rng(4))
    assert np.all(samples["weights"][samples["targets"] > 1.] == 0)
    assert np.any(samples["weights"][samples["targets"] < 1.] > 0)


def test_candidate_npz_cannot_bypass_provenance_declaration(tmp_path):
    case = payload()
    case["candidates"] = np.eye(4)[None]
    path = tmp_path / "case.npz"
    np.savez_compressed(path, **case)
    with pytest.raises(ValueError, match="provenance"):
        api().load_case(record(npz_path=str(path)))


def test_preparer_surface_sampling_follows_area_not_triangle_count():
    import trimesh
    module = importlib.import_module("scripts.prepare_registration_field_data")
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                         [10, 0, 0], [13, 0, 0], [10, 3, 0]], dtype=float)
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [3, 4, 5]], process=False)
    points = module.sample_mesh_surface(mesh, 10000, np.random.default_rng(7))
    # Two triangles but the distant triangle has nine times the area.
    assert 0.88 < np.mean(points[:, 0] > 5) < 0.92


def test_prepare_cli_imports_mesh_without_gt_dependent_crop(tmp_path):
    import trimesh
    mesh = trimesh.creation.box(extents=(4., 2., 1.))
    mesh_path = tmp_path / "ios.stl"
    mesh.export(mesh_path)
    gt_path = tmp_path / "gt.npy"
    np.save(gt_path, np.eye(4))
    legacy_data = tmp_path / "old_data"
    legacy_data.mkdir()
    np.savez_compressed(legacy_data / "001.npz", image=np.zeros((16, 16, 16), dtype=np.int16), affine=np.eye(4))
    old_csv = tmp_path / "old.csv"
    with old_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "case_id", "jaw", "cbct_path", "ios_path", "transform_path", "complete"])
        writer.writeheader()
        writer.writerow(dict(split="Train-Labeled", case_id="001", jaw="upper", cbct_path="", ios_path=str(mesh_path), transform_path=str(gt_path), complete="True"))
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("source,case_id,patient_id,split,reference_kind\nsite,001,p1,train,manual\n")
    output = tmp_path / "prepared"
    command = [sys.executable, "scripts/prepare_registration_field_data.py", "--manifest", str(old_csv),
               "--data-dir", str(legacy_data), "--metadata", str(metadata), "--output-dir", str(output),
               "--points", "64", "--anchors", "32", "--source", "site"]
    result = subprocess.run(command, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stdout + result.stderr
    rows = api().load_journal_manifest(output / "manifest.json")
    assert rows[0]["case_id"] == "site:001"
    case = api().load_case(rows[0])
    assert case["points"].shape == (64, 3) and case["anchors"].shape == (32, 3)
    assert rows[0]["surface_definition"] == "full_ios"
    assert "full IOS" in result.stderr
