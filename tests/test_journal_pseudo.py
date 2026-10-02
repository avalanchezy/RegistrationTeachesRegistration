import numpy as np
import pytest

from task2reg.journal.pseudo import spatial_sectors, assemble_pseudo_manifest
from task2reg.journal.data import load_journal_manifest, load_case
from test_journal_engine import synthetic_manifest


def test_spatial_sectors_are_balanced_and_deterministic():
    points = np.column_stack((np.arange(40), np.zeros(40), np.zeros(40)))
    sectors = spatial_sectors(points, 4)
    np.testing.assert_array_equal(np.bincount(sectors), [10]*4)
    np.testing.assert_array_equal(sectors, spatial_sectors(points + 100, 4))
    with pytest.raises(ValueError):
        spatial_sectors(points[:2], 4)


def test_verification_seed_is_bound_to_identity_not_manifest_order():
    from task2reg.journal.pseudo import verification_seed
    a = dict(source="s", case_id="c1", jaw="upper", patient_id="p1")
    b = a | {"case_id": "c2"}
    forward = {r["case_id"]: verification_seed(r, 12) for r in (a, b)}
    reverse = {r["case_id"]: verification_seed(r, 12) for r in (b, a)}
    assert forward == reverse
    assert forward["c1"] != forward["c2"]
    assert verification_seed(a, 12) != verification_seed(a | {"jaw": "lower"}, 12)


def test_pseudo_export_replaces_unlabeled_and_preserves_spatial_weights(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    records = load_journal_manifest(manifest)
    unlabeled = records[-1]
    result = dict(accepted=True, reasons=[], transform=np.eye(4),
                  point_weights=np.linspace(.1, 1, 48), metrics={"u95_mm": .2})
    path = assemble_pseudo_manifest(records, [(unlabeled, result)], tmp_path / "pseudo",
                                   teacher_id="synthetic-teacher", excluded_patient_ids=["p1"])
    exported = load_journal_manifest(path)
    pseudo = [r for r in exported if r["split"] == "pseudo"]
    assert len(pseudo) == 1
    assert not any(r["split"] == "unlabeled" for r in exported)
    case = load_case(pseudo[0])
    np.testing.assert_allclose(case["point_weights"], result["point_weights"])
    assert "label" not in case


def test_rejected_registration_never_becomes_pseudo(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    records = load_journal_manifest(manifest)
    path = assemble_pseudo_manifest(records, [(records[-1], {"accepted":False, "reasons":["pose_uncertainty"]})],
                                   tmp_path / "pseudo", teacher_id="teacher", excluded_patient_ids=["p1"])
    exported = load_journal_manifest(path)
    assert not any(r["split"] == "pseudo" for r in exported)


def test_holdout_starts_never_use_full_surface_refinement(tmp_path, monkeypatch):
    import torch
    import task2reg.journal.pseudo as pseudo
    from task2reg.journal.field import RegistrationField
    record = load_journal_manifest(synthetic_manifest(tmp_path))[-1]
    case = load_case(record)
    contaminated = np.eye(4)
    contaminated[0, 3] = 100
    monkeypatch.setattr(pseudo, "score_and_refine", lambda *a, **kw: [dict(transform=contaminated.tolist(), field_energy=0.)])
    initials = []
    def fake_refine(query_fn, points, initial, *args, **kwargs):
        initials.append(initial.detach().clone())
        return {"transform": initial.detach()}
    monkeypatch.setattr(pseudo, "refine_transform", fake_refine)
    model = RegistrationField(base_channels=4).eval().requires_grad_(False)
    pseudo.verify_case(model, case, record, "cpu", starts=4, sectors=4,
                       refinement_steps=0, point_budget=48)
    assert all(float(t[:, :3, 3].abs().max()) < 10 for t in initials)
