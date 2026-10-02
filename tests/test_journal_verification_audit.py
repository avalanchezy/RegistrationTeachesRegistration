import json

import numpy as np
import pytest


def api():
    from task2reg.journal import verification_audit
    return verification_audit


def row(patient="p1", case="c1", jaw="upper", accepted=True, wrong=False):
    return dict(patient_id=patient, source="s", case_id=case, jaw=jaw,
                reference_kind="manual", accepted=accepted, d_mm=3. if wrong else .1,
                registration_failed=wrong, verification_failed=False,
                metrics={"u95_mm": .2}, reasons=[])


def test_exact_risk_bound_does_not_call_small_zero_error_sample_safe():
    assert api().binomial_upper(0, 0) == 1
    assert api().binomial_upper(3, 3) == 1
    assert api().binomial_upper(0, 30) == pytest.approx(1 - .05 ** (1 / 30))
    assert api().binomial_upper(0, 30) > .05
    assert api().binomial_upper(0, 59) < .05
    with pytest.raises(ValueError):
        api().binomial_upper(2, 1)


def test_case_gate_then_patient_risk_prevents_double_counting_and_partial_exports():
    rows = [row(), row(jaw="lower", accepted=False),
            row(case="c2", wrong=True), row(case="c2", jaw="lower"),
            row(patient="p2", case="c3")]
    result = api().summarize_audit(rows)
    assert result["n_cases"] == 3
    assert result["n_accepted_cases"] == 2
    assert result["n_accepted_patients"] == 2
    assert result["n_incorrect_accepted_patients"] == 1
    assert result["accepted_patient_risk"] == .5
    assert result["case_coverage"] == pytest.approx(2 / 3)
    # Upper/lower and repeated scans remain one independent patient observation.
    repeated = rows + [row(case="c4", wrong=True)]
    assert api().summarize_audit(repeated)["risk_upper_bound"] == result["risk_upper_bound"]


def test_wrong_parity_is_failure_even_when_coplanar_anchor_displacement_zero():
    anchors = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.]])
    reflected = np.diag([1., 1., -1., 1.])
    judged = api().judge_result(dict(accepted=True, transform=reflected, metrics={}, reasons=[]),
                               np.eye(4), anchors, success_threshold_mm=2)
    assert judged["d_mm"] == 0
    assert judged["registration_failed"] is True


def test_empty_acceptance_retains_rejected_denominator():
    result = api().summarize_audit([row(accepted=False)])
    assert result["n_patients"] == 1
    assert result["accepted_patient_risk"] is None
    assert result["risk_upper_bound"] == 1
    assert result["meets_risk_target"] is False


def test_hidden_reference_inputs_use_explicit_allowlist():
    case = dict(image=np.zeros((2, 2, 2)), affine=np.eye(4), points=np.ones((8, 3)),
                anchors=np.ones((8, 3)), candidates=np.eye(4)[None],
                transform=np.eye(4), label=np.ones((2, 2, 2)), point_weights=np.ones(8),
                another_reference=np.eye(4))
    hidden = api().hidden_case(case)
    assert set(hidden) == {"image", "affine", "points", "anchors", "candidates"}
    assert "transform" in case  # do not erase the independent evaluator's reference


def test_audit_rejects_duplicate_records_and_mixed_reference_standards():
    with pytest.raises(ValueError, match="duplicate"):
        api().summarize_audit([row(), row()])
    with pytest.raises(ValueError, match="manual"):
        api().summarize_audit([row(), row(patient="p2", case="c2") | {"reference_kind": "silver"}])


def test_frozen_audit_rejects_checkpoint_selection_and_policy_development_overlap():
    selected = [row() | {"content_hash": "h1"}]
    saved = dict(training_patient_ids=[], training_content_hashes=[], training_sources=[],
                 provenance_schema_version=2, selection_patient_ids=["p1"], selection_content_hashes=[], selection_sources=[])
    policy = dict(development_patient_ids=[], development_content_hashes=[], development_sources=[])
    with pytest.raises(ValueError, match="selection"):
        api().check_audit_cohort(selected, saved, purpose="frozen", split="test", policy=policy)
    saved["selection_patient_ids"] = []
    policy["development_patient_ids"] = ["p1"]
    with pytest.raises(ValueError, match="development"):
        api().check_audit_cohort(selected, saved, purpose="frozen", split="test", policy=policy)
    # Val remains useful, explicitly descriptive even if it selected the checkpoint.
    api().check_audit_cohort(selected, saved, purpose="development", split="val")
    del saved["selection_patient_ids"]
    with pytest.raises(ValueError, match="provenance"):
        api().check_audit_cohort(selected, saved, purpose="frozen", split="test", policy=policy)


def test_freeze_retains_all_explored_development_cohorts_and_rejects_partial_run(tmp_path):
    from task2reg.journal.runtime import canonical_hash
    def development(patient, status="complete"):
        return dict(format="rtr-verification-audit-v1", purpose="development", status=status,
                    protocol={"fixed": True}, cohort=dict(patient_ids=[patient], content_hashes=[], sources=["s"]))
    chosen, extra = tmp_path / "chosen.json", tmp_path / "extra.json"
    chosen.write_text(json.dumps(development("p1")))
    extra.write_text(json.dumps(development("p2")))
    policy = api().freeze_policy(chosen, tmp_path / "policy.json", additional_development_audits=[extra])
    assert policy["development_patient_ids"] == ["p1", "p2"]
    assert policy["policy_sha256"] == canonical_hash({k: v for k, v in policy.items() if k != "policy_sha256"})
    with pytest.raises(FileExistsError):
        api().freeze_policy(chosen, tmp_path / "policy.json")
    chosen.write_text(json.dumps(development("p1", status="running")))
    with pytest.raises(ValueError, match="complete"):
        api().freeze_policy(chosen, tmp_path / "partial.json")


def test_collection_hides_reference_before_judging_a_consistently_wrong_teacher(tmp_path, monkeypatch):
    from test_journal_engine import synthetic_manifest
    module = api()
    manifest = synthetic_manifest(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture; model loading is replaced below")
    monkeypatch.setattr(module, "load_field", lambda *a: (object(), {"training_patient_ids": []}))
    def wrong_teacher(model, case, record, device, **settings):
        assert set(case) == {"image", "affine", "points", "anchors", "candidates"}
        assert "reference_kind" not in record and "npz_path" not in record
        transform = np.eye(4)
        transform[0, 3] = 3
        return dict(accepted=True, reasons=[], transform=transform, metrics={"u95_mm": 0.})
    monkeypatch.setattr(module, "verify_case", wrong_teacher)
    result = module.audit_verifier(manifest, checkpoint, tmp_path / "audit", device="cpu")
    assert result["records"][0]["d_mm"] == pytest.approx(3.)
    assert result["summary"]["accepted_patient_risk"] == 1
    assert result["supports_frozen_risk_target"] is False
    assert json.loads((tmp_path / "audit/audit.json").read_text())["status"] == "complete"
