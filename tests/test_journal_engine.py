import json

import numpy as np
import pytest
import torch

from task2reg.journal.engine import TrainingConfig, train, predict
from task2reg.journal.engine import validation_score, select_pseudo_records
from task2reg.journal.runtime import runtime_metadata, restore_scaler
from task2reg.journal.field import RegistrationField
from task2reg.journal.data import load_journal_manifest, load_case


def synthetic_manifest(tmp_path):
    rng = np.random.default_rng(3)
    points = rng.normal(size=(48, 3)).astype(np.float32)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points = points * 2 + 7.5
    records = []
    for index, split in enumerate(("train", "val", "unlabeled")):
        path = tmp_path / f"{split}.npz"
        candidates = np.repeat(np.eye(4)[None], 3, axis=0)
        candidates[1, 0, 3] = 2
        candidates[2, 1, 3] = -3
        image = np.full((16, 16, 16), 500 + index, np.int16)
        label = np.zeros_like(image, dtype=np.uint8)
        label[6:10, 6:10, 6:10] = 1
        payload = dict(image=image, affine=np.eye(4), points=points,
                       anchors=points.copy(), candidates=candidates)
        if split != "unlabeled":
            payload.update(transform=np.eye(4), label=label)
        np.savez_compressed(path, **payload)
        records.append(dict(case_id=f"synthetic:{index}", patient_id=f"p{index}",
                            source="synthetic", split=split, jaw="upper",
                            npz_path=path.name,
                            reference_kind="none" if split == "unlabeled" else "manual",
                            candidate_provenance={"kind": "geometry_only"}))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(schema_version=1, records=records)))
    return path


def tiny_config(**overrides):
    values = dict(base_channels=4, epochs=1, queries=32, candidate_points=24,
                  max_candidates=3, device="cpu", amp=False, rank_weight=.1,
                  rank_warmup_epochs=0, eikonal_weight=0., pose_weight=0.,
                  ssl_strategy="none", seed=11)
    values.update(overrides)
    return TrainingConfig(**values)


def test_training_resume_matches_uninterrupted_and_inference_has_no_gt(tmp_path):
    torch.set_num_threads(2)
    manifest = synthetic_manifest(tmp_path)
    first = train(manifest, tmp_path / "resume", tiny_config())
    assert first["epochs_completed"] == 1
    last = tmp_path / "resume" / "last.pt"
    train(manifest, tmp_path / "resume", tiny_config(epochs=2), resume=last)
    train(manifest, tmp_path / "full", tiny_config(epochs=2))
    resumed = torch.load(last, weights_only=False, map_location="cpu")
    full = torch.load(tmp_path / "full" / "last.pt", weights_only=False, map_location="cpu")
    assert resumed["global_step"] == full["global_step"] == 2
    for name, value in resumed["model"].items():
        torch.testing.assert_close(value, full["model"][name], rtol=0, atol=0)
    result = predict(manifest, last, tmp_path / "predictions", split="unlabeled",
                     device="cpu", refinement_steps=1)
    assert len(result["records"]) == 1
    row = result["records"][0]
    assert "reference_transform" not in row
    assert np.isfinite(row["candidates"][0]["field_energy"])
    assert np.linalg.det(np.array(row["candidates"][0]["transform"])[:3, :3]) == pytest.approx(1.)


def test_training_rejects_changed_resume_configuration(tmp_path):
    torch.set_num_threads(2)
    manifest = synthetic_manifest(tmp_path)
    train(manifest, tmp_path / "run", tiny_config())
    with pytest.raises(ValueError, match="configuration"):
        train(manifest, tmp_path / "run", tiny_config(epochs=2, learning_rate=.01),
              resume=tmp_path / "run" / "last.pt")


def test_invalid_configuration_fails_before_training():
    with pytest.raises(ValueError):
        TrainingConfig(queries=0)
    with pytest.raises(ValueError):
        TrainingConfig(ssl_strategy="unknown")


def test_validation_score_independent_of_training_loss_weights(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    row = load_journal_manifest(manifest)[1]
    model = RegistrationField(base_channels=4)
    case = load_case(row)
    a = validation_score(model, case, row, tiny_config(rank_weight=0), "cpu")
    b = validation_score(model, case, row, tiny_config(rank_weight=10), "cpu")
    assert a == pytest.approx(b)


def test_pseudo_budget_distinguishes_same_case_id_across_sources():
    rows = [dict(case_id="1", source=source, jaw=jaw)
            for source in ("a", "b") for jaw in ("upper", "lower")]
    selected = select_pseudo_records(rows, 1)
    assert len(selected) == 2
    assert len({r["source"] for r in selected}) == 1


def test_runtime_captures_revision_and_scaler_can_enable_after_cpu_resume():
    assert runtime_metadata()["git_revision"] != "unavailable"
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    restore_scaler(scaler, {})
    assert scaler.is_enabled()


def test_initialization_requires_audited_provenance(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        train(manifest, tmp_path / "run", tiny_config(), initialize_support=tmp_path / "old.pt")


def test_failed_prediction_is_retained_for_evaluation(tmp_path, monkeypatch):
    import task2reg.journal.inference as inference
    manifest = synthetic_manifest(tmp_path)
    train(manifest, tmp_path / "run", tiny_config())
    def fail(*args, **kwargs):
        raise FloatingPointError("synthetic optimizer failure")
    monkeypatch.setattr(inference, "score_and_refine", fail)
    result = predict(manifest, tmp_path / "run" / "last.pt", tmp_path / "pred",
                     split="val", device="cpu", evaluate=True)
    assert result["records"][0]["failed"]
    report = json.loads((tmp_path / "pred" / "evaluation_input.json").read_text())
    assert report["records"][0]["failed"]
    assert report["records"][0]["reference_kind"] == "manual"


def test_field_initialization_checks_previous_training_patients(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    train(manifest, tmp_path / "teacher", tiny_config())
    payload = json.loads(manifest.read_text())
    for row in payload["records"]:
        if row["split"] == "train":
            row["split"] = "val"
        elif row["split"] == "val":
            row["split"] = "train"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="initialization"):
        train(manifest, tmp_path / "student", tiny_config(), initialize_field=tmp_path / "teacher" / "last.pt")


@pytest.mark.parametrize("strategy", ["verified_field", "legacy_support"])
def test_ssl_student_trains_from_teacher_and_counts_pseudo_provenance(tmp_path, strategy):
    from task2reg.journal.pseudo import assemble_pseudo_manifest
    manifest = synthetic_manifest(tmp_path)
    train(manifest, tmp_path / "teacher", tiny_config())
    records = load_journal_manifest(manifest)
    unlabeled = records[-1]
    evidence = dict(accepted=True, reasons=[], transform=np.eye(4), point_weights=np.ones(48), metrics={"u95_mm":0.})
    combined = assemble_pseudo_manifest(records, [(unlabeled,evidence)], tmp_path / "pseudo",
                                        teacher_id="synthetic", excluded_patient_ids=["p1"])
    if strategy == "legacy_support":
        payload = json.loads(combined.read_text())
        row = next(r for r in payload["records"] if r["split"] == "pseudo")
        row["pseudo_kind"] = "support"
        path = combined.parent / row["npz_path"]
        with np.load(path) as archive:
            case = {k:archive[k] for k in archive.files if k not in {"transform","point_weights"}}
        labels = np.zeros(case["image"].shape, np.uint8)
        labels[6:10,6:10,6:10] = 1
        np.savez_compressed(path, **case, label=labels)
        combined.write_text(json.dumps(payload))
    result = train(combined, tmp_path / "student", tiny_config(ssl_strategy=strategy),
                   initialize_field=tmp_path / "teacher/last.pt")
    checkpoint = torch.load(result["checkpoint"],weights_only=False,map_location="cpu")
    assert checkpoint["training_patient_ids"] == ["p0","p2"]
    metrics = json.loads((tmp_path / "student/metrics.jsonl").read_text().splitlines()[0])
    assert np.isfinite(metrics["loss_components"]["pseudo"])
    assert result["global_step"] == 1


def test_candidate_scoring_bounds_query_batch_without_changing_energy():
    from task2reg.journal.engine import candidate_energies
    from task2reg.journal.field import registration_energy, transform_points
    class AnalyticField:
        def __init__(self):
            self.counts = []
        def query(self, context, points, affine, jaw):
            self.counts.append(points.shape[1])
            return torch.linalg.vector_norm(points, dim=-1)
    model = AnalyticField()
    points = torch.ones(1, 12, 3)
    candidates = torch.eye(4).repeat(21,1,1)
    candidates[:,0,3] = torch.arange(21)
    affine = torch.eye(4)[None]
    energy, _ = candidate_energies(model, {}, points, candidates, affine, torch.tensor([0]), (32,32,32))
    moved = transform_points(points.expand(21,-1,-1), candidates)
    expected, _ = registration_energy(torch.linalg.vector_norm(moved,dim=-1), moved, affine.expand(21,-1,-1), (32,32,32))
    torch.testing.assert_close(energy, expected)
    assert max(model.counts) <= 8 * points.shape[1]


def initialization_checkpoint(path, **overrides):
    config = tiny_config()
    payload = dict(format="rtr-journal-field-v1", config=config.as_dict(),
                   model=RegistrationField(4).state_dict(), provenance_schema_version=2,
                   training_patient_ids=["ancestor-train"], training_sources=["ancestor-source"],
                   training_content_hashes=["ancestor-training-content"],
                   excluded_patient_ids=["p1", "p2"],
                   selection_patient_ids=["ancestor-val"],
                   selection_sources=["ancestor-selection-source"],
                   selection_content_hashes=["ancestor-validation-content"])
    payload.update(overrides)
    torch.save(payload, path)
    return payload


def test_checkpoint_retains_selection_history_through_resume_and_initialization(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["records"][1]["content_hash"] = "current-validation-content"
    manifest.write_text(json.dumps(payload))
    parent = tmp_path / "parent.pt"
    initialization_checkpoint(parent)
    train(manifest, tmp_path / "run", tiny_config(), initialize_field=parent)
    last = tmp_path / "run/last.pt"
    train(manifest, tmp_path / "run", tiny_config(epochs=2), resume=last)
    train(manifest, tmp_path / "continuation", tiny_config(), initialize_field=last)
    for directory in ("run", "continuation"):
        saved = torch.load(tmp_path / directory / "last.pt", weights_only=False, map_location="cpu")
        provenance = json.loads((tmp_path / directory / "provenance.json").read_text())
        for source in (saved, provenance):
            assert source["provenance_schema_version"] == 2
            assert source["selection_patient_ids"] == ["ancestor-val", "p1"]
            assert source["selection_sources"] == ["ancestor-selection-source", "synthetic"]
            assert source["selection_content_hashes"] == ["ancestor-validation-content", "current-validation-content"]


@pytest.mark.parametrize("split", ["test", "external_test"])
@pytest.mark.parametrize("overlap", ["patient", "content"])
def test_initialization_cannot_promote_ancestor_selection_to_locked_test(tmp_path, split, overlap):
    manifest = synthetic_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    locked = payload["records"][-1]
    locked.update(split=split, reference_kind="manual", source="locked-source",
                  content_hash="selected-content")
    manifest.write_text(json.dumps(payload))
    parent = tmp_path / "parent.pt"
    initialization_checkpoint(parent,
                              selection_patient_ids=["p2" if overlap == "patient" else "another-patient"],
                              selection_content_hashes=["selected-content" if overlap == "content" else "other-content"])
    with pytest.raises(ValueError, match="selection"):
        train(manifest, tmp_path / "student", tiny_config(), initialize_field=parent)
    assert not (tmp_path / "student/last.pt").exists()


def test_external_source_cannot_have_selected_ancestor_model_on_different_patients(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["records"][-1].update(split="external_test", reference_kind="manual", source="locked-source")
    manifest.write_text(json.dumps(payload))
    parent = tmp_path / "parent.pt"
    initialization_checkpoint(parent, selection_sources=["locked-source"])
    with pytest.raises(ValueError, match="selection.*source"):
        train(manifest, tmp_path / "student", tiny_config(), initialize_field=parent)


def test_selection_source_history_cannot_be_silently_omitted(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    parent = tmp_path / "parent.pt"
    payload = initialization_checkpoint(parent)
    payload.pop("selection_sources")
    torch.save(payload, parent)
    with pytest.raises(ValueError, match="selection_sources"):
        train(manifest, tmp_path / "student", tiny_config(), initialize_field=parent)


def test_legacy_initialization_needs_an_explicit_hash_bound_selection_audit(tmp_path):
    from task2reg.journal.runtime import sha256_file
    manifest = synthetic_manifest(tmp_path)
    parent = tmp_path / "legacy.pt"
    payload = initialization_checkpoint(parent)
    for key in ("provenance_schema_version", "selection_patient_ids", "selection_sources", "selection_content_hashes"):
        payload.pop(key)
    torch.save(payload, parent)
    with pytest.raises(ValueError, match="selection"):
        train(manifest, tmp_path / "unknown", tiny_config(), initialize_field=parent)
    audit = {key: value for key, value in payload.items() if key not in {"model", "config", "format"}}
    audit.update(provenance_schema_version=2, selection_patient_ids=["audited-val"],
                 selection_sources=["audited-source"], selection_content_hashes=[], checkpoint_sha256="wrong-checkpoint")
    sidecar = tmp_path / "audit.json"
    sidecar.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        train(manifest, tmp_path / "wrong-audit", tiny_config(), initialize_field=parent,
              initialization_provenance=sidecar)
    audit["checkpoint_sha256"] = sha256_file(parent)
    sidecar.write_text(json.dumps(audit))
    train(manifest, tmp_path / "audited", tiny_config(), initialize_field=parent,
          initialization_provenance=sidecar)
    saved = torch.load(tmp_path / "audited/last.pt", weights_only=False, map_location="cpu")
    assert saved["selection_patient_ids"] == ["audited-val", "p1"]


def test_resume_rejects_unknown_selection_history(tmp_path):
    manifest = synthetic_manifest(tmp_path)
    train(manifest, tmp_path / "run", tiny_config())
    checkpoint = tmp_path / "run/last.pt"
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    for key in ("provenance_schema_version", "selection_patient_ids", "selection_sources", "selection_content_hashes"):
        saved.pop(key, None)
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="selection"):
        train(manifest, tmp_path / "run", tiny_config(epochs=2), resume=checkpoint)


def test_refined_validation_distinguishes_fields_with_equal_raw_candidate_quality():
    class AnalyticField:
        def __init__(self, stationary):
            self.stationary = stationary

        def encode(self, image):
            return {"features": (), "logits": image, "input_shape": image.shape[2:]}

        def query(self, context, points, affine, jaw):
            distance = (points[..., 0] - 5).abs()
            if self.stationary:
                distance = distance - torch.sin(2 * torch.pi * distance) / (2 * torch.pi)
            return distance

    points = np.array([[5, 5, 5], [5, 6, 5], [5, 5, 6], [5, 6, 6]], dtype=np.float32)
    candidates = np.repeat(np.eye(4)[None], 2, axis=0)
    candidates[:, 0, 3] = [1, 3]
    case = dict(image=np.zeros((16, 16, 16)), affine=np.eye(4), points=points,
                anchors=points.copy(), transform=np.eye(4), candidates=candidates)
    record = dict(jaw="upper", case_id="analytic")
    raw = tiny_config(candidate_points=4)
    useful, stationary = AnalyticField(False), AnalyticField(True)
    assert validation_score(useful, case, record, raw, "cpu") == pytest.approx(1.)
    assert validation_score(stationary, case, record, raw, "cpu") == pytest.approx(1.)
    refined = tiny_config(validation_metric="refined_selected_D_mm", validation_refinement_steps=20,
                          validation_point_budget=4, validation_refinement_learning_rate=.25)
    assert validation_score(useful, case, record, refined, "cpu") < .001
    assert validation_score(stationary, case, record, refined, "cpu") == pytest.approx(1.)


def manifest_with_locked_case(tmp_path, split):
    manifest = synthetic_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    with np.load(tmp_path / "val.npz", allow_pickle=False) as archive:
        case = {key: archive[key].copy() for key in archive.files}
    case["image"] += 100
    np.savez_compressed(tmp_path / "locked.npz", **case)
    payload["records"].append(dict(case_id="locked:1", patient_id="p3", source="locked-source",
                                   split=split, jaw="upper", npz_path="locked.npz",
                                   reference_kind="manual", content_hash="locked-content",
                                   candidate_provenance={"kind": "geometry_only"}))
    manifest.write_text(json.dumps(payload))
    return manifest


@pytest.mark.parametrize("route", ["predict", "pseudo_export"])
@pytest.mark.parametrize("split,overlap", [
    ("test", "patient"), ("test", "content"), ("test", "unknown"),
    ("external_test", "patient"), ("external_test", "content"), ("external_test", "source"),
])
def test_locked_routes_reject_checkpoint_selection_exposure(tmp_path, route, split, overlap):
    from task2reg.journal.pseudo import export_verified_pseudo
    manifest = manifest_with_locked_case(tmp_path, split)
    checkpoint = tmp_path / "teacher.pt"
    saved = initialization_checkpoint(checkpoint, excluded_patient_ids=["p1", "p3"])
    if overlap == "unknown":
        saved.pop("provenance_schema_version")
    else:
        key, value = {"patient": ("selection_patient_ids", "p3"),
                      "content": ("selection_content_hashes", "locked-content"),
                      "source": ("selection_sources", "locked-source")}[overlap]
        saved[key] = [value]
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="selection"):
        if route == "predict":
            predict(manifest, checkpoint, tmp_path / "predictions", split=split,
                    device="cpu", refinement_steps=0)
        else:
            export_verified_pseudo(manifest, checkpoint, tmp_path / "pseudo", device="cpu",
                                   starts=2, sectors=2, refinement_steps=0, point_budget=8)


def test_legacy_checkpoint_remains_readable_for_development_and_unlabeled_export(tmp_path):
    from task2reg.journal.pseudo import export_verified_pseudo
    manifest = synthetic_manifest(tmp_path)
    checkpoint = tmp_path / "legacy.pt"
    saved = initialization_checkpoint(checkpoint)
    for key in ("provenance_schema_version", "selection_patient_ids", "selection_content_hashes", "selection_sources"):
        saved.pop(key)
    torch.save(saved, checkpoint)
    result = predict(manifest, checkpoint, tmp_path / "development", split="val",
                     device="cpu", refinement_steps=0)
    assert not result["records"][0]["failed"]
    exported = export_verified_pseudo(manifest, checkpoint, tmp_path / "pseudo", device="cpu",
                                      starts=2, sectors=2, refinement_steps=0, point_budget=8)
    assert exported.exists()
