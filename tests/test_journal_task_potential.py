import pytest
import torch

from task2reg.journal.field import RegistrationField, registration_energy, refine_transform


@pytest.mark.parametrize("mode", ["implicit", "dense"])
def test_task_head_starts_as_geometry_but_can_change_without_changing_distance(mode):
    torch.manual_seed(1)
    model = RegistrationField(4, mode, energy_mode="task")
    image = torch.randn(1, 1, 16, 16, 16)
    context = model.encode(image)
    points = torch.tensor([[[5.2, 7.1, 8.4], [3., 4., 9.]]])
    affine, jaw = torch.eye(4)[None], torch.tensor([0])
    initial = model.registration_query(context, points, affine, jaw)
    torch.testing.assert_close(initial["distance"], initial["potential"])
    with torch.no_grad():
        model.task_head.bias.add_(2)
    changed = model.registration_query(model.encode(image), points, affine, jaw)
    torch.testing.assert_close(initial["distance"], changed["distance"])
    assert torch.all(changed["potential"] > initial["potential"])


def test_task_energy_uses_task_values_but_geometric_coverage():
    points = torch.ones(1, 10, 3)
    distance = torch.zeros(1, 10)
    potential = torch.full((1, 10), 3.)
    energy, metrics = registration_energy({"distance": distance, "potential": potential},
                                           points, torch.eye(4)[None], (16, 16, 16))
    assert metrics["coverage"].item() > .99
    assert energy.item() > 2.9
    assert metrics["distance_energy"].item() == pytest.approx(0)


def test_refiner_follows_task_potential_with_flat_geometry_distance():
    points = torch.tensor([[[4., 4., 4.], [5., 4., 4.], [4., 5., 4.]]])
    initial = torch.eye(4)[None]
    initial[:, 2, 3] = 2
    def query(xyz):
        return {"distance": xyz[..., 0] * 0, "potential": (xyz[..., 2] - 4.).square()}
    result = refine_transform(query, points, initial, torch.eye(4)[None], (16, 16, 16), steps=4, learning_rate=.2)
    assert abs(result["transform"][0, 2, 3]) < .5
    assert result["final_energy"].item() < result["initial_energy"].item()


@pytest.mark.parametrize("mode", ["implicit", "dense"])
def test_distance_checkpoint_upgrade_preserves_initial_registration_energy(mode):
    source = RegistrationField(4, mode)
    dual = RegistrationField(4, mode, energy_mode="task")
    dual.load_distance_initialization(source.state_dict())
    image = torch.randn(1, 1, 16, 16, 16)
    points, affine, jaw = torch.rand(1, 10, 3) * 12, torch.eye(4)[None], torch.tensor([0])
    before = source.query(source.encode(image), points, affine, jaw)
    after = dual.registration_query(dual.encode(image), points, affine, jaw)
    torch.testing.assert_close(before, after["distance"])
    torch.testing.assert_close(before, after["potential"])


def test_dual_head_secant_training_and_inference_use_saved_task_architecture(tmp_path):
    import json
    from test_journal_engine import synthetic_manifest, tiny_config
    from task2reg.journal.engine import train
    from task2reg.journal.inference import predict, load_field
    manifest = synthetic_manifest(tmp_path)
    train(manifest, tmp_path / "teacher", tiny_config())
    teacher = tmp_path / "teacher/last.pt"
    config = tiny_config(energy_mode="task", secant_weight=.2, secant_points=16,
                         secant_cached_candidates=1, secant_reference_perturbations=1)
    train(manifest, tmp_path / "task", config, initialize_field=teacher)
    metrics = json.loads((tmp_path / "task/metrics.jsonl").read_text())
    assert metrics["loss_components"]["secant"] > 0
    model, _ = load_field(tmp_path / "task/last.pt", "cpu")
    assert model.energy_mode == "task"
    assert not torch.equal(model.task_head.weight, model.query_mlp[-1].weight)
    report = predict(manifest, tmp_path / "task/last.pt", tmp_path / "predictions", split="unlabeled",
                     device="cpu", refinement_steps=1, point_budget=16)
    assert not report["records"][0]["failed"]


def test_reference_only_secant_needs_no_cached_candidates_and_supports_small_perturbations(tmp_path):
    import numpy as np
    from test_journal_engine import synthetic_manifest, tiny_config
    from task2reg.journal.data import load_journal_manifest, load_case
    from task2reg.journal.engine import case_loss
    record = load_journal_manifest(synthetic_manifest(tmp_path))[0]
    case = load_case(record)
    case.pop("candidates")
    config = tiny_config(energy_mode="task", rank_weight=0., pose_weight=0., secant_weight=1.,
                         secant_cached_candidates=0, secant_reference_perturbations=1,
                         secant_perturbation_mm=.05, secant_points=16)
    model = RegistrationField(4, energy_mode="task")
    loss, metrics = case_loss(model, case, record, config, "cpu", np.random.default_rng(3), 0, training=True)
    loss.backward()
    assert np.isfinite(metrics["secant"]) and metrics["secant_cached_used"] == 0
    assert torch.isfinite(model.task_head.weight.grad).all()


def test_unrolled_pose_control_does_not_try_to_repair_opposite_parity(tmp_path):
    import numpy as np
    from test_journal_engine import synthetic_manifest, tiny_config
    from task2reg.journal.data import load_journal_manifest, load_case
    from task2reg.journal.engine import case_loss
    record = load_journal_manifest(synthetic_manifest(tmp_path))[0]
    case = load_case(record)
    candidates = np.repeat(np.eye(4)[None], 2, axis=0)
    candidates[:, 0, 0] = -1
    candidates[:, 0, 3] = 15  # small same-anchor displacement but wrong parity
    case["candidates"] = candidates
    config = tiny_config(rank_weight=0., pose_weight=.1)
    _, metrics = case_loss(RegistrationField(4), case, record, config, "cpu", np.random.default_rng(1), 0, training=True)
    assert "pose" not in metrics
