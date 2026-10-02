"""Physical coordinates, differentiability and optimization behavior of journal fields."""
import pytest
import torch

from task2reg.journal.field import (
    RegistrationField,
    centered_increment,
    eikonal_loss,
    fixed_point_error,
    pairwise_rank_loss,
    refine_transform,
    registration_energy,
    roi_outside_distance,
    sample_features,
    transform_points,
    weighted_huber_loss,
)


def test_affine_sampling_has_correct_axis_order_and_world_gradient():
    # f(i,j,k)=i+10j+100k makes swapped tensor axes visible.
    ii, jj, kk = torch.meshgrid(
        torch.arange(4.), torch.arange(5.), torch.arange(6.), indexing="ij"
    )
    volume = (ii + 10 * jj + 100 * kk)[None, None].double().requires_grad_()
    affine = torch.tensor([[[0., -3., .2, 100.], [2., 0., 0., -70.],
                            [0., 0., 4., 9.], [0., 0., 0., 1.]]], dtype=torch.double)
    # voxel (1.2,2.3,3.4) -> world (93.78,-67.6,22.6).
    query = torch.tensor([[[93.78, -67.6, 22.6]]], dtype=torch.double, requires_grad=True)
    value = sample_features(volume, query, affine)
    torch.testing.assert_close(value, torch.tensor([[[364.2]]], dtype=torch.double))
    gradient = torch.autograd.grad(value.sum(), query, create_graph=True)[0]
    torch.testing.assert_close(
        gradient, torch.tensor([[[-10 / 3, .5, 25 + 1 / 6]]], dtype=torch.double)
    )
    gradient.square().sum().backward()
    assert volume.grad is not None and torch.isfinite(volume.grad).all()
    assert volume.grad.abs().sum() > 0


def test_sampler_handles_coarse_features_and_singleton_axes():
    volume = torch.tensor([[[[[0., 8.]], [[4., 12.]]]]])  # I=2,J=1,K=2
    affine = torch.eye(4)[None]
    query = torch.tensor([[[2., 0., 4.]]])
    sampled = sample_features(volume, query, affine, input_shape=(5, 1, 9))
    torch.testing.assert_close(sampled, torch.tensor([[[6.]]]))


def test_roi_penalty_uses_physical_millimeters_and_does_not_make_outside_attractive():
    affine = torch.diag(torch.tensor([2., 3., 4., 1.]))[None]
    points = torch.tensor([[[-2., 0., 0.], [6., 3., 4.], [2., 3., 4.]]], requires_grad=True)
    outside = roi_outside_distance(points, affine, (3, 3, 3))
    torch.testing.assert_close(outside, torch.tensor([[2., 2., 0.]]))
    energy, diagnostic = registration_energy(torch.zeros(1, 3), points, affine, (3, 3, 3))
    assert energy.item() > 2.
    torch.testing.assert_close(diagnostic["outside_fraction"], torch.tensor([2 / 3]))
    energy.sum().backward()
    assert points.grad[0, 0, 0] < 0 and points.grad[0, 1, 0] > 0


def test_weighted_huber_ignores_unknown_pseudo_points_and_zero_weight_is_finite():
    prediction = torch.tensor([1., 200.], requires_grad=True)
    target = torch.zeros(2)
    observed = weighted_huber_loss(prediction, target, torch.tensor([1., 0.]))
    torch.testing.assert_close(observed, torch.tensor(.5))
    observed.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([1., 0.]))
    ignored = weighted_huber_loss(prediction, target, torch.zeros(2))
    assert ignored.item() == 0 and ignored.requires_grad


def test_rank_loss_prefers_true_order_and_ignores_indistinguishable_pairs():
    errors = torch.tensor([[.1, 2., 5.]])
    correct = pairwise_rank_loss(torch.tensor([[.1, 1., 3.]]), errors)
    incorrect = pairwise_rank_loss(torch.tensor([[3., 1., .1]]), errors)
    assert correct < incorrect
    energies = torch.tensor([[1., 2.]], requires_grad=True)
    ignored = pairwise_rank_loss(energies, torch.tensor([[.1, .2]]))
    ignored.backward()
    assert ignored.item() == 0
    torch.testing.assert_close(energies.grad, torch.zeros_like(energies))


@pytest.mark.parametrize("parity", [1., -1.])
def test_centered_increment_preserves_parity_and_does_not_move_rotation_center(parity):
    initial = torch.eye(4, dtype=torch.double)[None]
    initial[:, 0, 0] = parity
    initial[:, :3, 3] = torch.tensor([1000., -500., 800.])
    center = initial[:, :3, 3].clone()
    twist = torch.tensor([[.1, -.05, .03, 0., 0., 0.]], dtype=torch.double)
    updated = centered_increment(initial, twist, center)
    torch.testing.assert_close(updated[:, :3, 3], center)
    torch.testing.assert_close(torch.linalg.det(updated[:, :3, :3]), torch.tensor([parity], dtype=torch.double))
    torch.testing.assert_close(updated[:, :3, :3] @ updated[:, :3, :3].transpose(-1, -2), torch.eye(3, dtype=torch.double)[None])


def test_centered_increment_has_finite_second_derivatives_at_zero_twist():
    twist = torch.zeros(1, 6, dtype=torch.double, requires_grad=True)
    transform = centered_increment(torch.eye(4, dtype=torch.double)[None], twist, torch.ones(1, 3, dtype=torch.double))
    first = torch.autograd.grad(transform.square().sum(), twist, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), twist)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()


@pytest.mark.parametrize("parity", [1., -1.])
def test_refinement_reduces_analytic_pose_error_and_preserves_parity(parity):
    points = torch.tensor([[[-2., -1., 0.], [2., -1., 0.], [0., 2., 0.]]])
    truth = torch.eye(4)[None]
    truth[:, 0, 0] = parity
    initial = truth.clone()
    initial[:, 2, 3] = 1.5
    affine = torch.eye(4)[None]
    affine[:, :3, 3] = -10.
    result = refine_transform(
        lambda xyz: xyz[..., 2].abs(), points, initial, affine, (21, 21, 21),
        steps=5, learning_rate=.3,
    )
    before = fixed_point_error(initial, truth, points)
    after = fixed_point_error(result["transform"], truth, points)
    assert after.item() < before.item() * .5
    assert result["final_energy"].item() < result["initial_energy"].item()
    assert torch.linalg.det(result["transform"][:, :3, :3]).item() == pytest.approx(parity)


def test_unrolled_refinement_backpropagates_pose_error_to_field_parameter():
    weight = torch.tensor(1., requires_grad=True)
    points = torch.tensor([[[-1., -1., 0.], [1., -1., 0.], [0., 2., 0.]]])
    initial = torch.eye(4)[None]
    initial[:, 2, 3] = 1.5
    affine = torch.eye(4)[None]
    affine[:, :3, 3] = -10.
    result = refine_transform(
        lambda xyz: weight * xyz[..., 2].square(), points, initial, affine, (21, 21, 21),
        steps=2, learning_rate=.1, differentiable=True,
    )
    fixed_point_error(result["transform"], torch.eye(4)[None], points).sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad) and weight.grad.abs() > 0


def test_refinement_is_invariant_to_world_origin_shift():
    points = torch.tensor([[[-1., -1., 0.], [1., -1., 0.], [0., 2., 0.]]], dtype=torch.double)
    initial = torch.eye(4, dtype=torch.double)[None]
    initial[:, 2, 3] = 1.5
    affine = torch.eye(4, dtype=torch.double)[None]
    affine[:, :3, 3] = -10.
    offset = torch.tensor([1234., -321., 456.], dtype=torch.double)
    shifted_initial = initial.clone()
    shifted_initial[:, :3, 3] += offset
    shifted_affine = affine.clone()
    shifted_affine[:, :3, 3] += offset
    ordinary = refine_transform(lambda xyz: xyz[..., 2].square(), points, initial, affine,
                                (21, 21, 21), steps=2, differentiable=True)
    shifted = refine_transform(lambda xyz: (xyz[..., 2] - offset[2]).square(), points,
                               shifted_initial, shifted_affine, (21, 21, 21),
                               steps=2, differentiable=True)
    torch.testing.assert_close(shifted["transform"][:, :3, :3], ordinary["transform"][:, :3, :3])
    torch.testing.assert_close(shifted["transform"][:, :3, 3] - offset, ordinary["transform"][:, :3, 3])


def test_two_step_actual_network_refinement_has_finite_parameter_gradients():
    torch.manual_seed(7)
    model = RegistrationField(base_channels=4)
    context = model.encode(torch.randn(1, 1, 16, 16, 16))
    affine = torch.eye(4)[None]
    points = torch.tensor([[[4.2, 5.3, 6.4], [7.2, 8.3, 9.4], [5.2, 9.3, 7.4]]])
    initial = torch.eye(4)[None]
    initial[:, 2, 3] = .5
    result = refine_transform(
        lambda xyz: model.query(context, xyz, affine, torch.tensor([0])),
        points, initial, affine, (16, 16, 16), steps=2, differentiable=True,
    )
    fixed_point_error(result["transform"], torch.eye(4)[None], points).sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


def test_refinement_rejects_nonhomogeneous_input_matrix():
    transform = torch.eye(4)[None]
    transform[:, 3, 0] = .2
    with pytest.raises(ValueError, match="homogeneous"):
        refine_transform(lambda xyz: xyz[..., 0].abs(), torch.zeros(1, 3, 3), transform,
                         torch.eye(4)[None], (16, 16, 16), steps=0)


def test_eikonal_penalizes_wrong_physical_gradient_away_from_unsigned_cusp():
    points = torch.tensor([[[1., 0., 0.], [2., 0., 0.]]], requires_grad=True)
    scale = torch.tensor(2., requires_grad=True)
    distances = scale * points[..., 0]
    loss = eikonal_loss(distances, points, torch.tensor([[1., 2.]]))
    torch.testing.assert_close(loss, torch.tensor(1.))
    loss.backward()
    assert scale.grad.item() == pytest.approx(2.)


def test_eikonal_excludes_clamped_outside_queries_with_impossible_unit_gradient():
    points = torch.tensor([[[-1., 0., 0.], [1., 0., 0.]]], requires_grad=True)
    distances = points[..., 0].clamp_min(0)
    loss = eikonal_loss(distances, points, torch.ones(1, 2), valid_mask=torch.tensor([[False, True]]))
    assert loss.item() == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("mode", ["dense", "implicit"])
def test_network_retains_support_logits_and_query_gradients(mode):
    torch.manual_seed(12)
    model = RegistrationField(base_channels=4, mode=mode)
    image = torch.randn(1, 1, 16, 16, 16)
    context = model.encode(image)
    assert context["logits"].shape == (1, 3, 16, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(context["logits"], model.backbone(image))
    points = torch.tensor([[[3.2, 5.4, 6.7], [8.3, 9.1, 10.6]]], requires_grad=True)
    distance = model.query(context, points, torch.eye(4)[None], torch.tensor([1]))
    assert distance.shape == (1, 2) and torch.all(distance > 0)
    gradient = torch.autograd.grad(distance.sum(), points, create_graph=True)[0]
    (gradient.square().sum() + distance.mean()).backward()
    assert torch.isfinite(gradient).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters())


def test_network_rejects_invalid_jaw_and_ranking_rejects_nonfinite_errors():
    model = RegistrationField(base_channels=4)
    context = model.encode(torch.zeros(1, 1, 16, 16, 16))
    with pytest.raises(ValueError, match="jaw"):
        model.query(context, torch.zeros(1, 1, 3), torch.eye(4)[None], torch.tensor([2]))
    with pytest.raises(ValueError, match="finite"):
        pairwise_rank_loss(torch.zeros(1, 2), torch.tensor([[0., float("nan")]]))


def test_implicit_query_accepts_float64_physical_metadata_with_float32_model():
    model = RegistrationField(base_channels=4)
    context = model.encode(torch.zeros(1, 1, 16, 16, 16))
    points = torch.tensor([[[1.2, 3.4, 5.6]]], dtype=torch.double, requires_grad=True)
    distances = model.query(context, points, torch.eye(4, dtype=torch.double)[None], torch.tensor([0]))
    gradient = torch.autograd.grad(distances.sum(), points)[0]
    assert torch.isfinite(gradient).all()
