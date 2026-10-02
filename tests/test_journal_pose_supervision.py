import math

import pytest
import torch
from torch.autograd.function import once_differentiable


def loss_function():
    from task2reg.journal.pose_supervision import pose_secant_loss
    return pose_secant_loss


def geometry(parity=1., scale=1.):
    points = scale * torch.tensor([[[4., 0., 0.], [-4., 0., 0.], [0., 2., 0.],
                                    [0., -2., 0.], [0., 0., 1.], [0., 0., -1.]]], dtype=torch.float64)
    # Separate anchors ensure the optimizer's centroid/radius come from points.
    anchors = torch.tensor([[[3., 1., 0.], [-2., -1., 1.], [0., 2., -2.], [1., 0., 3.]]], dtype=torch.float64)
    reference = torch.eye(4, dtype=torch.float64)[None]
    reference[:, 0, 0] = parity
    transforms = reference.repeat(2, 1, 1)
    transforms[:, :3, 3] = torch.tensor([[2., -1., .5], [-1., 3., 0.]], dtype=torch.float64)
    return transforms, reference, points, anchors


def rms_potential(anchors, reference, smoothing=1., offset=0.):
    target = anchors @ reference[0, :3, :3].T + reference[0, :3, 3]

    def energy(transforms):
        moved = anchors @ transforms[:, :3, :3].transpose(-1, -2) + transforms[:, None, :3, 3]
        return ((moved - target).square().sum(-1).mean(-1) + smoothing ** 2).sqrt() - smoothing + offset
    return energy


@pytest.mark.parametrize("parity", [1., -1.])
@pytest.mark.parametrize("step", [.5, 4.])
def test_exact_finite_scale_rms_potential_matches_all_targets_despite_energy_offset(parity, step):
    transforms, reference, points, anchors = geometry(parity)
    energy = rms_potential(anchors, reference, smoothing=.2, offset=7.)
    loss, metrics = loss_function()(energy, transforms, reference, points, anchors,
                                    step_mm=step, smoothing_mm=.2)
    assert float(loss) < 1e-20
    assert float(metrics["secant"]) < 1e-20
    assert float(metrics["local_minimum"]) < 1e-20


class FirstOrderPotential(torch.autograd.Function):
    """A real differentiable energy whose backward deliberately has no double backward."""

    @staticmethod
    def forward(ctx, scale, values):
        ctx.save_for_backward(scale, values)
        return 20 + scale * values

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        scale, values = ctx.saved_tensors
        return (gradient * values).sum(), gradient * scale


def test_ordinary_parameter_updates_correct_an_initially_reversed_potential():
    transforms, reference, points, anchors = geometry()
    transforms.requires_grad_()
    reference.requires_grad_()
    anchors.requires_grad_()
    oracle = rms_potential(anchors.detach(), reference.detach())
    scale = torch.nn.Parameter(torch.tensor(-1., dtype=torch.float64))
    optimizer = torch.optim.Adam([scale], lr=.2)

    def energy(matrices):
        return FirstOrderPotential.apply(scale, oracle(matrices))

    near, far = reference.detach().clone(), reference.detach().clone()
    near[:, 0, 3], far[:, 0, 3] = 1., 2.
    assert float(energy(far) - energy(near)) < 0
    initial_loss = None
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_function()(energy, transforms, reference, points, anchors)
        initial_loss = float(loss.detach()) if initial_loss is None else initial_loss
        loss.backward()
        assert torch.isfinite(scale.grad)
        optimizer.step()
    final_loss, _ = loss_function()(energy, transforms, reference, points, anchors)
    assert float(final_loss) < initial_loss * .2
    assert float(energy(far) - energy(near)) > 0
    # Input geometry and labels are fixed observations, not optimization variables.
    assert transforms.grad is None
    assert reference.grad is None
    assert anchors.grad is None


def test_stationary_gt_maximum_is_penalized_by_positive_neighborhood_gaps():
    _, reference, points, anchors = geometry()
    oracle = rms_potential(anchors, reference)
    wrong = lambda matrices: 10 - oracle(matrices)
    without_minimum, _ = loss_function()(wrong, reference, reference, points, anchors,
                                         local_minimum_weight=0.)
    with_minimum, metrics = loss_function()(wrong, reference, reference, points, anchors,
                                            local_minimum_weight=1.)
    assert float(without_minimum) < 1e-20
    assert float(metrics["secant"]) < 1e-20
    assert float(metrics["local_minimum"]) > .01
    assert float(with_minimum) > .01


@pytest.mark.parametrize("parity", [1., -1.])
@pytest.mark.parametrize("scale", [.01, 1., 100.])
def test_probes_have_exact_physical_step_about_optimizer_point_centroid(parity, scale):
    transforms, reference, points, anchors = geometry(parity, scale)
    transforms = transforms[:1]
    reference = reference.clone()
    reference[:, 0, 3] = -20.
    observed = []
    oracle = rms_potential(anchors, reference)

    def energy(matrices):
        observed.append(matrices.detach())
        return oracle(matrices)

    step = .5
    loss_function()(energy, transforms, reference, points, anchors, step_mm=step)
    probes = torch.cat(observed)
    moved = points @ transforms[:, :3, :3].transpose(-1, -2) + transforms[:, None, :3, 3]
    center = moved.mean(1)
    probe_centers = (points @ probes[:, :3, :3].transpose(-1, -2) + probes[:, None, :3, 3]).mean(1)
    # Reference probes are centered 22 mm away, independently of the crown radius.
    candidate = probes[torch.linalg.vector_norm(probe_centers - center, dim=-1) < 1.]
    assert len(candidate) == 12
    relative_rotations = candidate[:, :3, :3] @ transforms[0, :3, :3].T
    traces = torch.diagonal(relative_rotations, dim1=-2, dim2=-1).sum(-1)
    angles = ((traces - 1) / 2).clamp(-1, 1).acos()
    rotating = angles > 1e-7
    assert int(rotating.sum()) == 6
    radius = max(float((moved - center[:, None]).square().sum(-1).mean().sqrt()), 1.)
    torch.testing.assert_close(angles[rotating], angles.new_full((6,), step / radius), atol=1e-10, rtol=1e-8)
    centroids = (points @ candidate[:, :3, :3].transpose(-1, -2) + candidate[:, None, :3, 3]).mean(1)
    torch.testing.assert_close(centroids[rotating], center.expand(6, -1), atol=1e-9, rtol=0)
    translation_lengths = torch.linalg.vector_norm(centroids[~rotating] - center, dim=-1)
    torch.testing.assert_close(translation_lengths, translation_lengths.new_full((6,), step), atol=1e-9, rtol=0)
    torch.testing.assert_close(torch.linalg.det(candidate[:, :3, :3]), candidate.new_full((12,), parity))


def test_origin_shift_and_explicit_batch_expansion_preserve_the_loss():
    transforms, reference, points, anchors = geometry(-1.)
    energy = lambda matrices: 12 - .7 * rms_potential(anchors, reference)(matrices)
    ordinary, _ = loss_function()(energy, transforms, reference, points, anchors)
    expanded, _ = loss_function()(energy, transforms, reference.expand(2, -1, -1),
                                  points.expand(2, -1, -1), anchors.expand(2, -1, -1))
    offset = transforms.new_tensor([10000., -20000., 8000.])
    shifted_transforms, shifted_reference = transforms.clone(), reference.clone()
    shifted_transforms[:, :3, 3] += offset
    shifted_reference[:, :3, 3] += offset
    shifted_energy = lambda matrices: 12 - .7 * rms_potential(anchors, shifted_reference)(matrices)
    shifted, _ = loss_function()(shifted_energy, shifted_transforms, shifted_reference, points, anchors)
    torch.testing.assert_close(ordinary, expanded, atol=1e-12, rtol=0)
    torch.testing.assert_close(ordinary, shifted, atol=1e-9, rtol=0)


def test_wrong_reference_parity_is_rejected_before_energy_evaluation():
    transforms, reference, points, anchors = geometry()
    reference[:, 0, 0] = -1
    with pytest.raises(ValueError, match="parity"):
        loss_function()(rms_potential(anchors, reference), transforms, reference, points, anchors)


@pytest.mark.parametrize("keyword,value", [("step_mm", 0.), ("step_mm", float("nan")),
                                           ("smoothing_mm", 0.), ("local_minimum_weight", -1.)])
def test_invalid_loss_scales_are_rejected(keyword, value):
    transforms, reference, points, anchors = geometry()
    with pytest.raises(ValueError):
        loss_function()(rms_potential(anchors, reference), transforms, reference, points, anchors,
                        **{keyword: value})


@pytest.mark.parametrize("bad_energy", [lambda matrices: torch.zeros(len(matrices), 1),
                                        lambda matrices: torch.full((len(matrices),), math.nan)])
def test_invalid_energy_outputs_are_rejected(bad_energy):
    transforms, reference, points, anchors = geometry()
    with pytest.raises(ValueError, match="energy"):
        loss_function()(bad_energy, transforms, reference, points, anchors)


def diagnostic_function():
    from task2reg.journal.pose_supervision import gradient_alignment_diagnostic
    return gradient_alignment_diagnostic


def energy_as_query(values):
    # registration_energy's pseudo-Huber fit becomes exactly ``values``;
    # fixed geometry coverage contributes only a constant offset.
    return {"distance": torch.zeros_like(values),
            "potential": ((values + .1).square() - .1 ** 2).sqrt()}


def diagnostic_geometry():
    points = torch.tensor([[[0., -1., -1.], [0., 1., -1.],
                             [0., -1., 1.], [0., 1., 1.]]], dtype=torch.float64)
    reference = torch.eye(4, dtype=torch.float64)[None]
    initial = reference.clone()
    initial[:, 0, 3] = 1.
    affine = reference.clone()
    affine[:, :3, 3] = -10.
    return points, initial, reference, points.clone(), affine, (21, 21, 21)


def test_autograd_diagnostic_detects_wrong_derivative_despite_positive_finite_secant():
    h = .5
    field = lambda xyz: energy_as_query(2 + xyz[..., 0] - .1 * torch.sin(2 * torch.pi * xyz[..., 0] / h))
    x = torch.tensor([1.], dtype=torch.float64)
    potential = lambda coordinate: coordinate - .1 * torch.sin(2 * torch.pi * coordinate / h)
    assert float((potential(x + h) - potential(x - h)) / (2 * h)) == pytest.approx(1.)
    result = diagnostic_function()(field, *diagnostic_geometry())
    assert result["cosine"] == pytest.approx(-1.)
    assert result["dot"] < 0
    assert result["energy_norm"] > 0
    assert result["target_norm"] > 0
    assert result["descent"] is False


def test_gradient_diagnostic_reports_good_direction_and_excludes_reference_stationarity():
    parameter = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
    field = lambda xyz: energy_as_query(2 + parameter * xyz[..., 0].square())
    points, initial, reference, anchors, affine, shape = diagnostic_geometry()
    good = diagnostic_function()(field, points, initial, reference, anchors, affine, shape)
    stationary = diagnostic_function()(field, points, reference, reference, anchors, affine, shape)
    assert good["cosine"] == pytest.approx(1.)
    assert good["descent"] is True
    assert stationary["cosine"] is None
    assert stationary["descent"] is None
    assert stationary["target_norm"] == 0
    assert parameter.grad is None
    assert all(not isinstance(value, torch.Tensor) for value in good.values())


def test_gradient_whitening_uses_radius_and_optimizer_center_not_anchor_centroid():
    points = torch.tensor([[[20., 0., 0.], [-20., 0., 0.], [0., 20., 0.],
                             [0., -20., 0.], [0., 0., 20.], [0., 0., -20.]]], dtype=torch.float64)
    anchors = points + points.new_tensor([0., 40., 0.])
    reference = torch.eye(4, dtype=torch.float64)[None]
    initial, affine = reference.clone(), reference.clone()
    initial[:, 0, 3], affine[:, :3, 3] = 1., -100.
    field = lambda xyz: energy_as_query(100 + xyz[..., 0])
    result = diagnostic_function()(field, points, initial, reference, anchors, affine, (201, 201, 201))
    # Smooth RMS target: translation gradient=1/sqrt(2); torque=-40/sqrt(2).
    # rho=20 gives whitened target norm sqrt(2+.5), not raw torque magnitude.
    assert result["energy_norm"] == pytest.approx(1.)
    assert result["target_norm"] == pytest.approx(math.sqrt(2.5))
    assert result["dot"] == pytest.approx(1 / math.sqrt(2))
    assert result["cosine"] == pytest.approx(1 / math.sqrt(5))


def test_zero_field_gradient_is_stalled_not_a_missing_target_comparison():
    result = diagnostic_function()(lambda xyz: torch.ones_like(xyz[..., 0]), *diagnostic_geometry())
    assert result["target_norm"] > 0
    assert result["energy_norm"] == 0
    assert result["dot"] == 0
    assert result["cosine"] is None
    assert result["descent"] is False
