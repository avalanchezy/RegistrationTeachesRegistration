"""Physical finite-scale supervision of a registration energy.

The target is smooth same-anchor RMS displacement, not surface distance or
independent-landmark TRE. These finite differences constrain an energy at the
probe scale; they do not guarantee its infinitesimal gradients, global
convexity, or identifiability for symmetric unordered point sets.
"""
from collections.abc import Callable
import math

import torch
from torch.nn import functional as F

from .field import centered_increment, registration_energy, transform_points


def _fixed_batch(value, name, batch, trailing, device, dtype):
    if (not isinstance(value, torch.Tensor) or value.ndim != 3 or
            value.shape[0] not in {1, batch} or value.shape[-1] != trailing or
            value.shape[1] < 1 or not torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite [1 or B,N,{trailing}] tensor")
    return value.detach().to(device=device, dtype=dtype).expand(batch, -1, -1)


def _rigid(matrices, name):
    if matrices.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must contain 4x4 transforms")
    if not torch.allclose(matrices[:, 3], matrices.new_tensor([0., 0., 0., 1.]).expand(len(matrices), -1),
                          atol=1e-6, rtol=0):
        raise ValueError(f"{name} must be homogeneous")
    rotation = matrices[:, :3, :3]
    identity = torch.eye(3, dtype=matrices.dtype, device=matrices.device).expand_as(rotation)
    if not torch.allclose(rotation @ rotation.transpose(-1, -2), identity, atol=1e-4, rtol=1e-4):
        raise ValueError(f"{name} must contain rigid transforms of either parity")


def _probes(transforms, points, step_mm):
    """Six positive then six negative probes per transform, in eta=(rho*w,t)."""
    moved = transform_points(points, transforms)
    center = moved.mean(1)
    radius = (moved - center[:, None]).square().sum(-1).mean(-1).clamp_min(1.).sqrt()
    twists = torch.eye(6, dtype=transforms.dtype, device=transforms.device)[None].repeat(len(transforms), 1, 1)
    twists = twists * step_mm
    twists[:, :, :3] = twists[:, :, :3] / radius[:, None, None]
    twists = torch.cat((twists, -twists), dim=1)
    result = centered_increment(transforms[:, None].expand(-1, 12, -1, -1).reshape(-1, 4, 4),
                                twists.reshape(-1, 6), center[:, None].expand(-1, 12, -1).reshape(-1, 3))
    return result.reshape(len(transforms), 12, 4, 4)


def _smooth_error(probes, reference, anchors, smoothing_mm):
    moved = transform_points(anchors[:, None], probes)
    target = transform_points(anchors, reference)[:, None]
    squared_rms = (moved - target).square().sum(-1).mean(-1)
    return (squared_rms + smoothing_mm ** 2).sqrt() - smoothing_mm


def pose_secant_loss(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    transforms: torch.Tensor,
    reference: torch.Tensor,
    points: torch.Tensor,
    anchors: torch.Tensor,
    *,
    step_mm: float = .5,
    smoothing_mm: float = 1.,
    local_minimum_weight: float = .25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match physical SE(3) energy secants and positive reference-neighbor gaps.

    ``transforms`` is [B,4,4]; reference is [1 or B,4,4]. Points [1 or B,N,3]
    set the optimizer's current centroid and radius, while independent fixed
    anchors [1 or B,M,3] define the same-anchor target. The physical coordinates
    are eta=(rho*rotation_radians, translation_mm), with rho squared clamped to
    one exactly as in ``refine_transform``. Each axis uses the exact centered
    rigid increment, including for the finite-scale target; opposite reference
    parity is rejected because a proper increment cannot repair it.

    ``energy_fn`` receives flattened transforms and returns one scalar energy
    per transform. It must preserve row order and may chunk the query batch.
    There are 25*B evaluations (12*B candidate probes, 12*B reference probes,
    B reference centers), or 12*B when the local-minimum weight is zero. The
    encoder can be shared by all queries. Probe geometry and target values are
    detached; ordinary backpropagation through the returned energies trains
    model parameters without coordinate derivatives or ``create_graph``.

    Huber losses use delta=1 in energy/mm coordinates. Reference-neighbor gaps
    are divided by step_mm to match secant units. Matching positive gaps
    penalizes stationary reference maxima; all terms ignore additive energy
    offsets. Returned component metrics are detached tensors.
    """
    for name, value, positive in (("step_mm", step_mm, True), ("smoothing_mm", smoothing_mm, True),
                                  ("local_minimum_weight", local_minimum_weight, False)):
        if not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    if (not isinstance(transforms, torch.Tensor) or transforms.ndim != 3 or
            transforms.shape[1:] != (4, 4) or not len(transforms) or not torch.isfinite(transforms).all()):
        raise ValueError("transforms must be a nonempty finite [B,4,4] tensor")
    values = (transforms, reference, points, anchors)
    dtype = torch.float64 if any(isinstance(value, torch.Tensor) and value.dtype == torch.float64
                                for value in values) else torch.float32
    device, batch = transforms.device, len(transforms)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        transforms = transforms.detach().to(dtype=dtype)
        reference = _fixed_batch(reference, "reference", batch, 4, device, dtype)
        points = _fixed_batch(points, "points", batch, 3, device, dtype)
        anchors = _fixed_batch(anchors, "anchors", batch, 3, device, dtype)
        _rigid(transforms, "transforms")
        _rigid(reference, "reference")
        if (torch.linalg.det(transforms[:, :3, :3]) * torch.linalg.det(reference[:, :3, :3]) < 0).any():
            raise ValueError("transforms and reference must have matching parity")
        candidates = _probes(transforms, points, step_mm)
        target_values = _smooth_error(candidates, reference, anchors, smoothing_mm)
        target_secants = (target_values[:, :6] - target_values[:, 6:]) / (2 * step_mm)
        matrices = candidates.reshape(-1, 4, 4)
        if local_minimum_weight:
            neighbors = _probes(reference, points, step_mm)
            target_gaps = _smooth_error(neighbors, reference, anchors, smoothing_mm) / step_mm
            matrices = torch.cat((matrices, neighbors.reshape(-1, 4, 4), reference), dim=0)

    with torch.autocast(device_type=device.type, enabled=False):
        energies = energy_fn(matrices)
        if (not isinstance(energies, torch.Tensor) or energies.shape != (len(matrices),) or
                not torch.isfinite(energies).all() or energies.device != device):
            raise ValueError("energy_fn must return a finite energy vector on the transform device")
        energies = energies.to(dtype=dtype)
        candidate_values = energies[:12 * batch].reshape(batch, 12)
        predicted_secants = (candidate_values[:, :6] - candidate_values[:, 6:]) / (2 * step_mm)
        secant = F.huber_loss(predicted_secants, target_secants, delta=1.)
        local_minimum = secant.new_zeros(())
        if local_minimum_weight:
            reference_values = energies[24 * batch:]
            neighbor_values = energies[12 * batch:24 * batch].reshape(batch, 12)
            predicted_gaps = (neighbor_values - reference_values[:, None]) / step_mm
            local_minimum = F.huber_loss(predicted_gaps, target_gaps, delta=1.)
        loss = secant + local_minimum_weight * local_minimum
    return loss, {"secant": secant.detach(), "local_minimum": local_minimum.detach()}


def gradient_alignment_diagnostic(query_fn, points, initial, reference, anchors,
                                  affine, input_shape, smoothing_mm=1.):
    """Diagnose one pose's actual energy gradient without updating the model.

    Both gradients use eta=(rho*omega,t), hence rotational gradients are divided
    by rho, the square root of the optimizer's rotational preconditioner. The
    target is smooth same-anchor RMS error. Its analytic rotation derivative
    uses the fitting-point center, even when the fixed anchors have a different
    centroid. Positive dot predicts target descent for the infinitesimal,
    unclipped optimizer direction; finite steps and backtracking can differ.

    A zero target gradient has no comparable direction: cosine and descent are
    None. A zero energy gradient with a nonzero target is stalled (descent=False,
    cosine=None). Nonfinite gradients raise; a diagnostic caller may record this
    independently of the actual refinement outcome. Returns Python scalars only
    and never accumulates gradients into model parameters or input geometry.
    """
    if not math.isfinite(smoothing_mm) or smoothing_mm <= 0:
        raise ValueError("smoothing_mm must be positive and finite")
    if not isinstance(points, torch.Tensor):
        raise ValueError("points must be a [1,N,3] tensor")
    device = points.device
    dtype = torch.float64 if points.dtype == torch.float64 else torch.float32
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        points = _fixed_batch(points, "points", 1, 3, device, dtype)
        initial = _fixed_batch(initial, "initial", 1, 4, device, dtype)
        reference = _fixed_batch(reference, "reference", 1, 4, device, dtype)
        anchors = _fixed_batch(anchors, "anchors", 1, 3, device, dtype)
        _rigid(initial, "initial")
        _rigid(reference, "reference")
        if (torch.linalg.det(initial[:, :3, :3]) * torch.linalg.det(reference[:, :3, :3]) < 0).any():
            raise ValueError("initial and reference must have matching parity")
        moved = transform_points(points, initial)
        center = moved.mean(1)
        radius = (moved - center[:, None]).square().sum(-1).mean(-1).clamp_min(1.).sqrt()
        moved_anchors = transform_points(anchors, initial)
        residual = moved_anchors - transform_points(anchors, reference)
        denominator = (residual.square().sum(-1).mean(-1) + smoothing_mm ** 2).sqrt()
        target_gradient = torch.cat((torch.cross(moved_anchors - center[:, None], residual, dim=-1).mean(1),
                                     residual.mean(1)), dim=-1) / denominator[:, None]
        target_gradient[:, :3] /= radius[:, None]
    with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
        twist = points.new_zeros((1, 6), requires_grad=True)
        trial = centered_increment(initial, twist, center)
        query = transform_points(points, trial)
        energy, _ = registration_energy(query_fn(query), query, affine, input_shape)
        energy_gradient = torch.autograd.grad(energy.sum(), twist, create_graph=False)[0].detach()
    energy_gradient[:, :3] /= radius[:, None]
    if not torch.isfinite(energy_gradient).all() or not torch.isfinite(target_gradient).all():
        raise FloatingPointError("nonfinite initial energy or target gradient")
    dot = float((energy_gradient * target_gradient).sum())
    energy_norm = float(torch.linalg.vector_norm(energy_gradient))
    target_norm = float(torch.linalg.vector_norm(target_gradient))
    comparable = target_norm > 1e-12
    cosine = max(-1., min(1., dot / (energy_norm * target_norm))) if comparable and energy_norm > 1e-12 else None
    return dict(cosine=cosine, dot=dot, energy_norm=energy_norm, target_norm=target_norm,
                descent=bool(dot > 0) if comparable else None)
