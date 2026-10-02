"""Continuous registration fields in physical millimetres.

Volumes keep NIfTI array order ``[B,C,I,J,K]`` throughout. Explicit trilinear
interpolation avoids ``grid_sample``'s higher-order autograd restriction. Its
derivatives are piecewise smooth (voxel boundaries are not differentiable).
The released crown network implementation and checkpoint format are untouched.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

from task2reg.crown_network import CrownLocalizerUNet


def _shape3(shape: Sequence[int]) -> tuple[int, int, int]:
    result = tuple(int(size) for size in shape)
    if len(result) != 3 or min(result) < 1:
        raise ValueError("input_shape must contain three positive sizes")
    return result


def world_to_voxel(points_world: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    """Map batched physical points through the full affine, including obliquity."""
    if points_world.ndim != 3 or points_world.shape[-1] != 3:
        raise ValueError("points_world must have shape [B,N,3]")
    if affine.shape != (points_world.shape[0], 4, 4):
        raise ValueError("affine must have shape [B,4,4]")
    # linalg and the millimetre coordinate arithmetic should not run in float16.
    dtype = torch.float64 if points_world.dtype == torch.float64 else torch.float32
    points = points_world.to(dtype)
    matrix = affine.to(device=points.device, dtype=dtype)
    if not torch.isfinite(points).all() or not torch.isfinite(matrix).all():
        raise ValueError("points and affine must be finite")
    if not torch.allclose(matrix[:, 3], matrix.new_tensor([0., 0., 0., 1.]).expand(len(matrix), -1)):
        raise ValueError("affine must be homogeneous")
    with torch.autocast(device_type=points.device.type, enabled=False):
        return torch.linalg.solve(
            matrix[:, :3, :3], (points - matrix[:, None, :3, 3]).transpose(1, 2)
        ).transpose(1, 2)


def sample_features(
    volume: torch.Tensor,
    points_world: torch.Tensor,
    affine: torch.Tensor,
    input_shape: Sequence[int] | None = None,
) -> torch.Tensor:
    """Sample ``[B,C,I,J,K]`` at world points, returning ``[B,N,C]``.

    Multi-scale grids share the input grid's endpoint-aligned physical extent.
    Outside points use the nearest boundary feature. Callers must include the
    separate physical ROI penalty in a registration objective.
    """
    if volume.ndim != 5 or volume.shape[0] != points_world.shape[0]:
        raise ValueError("volume must have shape [B,C,I,J,K] with matching batch")
    spatial = _shape3(volume.shape[2:])
    full = _shape3(input_shape if input_shape is not None else spatial)
    coordinates = world_to_voxel(points_world, affine)
    scale = coordinates.new_tensor([
        (size - 1) / max(full_size - 1, 1) for size, full_size in zip(spatial, full)
    ])
    coordinates = coordinates * scale
    upper_bound = coordinates.new_tensor(spatial) - 1
    coordinates = torch.minimum(torch.clamp_min(coordinates, 0), upper_bound)
    lower = torch.floor(coordinates).to(torch.long)
    upper = torch.minimum(lower + 1, lower.new_tensor(spatial) - 1)
    fraction = coordinates - lower.to(coordinates.dtype)
    flattened = volume.flatten(2)
    sampled = None
    for corner in product((0, 1), repeat=3):
        indices = [upper[..., axis] if bit else lower[..., axis] for axis, bit in enumerate(corner)]
        flat_index = indices[0] * spatial[1] * spatial[2] + indices[1] * spatial[2] + indices[2]
        values = torch.gather(flattened, 2, flat_index[:, None].expand(-1, volume.shape[1], -1)).transpose(1, 2)
        weight = torch.ones_like(fraction[..., 0])
        for axis, bit in enumerate(corner):
            weight = weight * (fraction[..., axis] if bit else 1 - fraction[..., axis])
        weighted = values * weight[..., None]
        sampled = weighted if sampled is None else sampled + weighted
    return sampled


def roi_outside_distance(
    points_world: torch.Tensor, affine: torch.Tensor, input_shape: Sequence[int]
) -> torch.Tensor:
    """Physical displacement to the clamped voxel box, zero for inside points.

    This is Euclidean box distance for rotated anisotropic grids. For a sheared
    affine it is the norm of the voxel-axis clamp displacement, not the exact
    shortest distance to the resulting parallelepiped; it remains an explicit
    nonnegative physical penalty with zero exactly inside the ROI.
    """
    coordinates = world_to_voxel(points_world, affine)
    bounds = coordinates.new_tensor(_shape3(input_shape)) - 1
    clamped = torch.minimum(torch.clamp_min(coordinates, 0), bounds)
    displacement = coordinates - clamped
    matrix = affine[:, :3, :3].to(device=coordinates.device, dtype=coordinates.dtype)
    with torch.autocast(device_type=coordinates.device.type, enabled=False):
        physical = displacement @ matrix.transpose(-1, -2)
        # Smooth only the zero cusp, retaining zero penalty inside and finite
        # second derivatives needed by unrolled optimization.
        return torch.sqrt(physical.square().sum(-1) + 1e-12) - 1e-6


class RegistrationField(nn.Module):
    """Unchanged crown U-Net backbone plus dense or implicit distance queries."""

    def __init__(self, base_channels: int = 8, mode: str = "implicit", truncation_mm: float = 8.) -> None:
        super().__init__()
        if mode not in ("dense", "implicit"):
            raise ValueError("mode must be dense or implicit")
        if base_channels < 4 or base_channels % 4:
            raise ValueError("base_channels must be a positive multiple of four")
        if not 0 < truncation_mm < float("inf"):
            raise ValueError("truncation_mm must be positive and finite")
        self.mode = mode
        self.truncation_mm = float(truncation_mm)
        self.backbone = CrownLocalizerUNet(base_channels=base_channels)
        if mode == "dense":
            self.distance_head = nn.Conv3d(base_channels, 2, kernel_size=1)
        else:
            self.projections = nn.ModuleList([
                nn.Conv3d(base_channels * (2 ** level), 16, kernel_size=1)
                for level in range(3)
            ])
            self.jaw_embedding = nn.Embedding(2, 8)
            self.query_mlp = nn.Sequential(
                nn.Linear(48 + 3 + 8, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1),
            )

    def encode(self, image: torch.Tensor) -> dict:
        if image.ndim != 5 or image.shape[1] != 1 or min(image.shape[2:]) < 16:
            raise ValueError("image must have shape [B,1,I,J,K] with each spatial size >=16")
        # Equivalent explicit forward keeps the submitted backbone byte-for-byte
        # intact and gives the query head access to decoder features.
        network = self.backbone
        first = network.encoder1(image)
        second = network.encoder2(network.pool(first))
        third = network.encoder3(network.pool(second))
        fourth = network.encoder4(network.pool(third))
        center = network.bottleneck(network.pool(fourth))
        decoded4 = network.decoder4(torch.cat((network._match(network.up4(center), fourth), fourth), dim=1))
        decoded3 = network.decoder3(torch.cat((network._match(network.up3(decoded4), third), third), dim=1))
        decoded2 = network.decoder2(torch.cat((network._match(network.up2(decoded3), second), second), dim=1))
        decoded1 = network.decoder1(torch.cat((network._match(network.up1(decoded2), first), first), dim=1))
        logits = network.output(decoded1)
        if self.mode == "dense":
            features = (F.softplus(self.distance_head(decoded1)),)
        else:
            features = tuple(project(value) for project, value in zip(self.projections, (decoded1, decoded2, decoded3)))
        return {"logits": logits, "features": features, "input_shape": tuple(image.shape[2:])}

    def query(self, context: dict, points_world: torch.Tensor, affine: torch.Tensor, jaw: torch.Tensor) -> torch.Tensor:
        if jaw.shape != (points_world.shape[0],) or not ((jaw == 0) | (jaw == 1)).all():
            raise ValueError("jaw must have shape [B] with values 0 (upper) or 1 (lower)")
        jaw = jaw.to(device=points_world.device, dtype=torch.long)
        sampled = [sample_features(feature, points_world, affine, context["input_shape"]) for feature in context["features"]]
        if self.mode == "dense":
            return sampled[0].gather(-1, jaw[:, None, None].expand(-1, points_world.shape[1], 1)).squeeze(-1)
        voxel = world_to_voxel(points_world, affine)
        extent = voxel.new_tensor(context["input_shape"]) - 1
        normalized = 2 * voxel / extent.clamp_min(1) - 1
        embedding = self.jaw_embedding(jaw)[:, None].expand(-1, points_world.shape[1], -1)
        values = torch.cat((*sampled, normalized.clamp(-1, 1), embedding), dim=-1)
        # NIfTI affines and mesh coordinates commonly arrive as float64 while
        # the learned network is float32; the cast preserves coordinate grads.
        values = values.to(dtype=self.query_mlp[0].weight.dtype)
        return F.softplus(self.query_mlp(values).squeeze(-1))


def registration_energy(
    distances: torch.Tensor,
    points_world: torch.Tensor,
    affine: torch.Tensor,
    input_shape: Sequence[int],
    *,
    coverage_weight: float = 1.,
    outside_weight: float = 2.,
    coverage_threshold_mm: float = 2.,
    coverage_temperature_mm: float = .25,
    robust_epsilon_mm: float = .1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """All-point pseudo-Huber fit plus soft coverage and physical ROI penalties."""
    if distances.shape != points_world.shape[:2] or distances.shape[-1] == 0:
        raise ValueError("distances must match nonempty points [B,N]")
    if not torch.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("distances must be finite and nonnegative")
    if coverage_temperature_mm <= 0 or robust_epsilon_mm <= 0 or coverage_weight < 0 or outside_weight < 0:
        raise ValueError("energy scales must be positive and weights nonnegative")
    outside = roi_outside_distance(points_world, affine, input_shape)
    robust = torch.sqrt(distances.square() + robust_epsilon_mm ** 2) - robust_epsilon_mm
    coverage = torch.sigmoid((coverage_threshold_mm - distances) / coverage_temperature_mm).mean(-1)
    distance_energy = robust.mean(-1)
    outside_energy = outside.mean(-1)
    energy = distance_energy + coverage_weight * (1 - coverage) + outside_weight * outside_energy
    return energy, {
        "distance_energy": distance_energy,
        "coverage": coverage,
        "outside_distance_mm": outside_energy,
        "outside_fraction": (outside > 1e-5).to(distances.dtype).mean(-1),
    }


def weighted_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
    *, delta_mm: float = 1., distance_scale_mm: float = 2., background_weight: float = .1,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if delta_mm <= 0 or distance_scale_mm <= 0 or background_weight < 0:
        raise ValueError("loss scales must be positive and background weight nonnegative")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all() or (target < 0).any():
        raise ValueError("prediction and nonnegative target must be finite")
    weight = background_weight + torch.exp(-target / distance_scale_mm)
    if weights is not None:
        if weights.shape != target.shape or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("weights must match targets and be finite and nonnegative")
        weight = weight * weights
    loss = F.huber_loss(prediction, target, reduction="none", delta=delta_mm)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-12)


def pairwise_rank_loss(
    energies: torch.Tensor,
    errors_mm: torch.Tensor,
    *, minimum_gap_mm: float = .5, temperature: float = .25,
    margin_scale: float = .1, min_margin: float = .1, max_margin: float = 1.,
) -> torch.Tensor:
    """Soft ordering loss; only strictly distinguishable better/worse pairs count."""
    if energies.shape != errors_mm.shape or energies.ndim < 1:
        raise ValueError("energies and errors must have matching [...,K] shapes")
    if not torch.isfinite(energies).all() or not torch.isfinite(errors_mm).all():
        raise ValueError("energies and errors must be finite")
    if temperature <= 0 or minimum_gap_mm < 0 or margin_scale < 0 or min_margin < 0 or max_margin < min_margin:
        raise ValueError("invalid ranking loss scales")
    difference = errors_mm[..., None, :] - errors_mm[..., :, None]
    eligible = difference > minimum_gap_mm
    if not eligible.any():
        return energies.sum() * 0
    margin = (margin_scale * difference).clamp(min_margin, max_margin)
    logits = (energies[..., :, None] - energies[..., None, :] + margin) / temperature
    return F.softplus(logits[eligible]).mean()


def eikonal_loss(
    distances: torch.Tensor, points_world: torch.Tensor, target_mm: torch.Tensor,
    *, minimum_mm: float = .5, maximum_mm: float = 3.,
) -> torch.Tensor:
    """Physical unit-gradient penalty away from the unsigned-distance cusp."""
    if distances.shape != target_mm.shape or distances.shape != points_world.shape[:2]:
        raise ValueError("distance/target/query shapes must match")
    selected = (target_mm > minimum_mm) & (target_mm < maximum_mm)
    if not selected.any():
        return distances.sum() * 0
    gradients = torch.autograd.grad(distances.sum(), points_world, create_graph=True, retain_graph=True)[0]
    norms = torch.sqrt(gradients.square().sum(-1) + 1e-12)
    return (norms[selected] - 1).square().mean()


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    return points @ transform[..., :3, :3].transpose(-1, -2) + transform[..., None, :3, 3]


def fixed_point_error(transform: torch.Tensor, reference: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Same-anchor displacement in mm; this is not independent-landmark TRE."""
    difference = transform_points(points, transform) - transform_points(points, reference)
    return torch.linalg.vector_norm(difference, dim=-1).mean(-1)


def centered_increment(transform: torch.Tensor, twist: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Left-compose a proper SE(3) increment about a physical center.

    ``twist`` orders rotation radians then translation mm. Matrix exponential
    gives stable first/second derivatives at zero without Rodrigues divisions.
    Conjugating the increment by the center prevents scanner-origin coupling.
    """
    if twist.shape != (*transform.shape[:-2], 6) or center.shape != (*transform.shape[:-2], 3):
        raise ValueError("twist and center must match the transform batch")
    wx, wy, wz, tx, ty, tz = twist.unbind(-1)
    zero = torch.zeros_like(wx)
    algebra = torch.stack((
        zero, -wz, wy, tx, wz, zero, -wx, ty,
        -wy, wx, zero, tz, zero, zero, zero, zero,
    ), dim=-1).reshape(*twist.shape[:-1], 4, 4)
    with torch.autocast(device_type=twist.device.type, enabled=False):
        delta = torch.matrix_exp(algebra)
        rotation = delta[..., :3, :3]
        translation = center - (rotation @ center[..., None]).squeeze(-1) + delta[..., :3, 3]
        upper = torch.cat((rotation, translation[..., None]), dim=-1)
        centered = torch.cat((upper, delta[..., 3:4, :]), dim=-2)
        return centered @ transform


def refine_transform(
    query_fn: Callable[[torch.Tensor], torch.Tensor],
    points: torch.Tensor,
    initial: torch.Tensor,
    affine: torch.Tensor,
    input_shape: Sequence[int],
    *, steps: int = 10, learning_rate: float = .5, differentiable: bool = False,
    coverage_weight: float = 1., outside_weight: float = 2.,
) -> dict[str, torch.Tensor]:
    """Centered, parity-preserving field-gradient refinement.

    Angular gradients are normalized by the mean squared crown radius. Each
    step is capped at 1 mm and 0.1 rad. Inference uses backtracking, retaining
    the old transform if no improvement is found. ``differentiable=True`` uses
    the fixed step (no line search) and retains the higher-order graph for a
    short pose-supervised unroll. It does not guarantee decreasing energy.
    """
    if steps < 0 or learning_rate <= 0:
        raise ValueError("steps must be nonnegative and learning_rate positive")
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] == 0:
        raise ValueError("points must have nonempty shape [B,N,3]")
    if initial.shape != (points.shape[0], 4, 4) or not torch.isfinite(initial).all():
        raise ValueError("initial must be a finite [B,4,4] transform")
    if not torch.allclose(initial[:, 3], initial.new_tensor([0., 0., 0., 1.]).expand(len(initial), -1)):
        raise ValueError("initial must be homogeneous")
    dtype = torch.float64 if points.dtype == torch.float64 else torch.float32
    points, transform = points.to(dtype), initial.to(dtype)
    rotation = transform[:, :3, :3]
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype).expand_as(rotation)
    if not torch.allclose(rotation @ rotation.transpose(-1, -2), identity, atol=1e-4, rtol=1e-4):
        raise ValueError("initial linear block must be orthogonal")

    def evaluate(matrix):
        moved = transform_points(points, matrix)
        return registration_energy(query_fn(moved), moved, affine, input_shape,
                                   coverage_weight=coverage_weight, outside_weight=outside_weight)

    with torch.enable_grad(), torch.autocast(device_type=points.device.type, enabled=False):
        initial_energy, initial_parts = evaluate(transform)
        for _ in range(steps):
            if not differentiable:
                transform = transform.detach()
            moved = transform_points(points, transform)
            center = moved.mean(1)
            radius2 = (moved - center[:, None]).square().sum(-1).mean(-1).clamp_min(1.)
            twist = points.new_zeros((len(points), 6), requires_grad=True)
            trial = centered_increment(transform, twist, center)
            energy, _ = evaluate(trial)
            gradient = torch.autograd.grad(energy.sum(), twist, create_graph=differentiable)[0]
            step = -learning_rate * torch.cat((gradient[:, :3] / radius2[:, None], gradient[:, 3:]), dim=-1)
            rotation_norm = torch.sqrt(step[:, :3].square().sum(-1, keepdim=True) + 1e-12)
            translation_norm = torch.sqrt(step[:, 3:].square().sum(-1, keepdim=True) + 1e-12)
            step = torch.cat((step[:, :3] / (rotation_norm / .1).clamp_min(1.),
                              step[:, 3:] / translation_norm.clamp_min(1.)), dim=-1)
            if differentiable:
                transform = centered_increment(transform, step, center)
            else:
                with torch.no_grad():
                    accepted = transform
                    best_energy = energy.detach()
                    for reduction in range(6):
                        proposal = centered_increment(transform, step * (.5 ** reduction), center)
                        proposed_energy, _ = evaluate(proposal)
                        improved = proposed_energy < best_energy
                        accepted = torch.where(improved[:, None, None], proposal, accepted)
                        best_energy = torch.minimum(best_energy, proposed_energy)
                    transform = accepted
        final_energy, final_parts = evaluate(transform)
    result = {
        "transform": transform,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "initial_coverage": initial_parts["coverage"],
        "final_coverage": final_parts["coverage"],
        "initial_outside_fraction": initial_parts["outside_fraction"],
        "final_outside_fraction": final_parts["outside_fraction"],
    }
    return result if differentiable else {key: value.detach() for key, value in result.items()}
