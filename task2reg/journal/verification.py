"""Auditable NumPy gates for spatial pseudo supervision (all lengths in mm).

Held-out teacher distances measure self-consistency, not independent ground truth.
Default thresholds are development starting values, not a calibrated safety claim.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class VerificationConfig:
    u50_mm: float = .5
    u95_mm: float = 1.
    max_displacement_mm: float = 2.
    min_same_parity: float = 1.
    min_global_coverage: float = .85
    min_sector_coverage: float = .65
    max_heldout_median_mm: float = .75
    max_heldout_p90_mm: float = 1.5
    max_outside_fraction: float = .05
    coverage_distance_mm: float = 2.
    distinct_basin_mm: float = 2.
    min_relative_energy_gap: float = .1
    uncertainty_sigma_mm: float = .75
    max_point_uncertainty_mm: float = 1.5

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('min_same_parity', 'min_global_coverage', 'min_sector_coverage',
                     'max_outside_fraction', 'min_relative_energy_gap'):
            if getattr(self, name) > 1:
                raise ValueError(f'{name} must be at most one')
        if self.uncertainty_sigma_mm <= 0 or self.coverage_distance_mm <= 0:
            raise ValueError('uncertainty sigma and coverage distance must be positive')


def _points(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value) or not np.isfinite(value).all():
        raise ValueError(f'{name} must be nonempty finite [N,3]')
    return value


def _transforms(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 3 or value.shape[1:] != (4, 4) or not len(value) or not np.isfinite(value).all():
        raise ValueError(f'{name} must be nonempty finite [K,4,4]')
    if not np.allclose(value[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError(f'{name} has invalid homogeneous rows')
    rotation = value[:, :3, :3]
    if not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-3):
        raise ValueError(f'{name} must contain rigid transforms (either parity)')
    return value


def _apply(transforms, points):
    return np.einsum('kij,nj->kni', transforms[:, :3, :3], points) + transforms[:, None, :3, 3]


def pose_displacement(a, b, anchors):
    """Mean displacement of the same fixed IOS anchors, not landmark TRE."""
    points = _points(anchors, 'anchors')
    moved = _apply(_transforms(np.stack([a, b]), 'transforms'), points)
    return float(np.linalg.norm(moved[0] - moved[1], axis=-1).mean())


def transform_medoid(transforms, anchors):
    """Index minimizing total pairwise fixed-anchor displacement; ties use first."""
    transforms = _transforms(transforms, 'transforms')
    moved = _apply(transforms, _points(anchors, 'anchors'))
    totals = np.array([np.linalg.norm(moved - item, axis=-1).mean(axis=1).sum() for item in moved])
    return int(np.argmin(totals))


def point_uncertainty(transforms, medoid, points):
    """RMS per-point disagreement with the medoid across all refinement runs."""
    transforms = _transforms(transforms, 'transforms')
    medoid = _transforms(np.asarray(medoid)[None], 'medoid')
    points = _points(points, 'points')
    deviations = _apply(transforms, points) - _apply(medoid, points)
    return np.sqrt(np.square(deviations).sum(axis=-1).mean(axis=0))


def verify_registration(transforms=None, points=None, *, anchors=None, distances=None,
                        heldout_distances=None, sector_ids=None, outside_mask=None,
                        candidate_transforms=None, candidate_energies=None, config=None):
    """Return fail-closed gates, medoid, uncertainty and per-point training weights.

    ``transforms`` are results of independent starts and leave-sector-out runs.
    ``distances`` and ``outside_mask`` describe medoid-transformed surface points;
    ``heldout_distances`` contains only sectors excluded from their respective fits.
    ``candidate_*`` must include every evaluated basin and the scored medoid.
    U50/U95/max summarize per-anchor RMS disagreement across refinement runs.
    Invalid evidence returns rejection and zero weights instead of raising.
    Invalid configuration raises rather than silently substituting defaults.
    """
    config = VerificationConfig() if config is None else config
    if not isinstance(config, VerificationConfig):
        raise TypeError('config must be VerificationConfig')
    count = len(points) if points is not None and np.ndim(points) > 0 else 0
    result = dict(accepted=False, reasons=[], config=asdict(config), medoid_index=None,
                  transform=None, uncertainty_mm=None, point_weights=np.zeros(count), metrics={})
    try:
        points = _points(points, 'points')
        anchors = _points(anchors, 'anchors')
        transforms = _transforms(transforms, 'transforms')
        candidates = _transforms(candidate_transforms, 'candidate_transforms')
        if len(transforms) < 2:
            raise ValueError('transforms must contain at least two refinement runs')
        arrays = {}
        for name, value, size in [('distances', distances, len(points)),
                                  ('heldout_distances', heldout_distances, None),
                                  ('sector_ids', sector_ids, len(points)),
                                  ('outside_mask', outside_mask, len(points)),
                                  ('candidate_energies', candidate_energies, len(candidates))]:
            value = np.asarray(value, dtype=float)
            if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
                raise ValueError(f'{name} must be a nonempty finite vector')
            if size is not None and len(value) != size:
                raise ValueError(f'{name} has incorrect length')
            if np.any(value < 0):
                raise ValueError(f'{name} must be nonnegative')
            arrays[name] = value
        sectors = arrays['sector_ids']
        if np.any(sectors != sectors.astype(int)) or len(np.unique(sectors)) < 2:
            raise ValueError('sector_ids must define at least two nonnegative integer sectors')
        outside = arrays['outside_mask']
        if not np.isin(outside, [0, 1]).all():
            raise ValueError('outside_mask must contain booleans or 0/1')
    except (ValueError, TypeError, OverflowError) as exc:
        result['reasons'].append(f'invalid_evidence: {exc}')
        return result

    index = transform_medoid(transforms, anchors)
    medoid = transforms[index]
    displacement = point_uncertainty(transforms, medoid, anchors)
    uncertainty = point_uncertainty(transforms, medoid, points)
    parity = np.sign(np.linalg.det(transforms[:, :3, :3]))
    coverage = (arrays['distances'] <= config.coverage_distance_mm) & ~outside.astype(bool)
    sector_coverage = {str(int(s)): float(coverage[sectors == s].mean()) for s in np.unique(sectors)}
    heldout = arrays['heldout_distances']
    energies = arrays['candidate_energies']
    best = int(np.argmin(energies))
    gaps = []
    for candidate, energy in zip(candidates, energies):
        if pose_displacement(candidate, candidates[best], anchors) > config.distinct_basin_mm:
            gaps.append(float((energy - energies[best]) / max(abs(energies[best]), 1e-12)))
    metrics = dict(u50_mm=float(np.median(displacement)), u95_mm=float(np.quantile(displacement, .95)),
                   max_displacement_mm=float(displacement.max()),
                   same_parity_fraction=float((parity == parity[index]).mean()),
                   global_coverage=float(coverage.mean()), sector_coverage=sector_coverage,
                   min_sector_coverage=min(sector_coverage.values()),
                   heldout_median_mm=float(np.median(heldout)),
                   heldout_p90_mm=float(np.quantile(heldout, .9)),
                   outside_fraction=float(outside.mean()),
                   distinct_basin_relative_energy_gap=min(gaps) if gaps else None)
    reasons = []
    medoid_matches = np.all(np.isclose(candidates, medoid, rtol=1e-5, atol=1e-5), axis=(1, 2))
    if not medoid_matches.any():
        reasons.append('missing_medoid_candidate')
    else:
        medoid_energy = float(energies[medoid_matches].min())
        medoid_gap = (medoid_energy - energies[best]) / max(abs(energies[best]), 1e-12)
        metrics['medoid_relative_energy_gap'] = float(medoid_gap)
        if (pose_displacement(medoid, candidates[best], anchors) > config.distinct_basin_mm
                and medoid_gap >= config.min_relative_energy_gap):
            reasons.append('medoid_worse_basin')
    if (metrics['u50_mm'] > config.u50_mm or metrics['u95_mm'] > config.u95_mm or
            metrics['max_displacement_mm'] > config.max_displacement_mm):
        reasons.append('pose_uncertainty')
    if metrics['same_parity_fraction'] < config.min_same_parity:
        reasons.append('parity')
    if metrics['global_coverage'] < config.min_global_coverage:
        reasons.append('global_coverage')
    if metrics['min_sector_coverage'] < config.min_sector_coverage:
        reasons.append('sector_coverage')
    if metrics['heldout_median_mm'] > config.max_heldout_median_mm:
        reasons.append('heldout_median')
    if metrics['heldout_p90_mm'] > config.max_heldout_p90_mm:
        reasons.append('heldout_p90')
    if metrics['outside_fraction'] >= config.max_outside_fraction:
        reasons.append('outside_fraction')
    if gaps and min(gaps) < config.min_relative_energy_gap:
        reasons.append('distinct_basin_ambiguity')
    weights = np.exp(-np.square(uncertainty) / (2 * config.uncertainty_sigma_mm ** 2))
    weights[(uncertainty > config.max_point_uncertainty_mm) | outside.astype(bool)] = 0
    if not np.any(weights > 0):
        reasons.append('no_supervision_mass')
    if reasons:
        weights[:] = 0
    result.update(accepted=not reasons, reasons=reasons, medoid_index=index, transform=medoid.copy(),
                  uncertainty_mm=uncertainty, point_weights=weights, metrics=metrics)
    return result
