#!/usr/bin/env python3
"""TOY / ANALYTIC: exact point-set distance need not guide the correct pose.

This deterministic counterexample contains no clinical data or trained models.
It compares exact nearest-reference-point UDF energy with a correspondence-
preserving anchor target along one-dimensional pose slices. Its minima are
slice minima, not certified six-dimensional stationary points. A symmetric
control also shows what any unordered pointwise world-field aggregation cannot
identify, including a learned task potential.

Default artifacts are written outside the repository under the system temp dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.special import expit


def make_repeated_cusps():
    """Seven irregular copies of a nine-point peak, with a 4 mm base spacing.

    These are illustrative geometric motifs, not simulated tooth anatomy.
    Fixed unequal heights and peak scales break periodicity; finite endpoints
    also make any nonzero translation distinguishable from the identity.
    """
    motif = np.array([[0., 0., 1.2], [-.35, 0., .5], [.35, 0., .5],
                      [0., -.35, .5], [0., .35, .5], [-.4, -.3, 0.],
                      [-.4, .3, 0.], [.4, -.3, 0.], [.4, .3, 0.]])
    heights = [0., .035, -.025, .055, -.045, .020, .065]
    groups = []
    for index, height in enumerate(heights):
        points = motif.copy()
        points[:, 2] *= 1. + .015 * (index - 3)
        points += [(index - 3) * 4., 0., height]
        groups.append(points)
    return np.concatenate(groups)


def make_symmetric_ring(count=12, radius_mm=8.):
    """A regular polygon with exact 360/count-degree set symmetry."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 3:
        raise ValueError('count must be an integer >= 3')
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError('radius_mm must be positive and finite')
    angles = np.arange(count) * 2. * np.pi / count
    return np.column_stack((radius_mm * np.cos(angles), radius_mm * np.sin(angles), np.zeros(count)))


def move_points(points, parameter, mode):
    """Translate in world x (mm), or rotate about the world z axis (degrees)."""
    points = np.asarray(points, dtype=float)
    if mode == 'translation':
        return points + [parameter, 0., 0.]
    if mode == 'rotation':
        angle = np.radians(parameter)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        return points @ rotation.T
    raise ValueError('mode must be translation or rotation')


def scan_landscape(points, parameters, *, mode, secant_step, epsilon_mm=.1):
    """Exact point-set UDF plus smooth coverage; no finite ROI is imposed.

    Energy uses mean(sqrt(d^2 + .1^2) - .1) + 1mm*(1-coverage),
    with coverage=mean(sigmoid((2mm-d)/.25mm)). This matches the geometry
    aggregation used by the research model, with zero ROI penalty here.
    The target is sqrt(mean(||T*x-x||^2)+epsilon^2)-epsilon, preserving
    point identity. Both secants use the same physical slice coordinate.
    """
    points = np.asarray(points, dtype=float)
    grid = np.asarray(parameters, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
        raise ValueError('points must be finite nonempty Nx3')
    if grid.ndim != 1 or len(grid) < 3 or not np.isfinite(grid).all() or not (np.diff(grid) > 0).all():
        raise ValueError('parameters must contain at least three finite strictly increasing values')
    if not np.isfinite(secant_step) or secant_step <= 0 or not np.isfinite(epsilon_mm) or epsilon_mm <= 0:
        raise ValueError('secant_step and epsilon_mm must be positive and finite')
    tree = cKDTree(points)

    def evaluate(parameter):
        moved = move_points(points, parameter, mode)
        distances = tree.query(moved)[0]
        energy = np.mean(np.sqrt(distances ** 2 + .1 ** 2) - .1) + 1. - expit((2. - distances) / .25).mean()
        displacements = np.linalg.norm(moved - points, axis=1)
        target = np.sqrt(np.mean(displacements ** 2) + epsilon_mm ** 2) - epsilon_mm
        return float(energy), float(target), float(displacements.mean())

    values = np.asarray([evaluate(value) for value in grid])
    plus = np.asarray([evaluate(value + secant_step) for value in grid])
    minus = np.asarray([evaluate(value - secant_step) for value in grid])
    secants = (plus[:, :2] - minus[:, :2]) / (2. * secant_step)
    tolerance = 1e-8
    eligible = np.abs(secants[:, 1]) > tolerance
    opposed = eligible & (secants[:, 0] * secants[:, 1] < 0.) & (np.abs(secants[:, 0]) > tolerance)
    flat_wrong = eligible & (np.abs(secants[:, 0]) <= tolerance)
    eligible_count = int(eligible.sum())
    minima = []
    for index in find_peaks(-values[:, 0])[0]:
        minima.append(dict(parameter=float(grid[index]), oracle_energy=float(values[index, 0]),
                           anchor_smooth_rms_mm=float(values[index, 1]), anchor_d_mm=float(values[index, 2]),
                           is_reference_pose=bool(abs(grid[index]) <= 1e-10)))
    return dict(mode=mode, parameter_unit='mm' if mode == 'translation' else 'degrees',
                points_mm=points.tolist(), parameters=grid.tolist(), secant_step=float(secant_step),
                oracle_energy=values[:, 0].tolist(), anchor_smooth_rms_mm=values[:, 1].tolist(),
                anchor_d_mm=values[:, 2].tolist(), oracle_secant=secants[:, 0].tolist(),
                target_secant=secants[:, 1].tolist(), opposed_secants=opposed.tolist(),
                local_minima=minima,
                direction_diagnostics=dict(eligible_count=eligible_count, opposed_count=int(opposed.sum()),
                    opposed_fraction=float(opposed.sum() / eligible_count) if eligible_count else None,
                    flat_oracle_nonstationary_target_count=int(flat_wrong.sum()), tolerance=tolerance,
                    definition='Opposite nonzero central secants among grid positions with a nonstationary anchor target.'),
                minima_definition='Interior local minima of this sampled 1D pose slice, not certified SE(3) minima.')


def _plot(report, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True,
                                gridspec_kw={'height_ratios': [1., 1.25]})
    figure.suptitle('TOY / ANALYTIC: perfect distance values can give wrong pose directions\n'
                   'Deterministic point sets; no clinical data or trained model', fontsize=15)
    for column, scene in enumerate(report['scenes']):
        points = np.asarray(scene['points_mm'])
        example = 4. if scene['mode'] == 'translation' else 30.
        moved = move_points(points, example, scene['mode'])
        second_axis = 2 if scene['mode'] == 'translation' else 1
        geometry = axes[0, column]
        geometry.scatter(points[:, 0], points[:, second_axis], color='#1f2937', s=23,
                         label='Reference point set', zorder=3)
        geometry.scatter(moved[:, 0], moved[:, second_axis], facecolors='none', edgecolors='#d97706',
                         s=55, linewidths=1.3, label=f'Source after {example:g} {scene["parameter_unit"]}')
        geometry.set(xlabel='x (mm)', ylabel='z (mm)' if second_axis == 2 else 'y (mm)', title=scene['title'])
        geometry.set_aspect('equal', adjustable='datalim')
        geometry.legend(loc='upper left', fontsize=8)
        landscape = axes[1, column]
        grid = np.asarray(scene['parameters'])
        landscape.fill_between(grid, 0., 1., where=scene['opposed_secants'],
                               transform=landscape.get_xaxis_transform(), color='#ef4444', alpha=.12,
                               label='Opposing 1D secants', linewidth=0)
        landscape.plot(grid, scene['oracle_energy'], color='#d97706', linewidth=2,
                       label='Exact point-set UDF energy')
        landscape.plot(grid, scene['anchor_smooth_rms_mm'], color='#2563eb', linewidth=1.8,
                       label='Same-anchor pose target (smooth RMS)')
        selected = min(scene['local_minima'], key=lambda row: abs(row['parameter'] - example))
        landscape.scatter([selected['parameter']], [selected['oracle_energy']], color='#dc2626', s=45, zorder=5)
        landscape.annotate(f'Nonreference slice minimum\nD = {selected["anchor_d_mm"]:.2f} mm',
                           xy=(selected['parameter'], selected['oracle_energy']), xytext=(.52, .54),
                           textcoords='axes fraction', fontsize=9,
                           arrowprops={'arrowstyle': '->', 'color': '#dc2626'})
        landscape.axvline(0., color='#6b7280', linewidth=.8, linestyle=':')
        landscape.set(xlabel=('x translation (mm)' if scene['mode'] == 'translation' else 'z rotation (degrees)'),
                      ylabel='Energy / target (mm)', xlim=(grid[0], grid[-1]))
        landscape.grid(alpha=.16)
        landscape.legend(loc='upper left', fontsize=8)
    figure.supxlabel('Exact symmetry also defeats a learned pointwise task world-field. '
                     'Minima shown here are 1D slice minima, not SE(3) certificates.', fontsize=9)
    figure.savefig(output_dir / 'landscapes.png', dpi=180)
    figure.savefig(output_dir / 'landscapes.svg')
    plt.close(figure)


def run_probe(output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repeated = scan_landscape(make_repeated_cusps(), np.linspace(-6., 6., 601),
                              mode='translation', secant_step=.02)
    repeated.update(name='irregular_repeated_peaks', title='A. Repeated, irregular peaks: a wrong basin')
    ring_points = make_symmetric_ring()
    ring = scan_landscape(ring_points, np.linspace(-45., 45., 901), mode='rotation', secant_step=.1)
    ring.update(name='exact_symmetric_ring', title='B. Exact symmetry: an identifiability limit')
    rotated = move_points(ring_points, 30., 'rotation')

    def arbitrary_scalar(points):
        return .1 * points[:, 0] ** 3 + np.cos(.7 * points[:, 0] + .3 * points[:, 1]) + points[:, 1]

    report = dict(schema_version=1, toy_only=True, clinical_data=False, learned_models=False,
        interpretation='A deterministic analytic counterexample, not evidence of medical accuracy or learned-method benefit.',
        distance_definition='Exact unsigned distance to the finite saved reference point set, evaluated by a KD-tree; not a continuous mesh-distance claim.',
        energy=dict(fit='mean(sqrt(d^2+0.1^2)-0.1)', coverage='mean(sigmoid((2-d)/0.25))',
                    coverage_weight_mm=1., roi_penalty=0., roi_note='Unbounded toy world; no cropping or ROI penalty.'),
        target=dict(definition='sqrt(mean(||T*x-x||^2)+0.1^2)-0.1', epsilon_mm=.1,
                    note='Point identities are preserved; this differs from nearest-neighbor surface distance.'),
        scenes=[repeated, ring], symmetry_control=dict(period_degrees=30.,
            set_displacement_max_mm=float(cKDTree(ring_points).query(rotated)[0].max()),
            same_anchor_displacement_mm=float(np.linalg.norm(rotated - ring_points, axis=1).mean()),
            example_scalar_field='0.1*x^3 + cos(0.7*x + 0.3*y) + y',
            example_scalar_energy_difference=float(abs(arbitrary_scalar(rotated).mean() - arbitrary_scalar(ring_points).mean())),
            conclusion='A symmetry that permutes the same point set leaves every unordered pointwise scalar-field aggregation unchanged; a task world-field alone cannot resolve it.'),
        artifacts=dict(png=str(output_dir / 'landscapes.png'), svg=str(output_dir / 'landscapes.svg'),
                       json=str(output_dir / 'report.json')))
    _plot(report, output_dir)
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, default=Path(tempfile.gettempdir()) / 'rtr-registration-geometry-probe',
                        help='Artifact directory (default: system temp directory / rtr-registration-geometry-probe)')
    args = parser.parse_args()
    report = run_probe(args.output_dir)
    for scene in report['scenes']:
        diagnostics = scene['direction_diagnostics']
        print(f"{scene['name']}: {diagnostics['opposed_count']}/{diagnostics['eligible_count']} "
              f"sampled nonstationary positions have opposing secants")
    print(f"TOY / ANALYTIC only. Wrote {report['artifacts']['json']}")


if __name__ == '__main__':
    main()
