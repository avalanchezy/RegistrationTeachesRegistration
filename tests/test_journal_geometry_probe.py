"""Analytic counterexamples, not a synthetic model of a clinical population."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree


def test_repeated_cusps_are_nonperiodic_but_have_a_wrong_near_period_minimum():
    from scripts.probe_registration_energy_geometry import make_repeated_cusps, scan_landscape, move_points

    points = make_repeated_cusps()
    assert points.ndim == 2 and points.shape[1] == 3
    assert np.isfinite(points).all() and len(np.unique(points, axis=0)) == len(points)
    shifted = move_points(points, 4., 'translation')
    distances = cKDTree(points).query(shifted)[0]
    assert distances.max() > 3.  # A finite, irregular chain is not translation-symmetric.
    scene = scan_landscape(points, np.linspace(-6., 6., 601), mode='translation', secant_step=.02)
    wrong = [minimum for minimum in scene['local_minima'] if 3.8 < minimum['parameter'] < 4.2]
    assert wrong, 'The exact reference distance field should retain the one-repeat false basin.'
    assert wrong[0]['anchor_d_mm'] > 3.5
    assert wrong[0]['oracle_energy'] > scene['oracle_energy'][300]
    assert scene['direction_diagnostics']['opposed_fraction'] > .1
    # A translation moves every fixed anchor by its exact translation norm.
    assert scene['anchor_d_mm'][500] == pytest.approx(4.)


def test_exact_ring_symmetry_is_indistinguishable_to_any_pointwise_scalar_aggregation():
    from scripts.probe_registration_energy_geometry import make_symmetric_ring, move_points, scan_landscape

    points = make_symmetric_ring(count=12, radius_mm=8.)
    moved = move_points(points, 30., 'rotation')
    assert cKDTree(points).query(moved)[0].max() < 1e-12
    assert np.linalg.norm(moved - points, axis=1).mean() == pytest.approx(16. * np.sin(np.pi / 12.))
    # An unrelated, non-radial world scalar field must also give identical sums.
    def scalar(x):
        return .1 * x[:, 0] ** 3 + np.cos(.7 * x[:, 0] + .3 * x[:, 1]) + x[:, 1]
    assert scalar(moved).mean() == pytest.approx(scalar(points).mean(), abs=1e-12)
    scene = scan_landscape(points, np.linspace(-45., 45., 901), mode='rotation', secant_step=.1)
    assert scene['oracle_energy'][750] == pytest.approx(scene['oracle_energy'][450], abs=1e-12)
    wrong = [minimum for minimum in scene['local_minima'] if 29.8 < minimum['parameter'] < 30.2]
    assert wrong and wrong[0]['anchor_d_mm'] > 4.
    assert scene['anchor_smooth_rms_mm'][750] > 4.


def test_secant_direction_uses_the_same_coordinate_and_excludes_stationary_target():
    from scripts.probe_registration_energy_geometry import scan_landscape

    points = np.array([[0., 0., 0.]])
    scene = scan_landscape(points, np.linspace(-2., 2., 41), mode='translation', secant_step=.02)
    assert scene['direction_diagnostics']['opposed_count'] == 0
    assert scene['direction_diagnostics']['eligible_count'] == 40
    assert scene['target_secant'][20] == pytest.approx(0.)
    assert [row['parameter'] for row in scene['local_minima']] == pytest.approx([0.])


def test_cli_writes_finite_json_and_standalone_labeled_figures(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/probe_registration_energy_geometry.py'
    result = subprocess.run([sys.executable, str(script), '--output-dir', str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / 'report.json').read_text())
    assert len(report['scenes']) == 2
    assert report['scenes'][0]['direction_diagnostics']['opposed_count'] > 0
    assert report['symmetry_control']['set_displacement_max_mm'] < 1e-12
    assert report['symmetry_control']['same_anchor_displacement_mm'] > 4.
    assert report['symmetry_control']['example_scalar_energy_difference'] < 1e-12
    json.dumps(report, allow_nan=False)
    assert (tmp_path / 'landscapes.png').stat().st_size > 1000
    assert (tmp_path / 'landscapes.svg').stat().st_size > 1000


@pytest.mark.parametrize('grid', [[], [0.], [0., 0.], [1., 0.], [0., float('nan')]])
def test_invalid_landscape_grids_are_rejected(grid):
    from scripts.probe_registration_energy_geometry import scan_landscape

    with pytest.raises(ValueError):
        scan_landscape(np.array([[0., 0., 0.]]), grid, mode='translation', secant_step=.1)
