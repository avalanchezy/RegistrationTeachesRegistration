"""Controlled convergence tests; analytic fields exercise the real optimizer."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from task2reg.journal.config import TrainingConfig
from task2reg.journal.field import RegistrationField
from task2reg.journal.landscape import (
    benchmark_field, benchmark_landscape, generate_perturbations, summarize_landscape,
)


def identity(**overrides):
    return dict(patient_id='patient-1', source='hospital', case_id='case-1',
                jaw='upper', reference_kind='manual', **overrides)


def geometry():
    points = np.array([[-2., -1., 0.], [2., -1., 0.], [0., 2., 0.]])
    affine = np.eye(4)
    affine[:3, 3] = -10.
    return dict(points=points, anchors=points.copy(), reference_transform=np.eye(4),
                affine=affine, input_shape=(21, 21, 21))


def starts(record, **overrides):
    options = dict(rotation_degrees=[0., 17.], translation_mm=[0., 3.], repeats=2, seed=31)
    options.update(overrides)
    return generate_perturbations(record, np.eye(4), geometry()['anchors'], **options)


def test_starts_depend_on_identity_and_seed_not_manifest_order_or_grid_order():
    a = identity()
    b = identity() | {'case_id': 'case-2'}
    forward = {r['case_id']: starts(r) for r in [a, b]}
    backward = {r['case_id']: starts(r, rotation_degrees=[17., 0.], translation_mm=[3., 0.])
                for r in [b, a]}
    assert forward == backward
    assert forward[a['case_id']] != forward[b['case_id']]
    assert forward[a['case_id']] != starts(a, seed=32)


@pytest.mark.parametrize('parity', [1., -1.])
def test_physical_grid_keeps_centroid_translation_magnitude_and_reference_parity(parity):
    reference = np.eye(4)
    reference[0, 0] = parity
    reference[:3, 3] = [140., -79., 55.]
    anchors = geometry()['anchors'] + [4., 9., -7.]
    trials = generate_perturbations(identity(), reference, anchors, rotation_degrees=[35.],
                                   translation_mm=[3.], repeats=3, seed=8)
    center = (anchors @ reference[:3, :3].T + reference[:3, 3]).mean(0)
    for trial in trials:
        candidate = np.asarray(trial['initial_transform'])
        moved_center = (anchors @ candidate[:3, :3].T + candidate[:3, 3]).mean(0)
        assert np.linalg.norm(moved_center - center) == pytest.approx(3., abs=1e-10)
        relative = candidate[:3, :3] @ reference[:3, :3].T
        assert np.degrees(np.arccos((np.trace(relative) - 1.) / 2.)) == pytest.approx(35.)
        assert np.linalg.det(candidate[:3, :3]) == pytest.approx(parity)
        np.testing.assert_allclose(relative @ relative.T, np.eye(3), atol=1e-12)


def test_physical_starts_are_invariant_to_scanner_origin_and_ios_origin_shift():
    anchors = geometry()['anchors'] + [3., 9., 8.]
    reference = np.eye(4)
    reference[:3, 3] = [200., -31., 47.]
    kwargs = dict(rotation_degrees=[27.], translation_mm=[4.], repeats=2, seed=5)
    ordinary = generate_perturbations(identity(), reference, anchors, **kwargs)
    offset = np.array([1234., -434., 67.])
    shifted_reference = reference.copy()
    shifted_reference[:3, 3] += offset
    shifted = generate_perturbations(identity(), shifted_reference, anchors, **kwargs)
    ios_offset = np.array([100., -300., 9.])
    ios_reference = reference.copy()
    ios_reference[:3, 3] -= ios_offset
    ios_shifted = generate_perturbations(identity(), ios_reference, anchors + ios_offset, **kwargs)
    for a, b, c in zip(ordinary, shifted, ios_shifted):
        ta, tb, tc = [np.asarray(row['initial_transform']) for row in [a, b, c]]
        np.testing.assert_allclose(tb[:3, :3], ta[:3, :3], atol=1e-12)
        np.testing.assert_allclose(tb[:3, 3] - offset, ta[:3, 3], atol=1e-10)
        np.testing.assert_allclose(anchors @ ta[:3, :3].T + ta[:3, 3],
                                   (anchors + ios_offset) @ tc[:3, :3].T + tc[:3, 3], atol=1e-10)


@pytest.mark.parametrize('parity', [1., -1.])
def test_real_analytic_field_refinement_improves_displacement_and_preserves_parity(parity):
    case = geometry()
    case['reference_transform'][0, 0] = parity
    points = torch.tensor(case['points'] @ case['reference_transform'][:3, :3].T)

    def field(xyz):
        return torch.cdist(xyz, points.to(xyz)[None]).amin(-1)

    rows = benchmark_field(field, **case, record=identity(), method='analytic',
                           rotation_degrees=[0.], translation_mm=[1.5], repeats=2,
                           refinement_steps=8, learning_rate=.3, seed=12)
    assert len(rows) == 2
    for row in rows:
        assert not row['failed'], row['failure_reason']
        assert row['final_d_mm'] < row['initial_d_mm'] * .6
        assert row['final_energy'] < row['initial_energy']
        assert row['improved'] and not row['worsened']
        assert np.linalg.det(np.asarray(row['final_transform'])[:3, :3]) == pytest.approx(parity)


def test_invalid_field_retains_every_trial_and_initial_error_in_denominator():
    rows = benchmark_field(lambda xyz: xyz[..., 0] * float('nan'), **geometry(),
                           record=identity(), method='invalid', rotation_degrees=[0., 10.],
                           translation_mm=[3.], repeats=3, refinement_steps=1, seed=4)
    assert len(rows) == 6
    assert all(row['failed'] and not row['success'] and row['initial_d_mm'] > 0 for row in rows)
    assert all(row['final_d_mm'] is None for row in rows)
    report = summarize_landscape(rows, bootstrap_samples=30, seed=2)
    assert len(report['grid']) == 2
    for cell in report['grid']:
        assert cell['n_trials'] == cell['n_failed_trials'] == 3
        assert cell['success_rate']['mean'] == 0.
        assert cell['failure_rate']['mean'] == 1.
        assert cell['final_d_mm']['mean'] is None
        assert cell['capture_rate']['mean'] == 0.


def trial_row(patient, case, *, final_d, failed=False, method='field'):
    return dict(patient_id=patient, case_id=case, source='hospital', jaw='upper', method=method,
                reference_kind='manual', rotation_deg=0., translation_mm=4., repeat=0,
                trial_id=f'{case}:0', initial_transform=[[1., 0, 0, 4.], [0, 1., 0, 0], [0, 0, 1., 0], [0, 0, 0, 1.]],
                point_indices_sha256='a' * 64, reference_evidence_sha256='b' * 64,
                initial_d_mm=4., final_d_mm=final_d,
                delta_d_mm=None if failed else final_d - 4., initial_energy=4.,
                final_energy=final_d, delta_energy=None if failed else final_d - 4.,
                failed=failed, success=not failed and final_d <= 2., initial_success=False,
                improved=not failed and final_d < 4., worsened=not failed and final_d > 4.,
                unchanged=not failed and final_d == 4.)


def test_summary_averages_cases_within_patients_before_bootstrapping():
    rows = [trial_row('patient-a', f'a{i}', final_d=0.) for i in range(3)]
    rows.append(trial_row('patient-b', 'b1', final_d=4.))
    report = summarize_landscape(rows, bootstrap_samples=100, seed=2)
    cell = report['grid'][0]
    assert cell['n_patients'] == 2 and cell['n_trials'] == 4
    assert cell['final_d_mm']['mean'] == pytest.approx(2.)
    assert cell['success_rate']['mean'] == pytest.approx(.5)
    assert cell['final_d_mm']['n_patients'] == 2
    assert cell['final_d_mm']['ci95'] == [0., 4.]


def test_paired_summary_retains_failure_comparison_and_rejects_mismatched_start():
    a = trial_row('p', 'case', final_d=0., method='dense')
    b = trial_row('p', 'case', final_d=None, failed=True, method='implicit')
    report = summarize_landscape([a, b], bootstrap_samples=20)
    paired = report['paired_comparisons'][0]
    assert paired['n_common_patients'] == 1
    assert paired['failure_rate_b_minus_a']['mean'] == 1.
    assert paired['final_d_mm_b_minus_a']['n_patients'] == 0
    with pytest.raises(ValueError, match='initial|paired'):
        summarize_landscape([a, b | {'initial_d_mm': 8.}], bootstrap_samples=20)


def experiment(tmp_path, *, split='test', reference_kind='manual'):
    case = geometry()
    npz = tmp_path / 'case.npz'
    np.savez(npz, image=np.zeros((16, 16, 16), np.int16),
             affine=case['affine'], points=case['points'], anchors=case['anchors'],
             transform=case['reference_transform'])
    record = identity() | dict(npz_path=npz.name, split=split, reference_kind=reference_kind)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1, records=[record])))
    checkpoints = {}
    for mode in ['dense', 'implicit']:
        torch.manual_seed(3)
        config = TrainingConfig(mode=mode, base_channels=4, device='cpu', amp=False)
        checkpoint = tmp_path / f'{mode}.pt'
        torch.save(dict(format='rtr-journal-field-v1', config=config.as_dict(),
                        model=RegistrationField(4, mode).state_dict(), training_patient_ids=['train-p'],
                        training_content_hashes=[], training_sources=['train-source'],
                        provenance_schema_version=2, selection_patient_ids=[],
                        selection_content_hashes=[], selection_sources=[]), checkpoint)
        checkpoints[mode] = checkpoint
    return manifest, checkpoints


def test_cli_compares_dense_implicit_from_identical_starts_with_replay_metadata(tmp_path):
    manifest, checkpoints = experiment(tmp_path)
    output = tmp_path / 'report'
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts/benchmark_journal_landscape.py'),
               '--manifest', str(manifest), '--output-dir', str(output), '--device', 'cpu',
               '--rotation-degrees', '0', '15', '--translation-mm', '1.5', '--repeats', '1',
               '--refinement-steps', '1', '--point-budget', '3', '--bootstrap-samples', '10']
    for method, checkpoint in checkpoints.items():
        command.extend(['--checkpoint', f'{method}={checkpoint}'])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / 'report.json').read_text())
    trials = report['trials']
    assert len(trials) == 4 and len(report['paired_comparisons']) == 2
    dense = [r for r in trials if r['method'] == 'dense']
    implicit = [r for r in trials if r['method'] == 'implicit']
    assert [r['initial_transform'] for r in dense] == [r['initial_transform'] for r in implicit]
    assert report['protocol']['diagnostic_only']
    assert report['protocol']['checkpoints']['dense']['sha256']
    assert report['protocol']['manifest_fingerprint']
    assert report['protocol']['records'][0]['payload_sha256']
    assert report['protocol']['records'][0]['point_indices'] == [0, 1, 2]
    assert dense[0]['point_indices_sha256'] == implicit[0]['point_indices_sha256']
    assert report['protocol']['runtime']['git_revision']


@pytest.mark.parametrize('leak', ['patient', 'content', 'external_source'])
def test_entry_rejects_checkpoint_training_leakage(tmp_path, leak):
    manifest, checkpoints = experiment(tmp_path, split='external_test')
    checkpoint = checkpoints['dense']
    saved = torch.load(checkpoint, weights_only=False)
    if leak == 'patient':
        saved['training_patient_ids'] = ['patient-1']
    elif leak == 'content':
        records = json.loads(manifest.read_text())
        records['records'][0]['content_hash'] = 'shared-image'
        manifest.write_text(json.dumps(records))
        saved['training_content_hashes'] = ['shared-image']
    else:
        saved['training_sources'] = ['hospital']
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match='train'):
        benchmark_landscape(manifest, {'dense': checkpoint}, tmp_path / 'report',
                            split='external_test', device='cpu', rotation_degrees=[0.],
                            translation_mm=[0.], repeats=1, refinement_steps=0, bootstrap_samples=10)


def test_entry_rejects_pooling_manual_and_silver_references(tmp_path):
    manifest, checkpoints = experiment(tmp_path)
    payload = json.loads(manifest.read_text())
    payload['records'].append(payload['records'][0] | dict(case_id='other', patient_id='p-other',
                                                         reference_kind='silver'))
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='reference_kind|manual.*silver'):
        benchmark_landscape(manifest, checkpoints, tmp_path / 'report', device='cpu')


def test_failure_after_finite_initial_energy_retains_initial_measurement():
    # Unsigned distance has a cusp with undefined sqrt derivative at z=0.
    # This exercises a real forward-finite/backward-NaN field, not a mocked runner.
    rows = benchmark_field(lambda xyz: xyz[..., 2].square().sqrt(), **geometry(),
                           record=identity(), method='cusp', rotation_degrees=[0.],
                           translation_mm=[0.], repeats=1, refinement_steps=1)
    assert rows[0]['failed']
    assert rows[0]['initial_energy'] is not None
    assert np.isfinite(rows[0]['initial_energy'])
    assert rows[0]['initial_d_mm'] == 0.
    assert rows[0]['final_energy'] is None


def test_partial_numerical_failures_do_not_abort_later_starts():
    def half_space_field(xyz):
        # A limited-domain analytic field is invalid for one translation hemisphere.
        values = torch.linalg.vector_norm(xyz, dim=-1)
        return torch.where(xyz[..., 2].mean() >= 0., values, values * float('nan'))

    rows = benchmark_field(half_space_field, **geometry(), record=identity(), method='partial',
                           rotation_degrees=[0.], translation_mm=[3.], repeats=8,
                           refinement_steps=0, seed=4)
    failures = sum(row['failed'] for row in rows)
    assert 0 < failures < 8
    assert len(rows) == 8 and [row['repeat'] for row in rows] == list(range(8))
    cell = summarize_landscape(rows, bootstrap_samples=10)['grid'][0]
    assert cell['n_trials'] == 8 and cell['n_failed_trials'] == failures
    assert cell['failure_rate']['mean'] == failures / 8


@pytest.mark.parametrize('options', [dict(repeats=0), dict(rotation_degrees=[181.]),
                                    dict(rotation_degrees=[0., 0.]), dict(translation_mm=[float('nan')])])
def test_invalid_physical_grid_is_rejected(options):
    with pytest.raises(ValueError):
        starts(identity(), **options)


def test_comparison_rejects_missing_method_trials_instead_of_silently_intersecting():
    rows = [trial_row('p1', 'case1', final_d=0., method='dense'),
            trial_row('p1', 'case1', final_d=1., method='implicit'),
            trial_row('p2', 'case2', final_d=3., method='dense')]
    with pytest.raises(ValueError, match='same|cohort|missing'):
        summarize_landscape(rows, bootstrap_samples=10)


@pytest.mark.parametrize('evidence', ['initial_transform', 'point_indices_sha256', 'reference_evidence_sha256'])
def test_comparison_rejects_absent_start_or_reference_evidence(evidence):
    rows = [trial_row('p1', 'case1', final_d=0., method=method) for method in ['dense', 'implicit']]
    for row in rows:
        row.pop(evidence, None)
    with pytest.raises(ValueError, match='evidence|paired'):
        summarize_landscape(rows, bootstrap_samples=10)


@pytest.mark.parametrize('leak', ['patient', 'content', 'external_source', 'unknown'])
def test_locked_diagnosis_rejects_checkpoint_selection_leakage_or_unknown_history(tmp_path, leak):
    manifest, checkpoints = experiment(tmp_path, split='external_test')
    checkpoint = checkpoints['dense']
    saved = torch.load(checkpoint, weights_only=False)
    if leak == 'patient':
        saved['selection_patient_ids'] = ['patient-1']
    elif leak == 'content':
        records = json.loads(manifest.read_text())
        records['records'][0]['content_hash'] = 'selected-image'
        manifest.write_text(json.dumps(records))
        saved['selection_content_hashes'] = ['selected-image']
    elif leak == 'external_source':
        saved['selection_sources'] = ['hospital']
    else:
        saved.pop('provenance_schema_version', None)
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match='selection|provenance'):
        benchmark_landscape(manifest, {'dense': checkpoint}, tmp_path / 'report',
                            split='external_test', device='cpu', rotation_degrees=[0.],
                            translation_mm=[0.], repeats=1, refinement_steps=0, bootstrap_samples=10)


def test_comparison_releases_previous_model_and_encoder_context_before_next_load(tmp_path, monkeypatch):
    # Real CPU models/tensors expose the same reference lifetime as CUDA allocations.
    import weakref
    import task2reg.journal.landscape as landscape
    manifest, checkpoints = experiment(tmp_path)
    model_refs, context_refs = [], []
    real_load, real_encode = landscape.load_field, landscape.encode_case

    def load(*args, **kwargs):
        assert not any(ref() is not None for ref in model_refs), 'previous method still occupies memory'
        result = real_load(*args, **kwargs)
        model_refs.append(weakref.ref(result[0]))
        return result

    def remember(value):
        if isinstance(value, torch.Tensor):
            context_refs.append(weakref.ref(value))
        elif isinstance(value, dict):
            for item in value.values():
                remember(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                remember(item)

    def encode(*args, **kwargs):
        assert not any(ref() is not None for ref in context_refs), 'previous case context still occupies memory'
        result = real_encode(*args, **kwargs)
        remember(result)
        return result

    monkeypatch.setattr(landscape, 'load_field', load)
    monkeypatch.setattr(landscape, 'encode_case', encode)
    report = benchmark_landscape(manifest, checkpoints, tmp_path / 'report', device='cpu',
                                 rotation_degrees=[0.], translation_mm=[0.], repeats=1,
                                 refinement_steps=0, bootstrap_samples=10)
    assert len(report['trials']) == 2
    assert not any(ref() is not None for ref in model_refs + context_refs)


def test_legacy_selection_history_is_allowed_only_as_descriptive_validation(tmp_path):
    manifest, checkpoints = experiment(tmp_path, split='val')
    checkpoint = checkpoints['dense']
    saved = torch.load(checkpoint, weights_only=False)
    saved.pop('provenance_schema_version')
    torch.save(saved, checkpoint)
    report = benchmark_landscape(manifest, {'dense': checkpoint}, tmp_path / 'report',
                                 split='val', device='cpu', rotation_degrees=[0.],
                                 translation_mm=[0.], repeats=1, refinement_steps=0, bootstrap_samples=10)
    assert report['protocol']['cohort_role'] == 'descriptive_validation'
    assert report['protocol']['checkpoints']['dense']['selection_provenance_known'] is False
    assert report['protocol']['checkpoints']['dense']['selection_patient_ids'] is None


def test_trials_report_actual_gradient_alignment_and_skip_gt_direction_denominator():
    target = torch.tensor(geometry()['points'])
    field = lambda xyz: torch.cdist(xyz, target.to(xyz)[None]).amin(-1)
    rows = benchmark_field(field, **geometry(), record=identity(), method='analytic',
                           rotation_degrees=[0.], translation_mm=[0., 1.], repeats=1,
                           refinement_steps=0)
    stationary, displaced = rows
    assert stationary['initial_gradient_cosine'] is None
    assert stationary['initial_gradient_descent'] is None
    assert displaced['initial_gradient_cosine'] > .99
    assert displaced['initial_gradient_descent'] is True
    report = summarize_landscape(rows, bootstrap_samples=10)
    cells = {row['translation_mm']: row for row in report['grid']}
    assert cells[0.]['n_gradient_comparable_trials'] == 0
    assert cells[0.]['initial_gradient_descent_rate']['mean'] is None
    assert cells[1.]['n_gradient_comparable_trials'] == 1
    assert cells[1.]['initial_gradient_descent_rate']['mean'] == 1.


def test_gradient_summary_weights_patients_and_retains_the_valid_direction_denominator():
    rows = [trial_row('a', f'a{i}', final_d=1.) |
            dict(initial_gradient_cosine=1., initial_gradient_descent=True) for i in range(3)]
    rows.append(trial_row('b', 'b1', final_d=1.) |
                dict(initial_gradient_cosine=-1., initial_gradient_descent=False))
    rows.append(trial_row('b', 'b0', final_d=0.) |
                dict(initial_gradient_cosine=None, initial_gradient_descent=None))
    cell = summarize_landscape(rows, bootstrap_samples=10)['grid'][0]
    assert cell['n_trials'] == 5
    assert cell['n_gradient_comparable_trials'] == 4
    assert cell['n_gradient_cosine_trials'] == 4
    assert cell['initial_gradient_cosine']['mean'] == pytest.approx(0.)
    assert cell['initial_gradient_descent_rate']['mean'] == pytest.approx(.5)


def test_diagnostic_gradient_failure_does_not_change_zero_step_registration_result():
    rows = benchmark_field(lambda xyz: xyz[..., 2].square().sqrt(), **geometry(),
                           record=identity(), method='cusp', rotation_degrees=[0.],
                           translation_mm=[0.], repeats=1, refinement_steps=0)
    assert not rows[0]['failed']
    assert rows[0]['final_d_mm'] == pytest.approx(0.)
    assert rows[0]['initial_gradient_failure_reason']
    assert rows[0]['initial_gradient_descent'] is None
    cell = summarize_landscape(rows, bootstrap_samples=10)['grid'][0]
    assert cell['n_gradient_diagnostic_failures'] == 1
    assert cell['n_gradient_comparable_trials'] == 0
