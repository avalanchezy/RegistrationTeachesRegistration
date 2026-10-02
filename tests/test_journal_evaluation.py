import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from task2reg.journal.evaluation import evaluate_records


def transform(x):
    value = np.eye(4)
    value[0, 3] = x
    return value.tolist()


def record(patient='p1', jaw='upper', method='field', error=2.):
    return dict(patient_id=patient, case_id=patient, jaw=jaw, method=method,
                anchors=[[0, 0, 0], [1, 2, 3]], reference_transform=transform(0),
                initial_candidates=[transform(4), transform(1)],
                final_candidates=[transform(error), transform(.5)], selected_index=0,
                confidence=.9)


def test_separates_selected_oracle_regret_and_initial_final_displacement():
    report = evaluate_records([record()], bootstrap_samples=20)
    row = report['cases'][0]
    assert row['initial_selected_d_mm'] == pytest.approx(4.)
    assert row['final_selected_d_mm'] == pytest.approx(2.)
    assert row['initial_oracle_d_mm'] == pytest.approx(1.)
    assert row['final_oracle_d_mm'] == pytest.approx(.5)
    assert row['initial_regret_mm'] == pytest.approx(3.)
    assert row['final_regret_mm'] == pytest.approx(1.5)
    assert row['refinement_delta_mm'] == pytest.approx(-2.)
    assert 'TRE' not in report['metric']


def test_patient_is_bootstrap_unit_and_jaws_do_not_reweight_patients():
    records = [record('a', 'upper', error=0), record('a', 'lower', error=0), record('b', error=9)]
    report = evaluate_records(records, bootstrap_samples=100, seed=7)
    summary = report['methods']['field']
    assert summary['n_patients'] == 2
    assert summary['final_selected_d_mm']['mean'] == pytest.approx(4.5)
    assert summary['final_selected_d_mm']['ci95'] == [0., 9.]
    assert summary['final_selected_d_mm']['n_patients'] == 2
    assert report == evaluate_records(records, bootstrap_samples=100, seed=7)


def test_failed_and_nan_predictions_count_as_failures_in_risk_coverage():
    failed = record('failed')
    failed.update(failed=True, failure_reason='optimizer failure', confidence=1.)
    invalid = record('nan')
    invalid['final_candidates'][0][0][3] = float('nan')
    invalid['confidence'] = .95
    valid = record('valid', error=1)
    report = evaluate_records([failed, invalid, valid], bootstrap_samples=30)
    assert report['methods']['field']['n_failed_cases'] == 2
    assert report['methods']['field']['failure_rate']['mean'] == pytest.approx(2/3)
    assert report['risk_coverage']['field'][0]['risk'] == 1.
    assert report['risk_coverage']['field'][-1]['coverage'] == 1.
    assert report['risk_coverage']['field'][-1]['risk'] == pytest.approx(2/3)
    assert report['risk_coverage']['field'][-1]['n_failed_cases'] == 2
    json.dumps(report, allow_nan=False)


def test_paired_comparison_matches_patient_and_counts_unpaired_failures():
    records = [record('a', method='base', error=4), record('b', method='base', error=6),
               record('a', method='new', error=2), record('b', method='new', error=4),
               record('c', method='new', error=20)]
    report = evaluate_records(records, bootstrap_samples=30)
    comparison = report['paired_comparisons'][0]
    assert comparison['method_a'] == 'base'
    assert comparison['method_b'] == 'new'
    assert comparison['n_common_patients'] == 2
    assert comparison['final_selected_d_mm_b_minus_a']['mean'] == pytest.approx(-2.)
    assert comparison['final_selected_d_mm_b_minus_a']['ci95'] == [-2., -2.]


def test_missing_anchors_fail_closed_and_duplicate_observations_raise():
    value = record()
    del value['anchors']
    report = evaluate_records([value], bootstrap_samples=10)
    assert report['cases'][0]['failed']
    assert report['methods']['field']['final_selected_d_mm']['mean'] is None
    with pytest.raises(ValueError, match='duplicate'):
        evaluate_records([record(), record()], bootstrap_samples=10)


def test_cli_reads_schema_and_writes_strict_json(tmp_path):
    source = tmp_path / 'predictions.json'
    output = tmp_path / 'report.json'
    source.write_text(json.dumps({'schema_version': 1, 'records': [record()]}))
    script = Path(__file__).resolve().parents[1] / 'scripts/evaluate_journal_registration.py'
    result = subprocess.run([sys.executable, str(script), '--input', str(source), '--output', str(output),
                             '--bootstrap-samples', '10'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())['methods']['field']['n_patients'] == 1


def test_source_namespaces_case_identity_and_patient_consistency():
    first = record('a')
    second = record('b')
    first.update(source='hospital_a', case_id='case001')
    second.update(source='hospital_b', case_id='case001')
    report = evaluate_records([first, second], bootstrap_samples=10)
    assert report['methods']['field']['n_patients'] == 2
    conflicting = dict(first, method='other', patient_id='different_patient')
    with pytest.raises(ValueError, match='patient'):
        evaluate_records([first, conflicting], bootstrap_samples=10)


def test_coplanar_zero_error_wrong_parity_is_registration_failure():
    value = record()
    value['anchors'] = [[0, 0, 0], [1, 2, 0], [2, -1, 0]]
    reflection = np.diag([1., 1., -1., 1.]).tolist()
    value['initial_candidates'] = [reflection]
    value['final_candidates'] = [reflection]
    report = evaluate_records([value], bootstrap_samples=10)
    row = report['cases'][0]
    assert row['final_selected_d_mm'] == pytest.approx(0.)
    assert row['parity_mismatch']
    assert row['registration_failed']
    assert not row['failed']
    assert row['final_rotation_deg'] is None
    summary = report['methods']['field']
    assert summary['invalid_output_rate']['mean'] == 0.
    assert summary['registration_failure_rate']['mean'] == 1.
    assert summary['failure_rate_1mm']['mean'] == 1.
    assert report['risk_coverage']['field'][-1]['risk'] == 1.


def test_finite_large_displacement_counts_failure_and_thresholds_separately():
    report = evaluate_records([record(error=2.5)], bootstrap_samples=10)
    summary = report['methods']['field']
    assert summary['invalid_output_rate']['mean'] == 0.
    assert summary['registration_failure_rate']['mean'] == 1.
    assert summary['failure_rate']['mean'] == 1.
    assert summary['failure_rate_1mm']['mean'] == 1.
    assert summary['failure_rate_2mm']['mean'] == 1.
    assert summary['failure_rate_3mm']['mean'] == 0.
    assert summary['n_failed_cases'] == 1
    assert summary['n_invalid_cases'] == 0
    assert report['cases'][0]['final_rotation_deg'] == pytest.approx(0.)


def test_rotation_angle_for_same_improper_parity_remains_defined():
    value = record()
    reference = np.diag([-1., 1., 1., 1.])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    predicted = reference.copy()
    predicted[:3, :3] = rotation @ reference[:3, :3]
    value['reference_transform'] = reference.tolist()
    value['initial_candidates'] = [reference.tolist()]
    value['final_candidates'] = [predicted.tolist()]
    row = evaluate_records([value], bootstrap_samples=10)['cases'][0]
    assert not row['parity_mismatch']
    assert row['initial_rotation_deg'] == pytest.approx(0.)
    assert row['final_rotation_deg'] == pytest.approx(90.)


def test_paired_registration_failures_are_not_invalid_output_failures():
    report = evaluate_records([record(method='base', error=2.5), record(method='new', error=1.5)],
                              bootstrap_samples=10)
    pair = report['paired_comparisons'][0]
    assert pair['registration_failure_rate_b_minus_a']['mean'] == -1.
    assert pair['invalid_output_rate_b_minus_a']['mean'] == 0.


@pytest.mark.parametrize('evidence_key', ['anchors', 'reference_transform'])
def test_paired_methods_require_identical_reference_evidence(evidence_key):
    first = record(method='base')
    second = record(method='new')
    second[evidence_key][0][0] += .1
    with pytest.raises(ValueError, match='reference evidence'):
        evaluate_records([first, second], bootstrap_samples=10)


def test_mixed_reference_kinds_cannot_be_silently_aggregated():
    manual = dict(record('manual'), reference_kind='manual')
    silver = dict(record('silver'), reference_kind='silver')
    with pytest.raises(ValueError, match='reference_kind'):
        evaluate_records([manual, silver], bootstrap_samples=10)
    report = evaluate_records([manual], bootstrap_samples=10)
    assert report['cases'][0]['reference_kind'] == 'manual'
    assert report['methods']['field']['reference_kind'] == 'manual'
    assert evaluate_records([record()], bootstrap_samples=10)['methods']['field']['reference_kind'] == 'unspecified'


def test_paired_reference_kind_labels_must_agree():
    first = dict(record(method='base'), reference_kind='manual')
    second = dict(record(method='new'), reference_kind='silver')
    with pytest.raises(ValueError, match='reference_kind'):
        evaluate_records([first, second], bootstrap_samples=10)


def test_cli_reference_and_source_filters_preserve_reference_type(tmp_path):
    records = [dict(record('manual_a'), reference_kind='manual', source='a'),
               dict(record('manual_b'), reference_kind='manual', source='b'),
               dict(record('silver_a'), reference_kind='silver', source='a')]
    source = tmp_path / 'mixed.json'
    output = tmp_path / 'filtered.json'
    source.write_text(json.dumps({'schema_version': 1, 'records': records}))
    script = Path(__file__).resolve().parents[1] / 'scripts/evaluate_journal_registration.py'
    command = [sys.executable, str(script), '--input', str(source), '--output', str(output),
               '--bootstrap-samples', '10', '--reference-kind', 'manual', '--source', 'a']
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert len(report['cases']) == 1
    assert report['cases'][0]['patient_id'] == 'manual_a'
    assert report['methods']['field']['reference_kind'] == 'manual'
    command[-1] = 'absent'
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'No records' in result.stderr
