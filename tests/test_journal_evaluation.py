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


def test_paired_comparison_matches_patient_on_the_complete_shared_cohort():
    records = [record('a', method='base', error=4), record('b', method='base', error=6),
               record('a', method='new', error=2), record('b', method='new', error=4)]
    report = evaluate_records(records, bootstrap_samples=30)
    comparison = report['paired_comparisons'][0]
    assert comparison['method_a'] == 'base'
    assert comparison['method_b'] == 'new'
    assert comparison['n_common_patients'] == 2
    assert comparison['final_selected_d_mm_b_minus_a']['mean'] == pytest.approx(-2.)
    assert comparison['final_selected_d_mm_b_minus_a']['ci95'] == [-2., -2.]


def test_methods_with_different_case_sets_cannot_silently_compare_the_intersection():
    records = [record('easy', method='base', error=0), record('hard', method='base', error=9),
               record('easy', method='new', error=0)]
    with pytest.raises(ValueError, match='cohort'):
        evaluate_records(records, bootstrap_samples=10)


def test_expected_cohort_requires_missing_cases_even_for_a_single_method():
    expected = [record('easy'), record('hard')]
    with pytest.raises(ValueError, match='missing'):
        evaluate_records([record('easy')], expected_cohort=expected, bootstrap_samples=10)


def test_explicit_missing_predictions_are_failures_including_an_absent_method():
    expected = [record('easy'), record('hard')]
    report = evaluate_records([record('easy', error=0)], expected_cohort=expected,
                              expected_methods=['field', 'absent'], missing_as_failure=True,
                              bootstrap_samples=10)
    assert report['methods']['field']['n_cases'] == 2
    assert report['methods']['field']['failure_rate']['mean'] == .5
    assert report['methods']['absent']['n_cases'] == 2
    assert report['methods']['absent']['invalid_output_rate']['mean'] == 1.
    assert report['paired_comparisons'][0]['n_common_cases'] == 2
    missing = [row for row in report['cases'] if row['failure_reason'] == 'missing_prediction']
    assert len(missing) == 3
    assert all(row['failed'] and row['registration_failed'] for row in missing)
    assert all(row['final_selected_d_mm'] is None for row in missing)
    assert report['cohort']['missing_as_failure']
    assert report['cohort']['n_expected_cases'] == 2
    assert report['cohort']['expected_methods'] == ['absent', 'field']


def test_missing_as_failure_requires_an_external_expected_cohort():
    with pytest.raises(ValueError, match='expected_cohort'):
        evaluate_records([record()], missing_as_failure=True, bootstrap_samples=10)


def test_empty_predictions_can_only_be_scored_against_explicit_case_and_method_rosters():
    report = evaluate_records([], expected_cohort=[record()], expected_methods=['field'],
                              missing_as_failure=True, bootstrap_samples=10)
    assert report['methods']['field']['failure_rate']['mean'] == 1.
    with pytest.raises(ValueError, match='expected_methods'):
        evaluate_records([], expected_cohort=[record()], missing_as_failure=True, bootstrap_samples=10)


@pytest.mark.parametrize('changed', [dict(patient_id='wrong'), dict(reference_kind='silver'),
                                   dict(case_id='unexpected'), dict(source='unexpected')])
def test_expected_cohort_rejects_identity_reference_and_extra_case_mismatches(changed):
    observed = dict(record(), **changed)
    with pytest.raises(ValueError, match='cohort|reference_kind|patient'):
        evaluate_records([observed], expected_cohort=[record()], missing_as_failure=True,
                         bootstrap_samples=10)


def test_expected_method_roster_rejects_extra_or_silently_absent_methods():
    with pytest.raises(ValueError, match='method'):
        evaluate_records([record(method='extra')], expected_cohort=[record()],
                         expected_methods=['field'], missing_as_failure=True, bootstrap_samples=10)
    with pytest.raises(ValueError, match='missing'):
        evaluate_records([record()], expected_methods=['field', 'absent'], bootstrap_samples=10)


def test_expected_cohort_and_method_rosters_reject_duplicates():
    with pytest.raises(ValueError, match='duplicate'):
        evaluate_records([record()], expected_cohort=[record(), record()], bootstrap_samples=10)
    with pytest.raises(ValueError, match='duplicate'):
        evaluate_records([record()], expected_methods=['field', 'field'], bootstrap_samples=10)


def test_risk_coverage_includes_all_ties_at_a_threshold_and_one_missing_score_tail():
    rows = [record('a', error=0), record('b', error=8),
            dict(record('c', error=0), confidence=None), dict(record('d', error=8), confidence=None)]
    report = evaluate_records(rows, bootstrap_samples=10)
    curve = report['risk_coverage']['field']
    assert len(curve) == 2
    assert [(row['coverage'], row['confidence_threshold'], row['risk']) for row in curve] == [
        (.5, .9, .5), (1., None, .5)]
    assert [row['n_cases'] for row in curve] == [2, 4]
    assert curve[-1]['includes_missing_confidence']
    assert not curve[0]['includes_missing_confidence']


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


def test_cli_manifest_filters_define_the_denominator_and_detect_omitted_predictions(tmp_path):
    manifest_rows = []
    for patient, split, source, reference in [
            ('easy', 'test', 'a', 'manual'), ('missing', 'test', 'a', 'manual'),
            ('validation', 'val', 'a', 'manual'), ('silver', 'test', 'a', 'silver'),
            ('other_source', 'test', 'b', 'manual')]:
        manifest_rows.append(dict(patient_id=patient, case_id=patient, jaw='upper',
                                  split=split, source=source, reference_kind=reference,
                                  npz_path=patient + '.npz'))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1, records=manifest_rows)))
    # The evaluator must not require local NPZ files or prediction split fields.
    predictions = [dict(record(row['patient_id'], error=0), source=row['source'],
                        reference_kind=row['reference_kind'])
                   for row in manifest_rows if row['patient_id'] != 'missing']
    source = tmp_path / 'predictions.json'
    source.write_text(json.dumps(dict(schema_version=1, records=predictions)))
    output = tmp_path / 'report.json'
    script = Path(__file__).resolve().parents[1] / 'scripts/evaluate_journal_registration.py'
    command = [sys.executable, str(script), '--input', str(source), '--output', str(output),
               '--expected-manifest', str(manifest), '--split', 'test', '--source', 'a',
               '--reference-kind', 'manual', '--expected-method', 'field', '--bootstrap-samples', '10']
    strict = subprocess.run(command, capture_output=True, text=True)
    assert strict.returncode != 0
    assert 'missing' in strict.stderr
    assert not output.exists()
    result = subprocess.run(command + ['--missing-as-failure'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report['methods']['field']['n_cases'] == 2
    assert report['methods']['field']['failure_rate']['mean'] == .5
    assert {row['patient_id'] for row in report['cases']} == {'easy', 'missing'}
    predictions.append(dict(record('unknown'), source='a', reference_kind='manual'))
    source.write_text(json.dumps(dict(schema_version=1, records=predictions)))
    extra = subprocess.run(command + ['--missing-as-failure'], capture_output=True, text=True)
    assert extra.returncode != 0
    assert 'cohort' in extra.stderr


def test_cli_empty_predictions_with_rosters_count_the_whole_cohort_as_failed(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1, records=[dict(
        patient_id='p1', case_id='p1', jaw='upper', split='test', source='a',
        reference_kind='manual', npz_path='absent.npz')])))
    source = tmp_path / 'predictions.json'
    source.write_text(json.dumps(dict(schema_version=1, records=[])))
    output = tmp_path / 'report.json'
    script = Path(__file__).resolve().parents[1] / 'scripts/evaluate_journal_registration.py'
    result = subprocess.run([sys.executable, str(script), '--input', str(source), '--output', str(output),
                             '--expected-manifest', str(manifest), '--split', 'test',
                             '--expected-method', 'field', '--missing-as-failure',
                             '--bootstrap-samples', '10'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report['methods']['field']['n_cases'] == 1
    assert report['methods']['field']['failure_rate']['mean'] == 1.
