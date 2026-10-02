import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.diagnose_journal_candidates import diagnose_candidates


def fixture(tmp_path, *, sources=('hospital',)):
    candidates = np.repeat(np.eye(4)[None], 2, axis=0)
    candidates[1, 0, 3] = 4.
    records = []
    run = tmp_path / 'old_run'
    for i, source in enumerate(sources):
        case_id = f'legacy{i}'
        points = np.array([[0., 0, 0], [1., 2, 0], [0, 0, 3.]])
        npz = tmp_path / f'{case_id}.npz'
        np.savez(npz, image=np.zeros((2, 2, 2)), affine=np.eye(4), points=points,
                 anchors=points, transform=np.eye(4), candidates=candidates)
        records.append(dict(case_id=f'{source}:{case_id}', original_case_id=case_id,
                            patient_id=f'p{i}', source=source, split='val', jaw='upper',
                            npz_path=npz.name, reference_kind='manual',
                            candidate_provenance={'kind': 'geometry_only'}))
        directory = run / f'{case_id}_upper'
        directory.mkdir(parents=True)
        (directory / 'candidates.json').write_text(json.dumps([
            {'transform': candidates[0].tolist(), 'mean_tre_mm': 100.},
            {'transform': candidates[1].tolist(), 'mean_tre_mm': 0.}]))
        (directory / 'result.json').write_text(json.dumps({
            'registration': {'transform': candidates[1].tolist()},
            'metrics': {'mean_tre_mm': 0.}}))
    manifest = tmp_path / 'journal.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'records': records}))
    return manifest, run


def test_real_legacy_selected_transform_produces_regret_without_gt_selection(tmp_path):
    manifest, run = fixture(tmp_path)
    report = diagnose_candidates(manifest, run, tmp_path / 'report', bootstrap_samples=10)
    case = report['cases'][0]
    assert case['final_selected_d_mm'] == pytest.approx(4.)
    assert case['final_oracle_d_mm'] == pytest.approx(0.)
    assert case['final_regret_mm'] == pytest.approx(4.)
    assert case['method'] == 'RTR-General-cache'
    saved = json.loads((tmp_path / 'report/evaluation_input.json').read_text())
    assert saved['records'][0]['selected_index'] == 1
    assert saved['records'][0]['initial_candidates'] == saved['records'][0]['final_candidates']
    assert saved['records'][0]['cache_provenance']['candidate_pool_verified_against_prepared']


@pytest.mark.parametrize('failure', ['missing_result', 'selection_not_in_pool'])
def test_missing_or_unmatched_selection_is_failure_with_reference_retained(tmp_path, failure):
    manifest, run = fixture(tmp_path)
    result = run / 'legacy0_upper/result.json'
    if failure == 'missing_result':
        result.unlink()
    else:
        transform = np.eye(4)
        transform[0, 3] = 12.
        result.write_text(json.dumps({'registration': {'transform': transform.tolist()}}))
    report = diagnose_candidates(manifest, run, tmp_path / 'report', bootstrap_samples=10)
    assert report['methods']['RTR-General-cache']['n_invalid_cases'] == 1
    record = json.loads((tmp_path / 'report/evaluation_input.json').read_text())['records'][0]
    assert record['failed']
    assert record['anchors'] and record['reference_transform']


def test_multi_source_requires_explicit_source(tmp_path):
    manifest, run = fixture(tmp_path, sources=('a', 'b'))
    with pytest.raises(ValueError, match='source'):
        diagnose_candidates(manifest, run, tmp_path / 'report', bootstrap_samples=10)
    report = diagnose_candidates(manifest, run, tmp_path / 'report', source='b', bootstrap_samples=10)
    assert len(report['cases']) == 1
    assert report['cases'][0]['source'] == 'b'


def test_different_run_pool_cannot_reuse_prepared_provenance(tmp_path):
    manifest, run = fixture(tmp_path)
    path = run / 'legacy0_upper/candidates.json'
    rows = json.loads(path.read_text())
    rows[0]['transform'][0][3] = 13.
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='prepared'):
        diagnose_candidates(manifest, run, tmp_path / 'report', bootstrap_samples=10)


def test_cli_writes_diagnostic_report_without_field_checkpoint(tmp_path):
    manifest, run = fixture(tmp_path)
    script = Path(__file__).resolve().parents[1] / 'scripts/diagnose_journal_candidates.py'
    output = tmp_path / 'cli_report'
    result = subprocess.run([sys.executable, str(script), '--manifest', str(manifest),
                             '--candidate-run', str(run), '--split', 'val',
                             '--output-dir', str(output), '--bootstrap-samples', '10'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((output / 'report.json').read_text())['cases'][0]['final_regret_mm'] == 4.
