"""The control changes positive spatial weights and no accepted geometry."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from scripts.ablate_field_pseudo_weights import ablate_field_pseudo_weights
from task2reg.journal.data import load_case, load_journal_manifest
from task2reg.journal.runtime import sha256_file


def pseudo_manifest(tmp_path, *, pseudo_kind='field'):
    points = np.array([[0., 0., 0.], [1., 2., 0.], [1., 0., 3.], [2., 1., 3.]], np.float32)
    transform = np.eye(4)
    transform[:3, 3] = [3., 1., -2.]
    arrays = dict(image=np.zeros((4, 4, 4), np.int16), affine=np.eye(4), points=points,
                  anchors=points[:3], candidates=np.eye(4)[None], transform=transform)
    records = []
    for split in ['val', 'pseudo']:
        name = f'{split}.npz'
        payload = dict(arrays)
        if split == 'pseudo':
            if pseudo_kind == 'support':
                payload.pop('transform')
                payload['label'] = np.zeros((4, 4, 4), np.uint8)
            else:
                payload['point_weights'] = np.array([0., .05, .5, 1.], np.float32)
                payload['extra_measured_array'] = np.arange(5, dtype=np.int16)
        np.savez_compressed(tmp_path / name, **payload)
        row = dict(source='hospital', case_id=split, patient_id=f'p-{split}', jaw='upper', split=split,
                   npz_path=name, reference_kind='manual' if split == 'val' else 'pseudo',
                   candidate_provenance={'kind': 'geometry_only'})
        if split == 'pseudo':
            row.update(pseudo_kind=pseudo_kind, supervision_radius_mm=1.25,
                       pseudo_provenance=dict(teacher_id='checkpoint-sha', excluded_patient_ids=['p-val']),
                       verification_metrics={'u95_mm': .4})
        records.append(row)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1, records=records,
                                       export_settings={'verifier': 'frozen', 'checkpoint_sha256': 'checkpoint-sha'})))
    return manifest


def test_uniform_control_preserves_zero_mask_all_geometry_and_parent_provenance(tmp_path):
    manifest = pseudo_manifest(tmp_path)
    before_manifest = manifest.read_bytes()
    before_hashes = {r['npz_path']: sha256_file(r['npz_path']) for r in load_journal_manifest(manifest)}
    output = ablate_field_pseudo_weights(manifest, tmp_path / 'uniform')
    records = load_journal_manifest(output)
    parent = load_journal_manifest(manifest)
    assert len(records) == len(parent) == 2
    for old, new in zip(parent, records):
        assert {k: v for k, v in new.items() if k != 'npz_path'} == {k: v for k, v in old.items() if k != 'npz_path'}
        with np.load(old['npz_path'], allow_pickle=False) as a, np.load(new['npz_path'], allow_pickle=False) as b:
            assert a.files == b.files
            for name in a.files:
                if old['split'] == 'pseudo' and name == 'point_weights':
                    np.testing.assert_array_equal(b[name], [0., 1., 1., 1.])
                    assert b[name].dtype == a[name].dtype
                else:
                    np.testing.assert_array_equal(b[name], a[name])
        if old['split'] != 'pseudo':
            assert sha256_file(new['npz_path']) == sha256_file(old['npz_path'])
    metadata = json.loads(output.read_text())
    assert metadata['export_settings']['checkpoint_sha256'] == 'checkpoint-sha'
    assert metadata['weight_ablation']['mode'] == 'uniform_positive_mask'
    assert metadata['weight_ablation']['parent_manifest_sha256'] == sha256_file(manifest)
    assert metadata['weight_ablation']['teacher_ids'] == ['checkpoint-sha']
    assert metadata['weight_ablation']['records'][1]['parent_payload_sha256']
    assert manifest.read_bytes() == before_manifest
    assert all(sha256_file(path) == digest for path, digest in before_hashes.items())


def test_all_payload_paths_remain_valid_when_only_new_output_directory_is_moved(tmp_path):
    parent = tmp_path / 'parent'
    parent.mkdir()
    manifest = pseudo_manifest(parent)
    output = ablate_field_pseudo_weights(manifest, tmp_path / 'uniform')
    payload = json.loads(output.read_text())
    assert all(not Path(r['npz_path']).is_absolute() and '..' not in Path(r['npz_path']).parts
               for r in payload['records'])
    moved = tmp_path / 'moved'
    shutil.move(str(output.parent), moved)
    shutil.rmtree(parent)
    records = load_journal_manifest(moved / 'manifest.json')
    for record in records:
        load_case(record)
    assert load_case(records[1])['point_weights'].tolist() == [0., 1., 1., 1.]


@pytest.mark.parametrize('kind', ['support_only', 'no_pseudo', 'empty'])
def test_rejects_manifest_without_field_pseudo_supervision(tmp_path, kind):
    manifest = pseudo_manifest(tmp_path, pseudo_kind='support' if kind == 'support_only' else 'field')
    payload = json.loads(manifest.read_text())
    if kind == 'no_pseudo':
        payload['records'] = payload['records'][:1]
    elif kind == 'empty':
        payload['records'] = []
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='field|nonempty'):
        ablate_field_pseudo_weights(manifest, tmp_path / 'uniform')


def test_nonempty_output_is_rejected_without_overwriting_existing_files(tmp_path):
    manifest = pseudo_manifest(tmp_path)
    output = tmp_path / 'uniform'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('keep')
    with pytest.raises(ValueError, match='empty'):
        ablate_field_pseudo_weights(manifest, output)
    assert list(output.iterdir()) == [sentinel] and sentinel.read_text() == 'keep'


def test_cli_creates_the_uniform_control_without_loading_a_teacher(tmp_path):
    manifest = pseudo_manifest(tmp_path)
    output = tmp_path / 'uniform'
    script = Path(__file__).resolve().parents[1] / 'scripts/ablate_field_pseudo_weights.py'
    result = subprocess.run([sys.executable, str(script), '--manifest', str(manifest),
                             '--output-dir', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert load_case(load_journal_manifest(output / 'manifest.json')[1])['point_weights'].tolist() == [0., 1., 1., 1.]
