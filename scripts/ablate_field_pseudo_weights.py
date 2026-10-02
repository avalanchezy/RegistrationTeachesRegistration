"""Copy verified field pseudo labels with uniform weights on the same valid mask.

This control keeps accepted transforms and every non-weight array fixed. It does
not rerun verification or claim the teacher's accepted poses are correct. All
payloads are copied so the output directory can move independently of its parent.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2reg.journal.data import load_case, load_journal_manifest, validate_protocol
from task2reg.journal.inference import record_directory
from task2reg.journal.runtime import manifest_fingerprint, sha256_file, write_json


def ablate_field_pseudo_weights(manifest, output_dir, *, data_root=None):
    """Create a portable uniform-positive control without modifying the parent.

    Field pseudo records retain zero weights exactly and map positive weights to
    one in their original dtype. Other records, including any support pseudo
    records in a mixed manifest, are copied byte for byte. A field pseudo cohort
    must be present; support-only and empty cohorts cannot define this control.
    """
    manifest = Path(manifest).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError('output_dir must be a new or empty directory')
    source_payload = json.loads(manifest.read_text(encoding='utf-8'))
    records = load_journal_manifest(manifest, data_root=data_root)

    def is_field(row):
        return row['split'] == 'pseudo' and row.get('pseudo_kind', 'field') == 'field'

    targets = [row for row in records if is_field(row)]
    if not targets:
        raise ValueError('The manifest must contain nonempty field pseudo supervision; support-only is not supported')
    parent_hash = sha256_file(manifest)
    parent_fingerprint = manifest_fingerprint(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = output_dir / 'cases'
    payload_dir.mkdir()
    output_records, provenance = [], []
    for record in records:
        # Validate using the existing physical/provenance contract. Read original
        # arrays separately because load_case adds a derived supervision radius.
        load_case(record)
        source_path = Path(record['npz_path'])
        relative = Path('cases') / f'{record_directory(record)}.npz'
        target_path = output_dir / relative
        transformed = is_field(record)
        details = dict(source=record['source'], case_id=record['case_id'], jaw=record['jaw'],
                       patient_id=record['patient_id'], parent_npz_path=str(source_path),
                       parent_payload_sha256=sha256_file(source_path), npz_path=relative.as_posix(),
                       converted_to_uniform_positive=transformed)
        if transformed:
            with np.load(source_path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            weights = arrays['point_weights']
            positive = weights > 0
            uniform = weights.copy()
            uniform[positive] = 1
            arrays['point_weights'] = uniform
            np.savez_compressed(target_path, **arrays)
            details.update(n_positive_weights=int(positive.sum()), n_zero_weights=int((weights == 0).sum()),
                           pseudo_provenance=record['pseudo_provenance'],
                           supervision_radius_mm=record.get('supervision_radius_mm', 2.))
        else:
            shutil.copy2(source_path, target_path)
        details['output_payload_sha256'] = sha256_file(target_path)
        provenance.append(details)
        output_records.append(record | {'npz_path': relative.as_posix()})
    validate_protocol(output_records)
    output_payload = source_payload | dict(records=output_records)
    output_payload['weight_ablation'] = dict(
        mode='uniform_positive_mask', schema_version=1,
        interpretation='Same accepted geometry and zero-weight mask; only positive spatial weights become one.',
        parent_manifest_path=str(manifest), parent_manifest_sha256=parent_hash,
        parent_manifest_fingerprint=parent_fingerprint,
        parent_manifest_metadata={key: value for key, value in source_payload.items() if key != 'records'},
        teacher_ids=sorted({row['pseudo_provenance']['teacher_id'] for row in targets}),
        n_records=len(records), n_field_pseudo_records=len(targets), records=provenance,
    )
    output_manifest = output_dir / 'manifest.json'
    write_json(output_manifest, output_payload)
    return output_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args()
    manifest = ablate_field_pseudo_weights(args.manifest, args.output_dir, data_root=args.data_root)
    print(f'Uniform-positive pseudo-weight control written to {manifest}')


if __name__ == '__main__':
    main()
