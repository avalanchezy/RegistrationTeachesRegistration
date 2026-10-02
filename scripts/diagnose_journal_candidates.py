#!/usr/bin/env python3
"""Measure cached RTR-General selection regret before training a field model.

Reads <original_case_id>_<jaw>/candidates.json and result.json from an existing
run_geometry_benchmark run. The selected transform comes only from
result['registration']['transform']; stored GT diagnostic scores are ignored.
Prepared anchors/reference define D. The run's candidate transforms must occur
in the prepared manifest's candidate pool so its declared provenance applies.
The prepared pool may concatenate several runs; this report evaluates only the
specified run. No field checkpoint or training is required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.journal.data import load_case, load_journal_manifest
from task2reg.journal.evaluation import evaluate_records
from task2reg.journal.verification import _transforms


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def diagnose_candidates(manifest, candidate_run, output_dir, *, split='val', source=None,
                        bootstrap_samples=2000, seed=0, data_root=None):
    records = load_journal_manifest(Path(manifest), data_root=data_root)
    records = [row for row in records if row['split'] == split]
    sources = {row['source'] for row in records}
    if source is None and len(sources) > 1:
        raise ValueError('Multiple sources in selected split: specify --source for this single-source cached run')
    if source is not None:
        records = [row for row in records if row['source'] == source]
    if not records:
        raise ValueError('No records match the requested split/source')
    if any(row['reference_kind'] not in {'manual', 'silver'} for row in records):
        raise ValueError('Candidate diagnosis requires manual/silver references, never unlabeled or pseudo records')
    run = Path(candidate_run).expanduser().resolve()
    evaluations = []
    for record in records:
        case = load_case(record)
        if 'candidates' not in case or not len(case['candidates']):
            raise ValueError('The prepared manifest must include imported candidates and their explicit provenance')
        identity = record.get('original_case_id', record['case_id'])
        directory = run / f"{identity}_{record['jaw']}"
        candidates_path, result_path = directory / 'candidates.json', directory / 'result.json'
        row = {key: record[key] for key in ('source', 'patient_id', 'case_id', 'jaw', 'reference_kind')}
        row.update(method='RTR-General-cache', anchors=case['anchors'].tolist(),
                   reference_transform=case['transform'].tolist(),
                   cache_provenance=dict(candidate_run=str(run), original_case_id=identity,
                                         candidates_sha256=_digest(candidates_path), result_sha256=_digest(result_path),
                                         declared_candidate_provenance=record['candidate_provenance'],
                                         candidate_pool_verified_against_prepared=False))
        try:
            payload = json.loads(candidates_path.read_text(encoding='utf-8'))
            if not isinstance(payload, list) or not payload:
                raise ValueError('Cached candidates.json must contain a nonempty list')
            candidates = _transforms([entry['transform'] for entry in payload], 'cached candidates')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            row.update(failed=True, failure_reason=f'cached_candidates_failure: {exc}')
            evaluations.append(row)
            continue
        # A provenance declaration for an unrelated candidate run is not valid.
        prepared = case['candidates']
        for candidate in candidates:
            if not np.any(np.all(np.isclose(prepared, candidate, rtol=1e-6, atol=1e-5), axis=(1, 2))):
                raise ValueError(f'Cached candidate is absent from prepared candidate pool: {directory}')
        row['cache_provenance']['candidate_pool_verified_against_prepared'] = True
        row.update(initial_candidates=candidates.tolist(), final_candidates=candidates.tolist())
        try:
            result = json.loads(result_path.read_text(encoding='utf-8'))
            selected = _transforms([result['registration']['transform']], 'cached selected transform')[0]
            matches = np.flatnonzero(np.all(np.isclose(candidates, selected, rtol=1e-6, atol=1e-5), axis=(1, 2)))
            if not len(matches):
                raise ValueError('Selected registration transform is absent from cached candidate pool')
            row.update(selected_index=int(matches[0]), initial_selected_index=int(matches[0]), failed=False)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            row.update(failed=True, failure_reason=f'cached_selection_failure: {exc}')
        evaluations.append(row)
    provenance = dict(manifest=str(Path(manifest).resolve()), candidate_run=str(run),
                      source=records[0]['source'], split=split,
                      candidate_pool_definition='Specified run, checked as a subset of prepared candidate transforms',
                      selection_definition='Cached result.registration.transform matched to candidates; no GT selection')
    report = evaluate_records(evaluations, bootstrap_samples=bootstrap_samples, seed=seed)
    report['diagnostic_provenance'] = provenance
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / 'evaluation_input.json', dict(schema_version=1, records=evaluations,
                                                     diagnostic_provenance=provenance))
    _write_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--candidate-run', type=Path, required=True)
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test', 'external_test'])
    parser.add_argument('--source', help='Required if the selected split contains multiple sources')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args()
    report = diagnose_candidates(args.manifest, args.candidate_run, args.output_dir, split=args.split,
                                 source=args.source, bootstrap_samples=args.bootstrap_samples,
                                 seed=args.seed, data_root=args.data_root)
    print(f"Diagnosed {len(report['cases'])} cached cases; wrote {args.output_dir / 'report.json'}")


if __name__ == '__main__':
    main()
