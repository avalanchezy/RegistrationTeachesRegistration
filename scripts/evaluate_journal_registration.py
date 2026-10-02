#!/usr/bin/env python3
"""Evaluate {schema_version: 1, records: [...]} using fixed IOS anchors.

Record keys: patient_id, case_id, jaw, method, anchors[N,3],
reference_transform[4,4], initial_candidates[K,4,4], final_candidates[K,4,4],
selected_index; optional reference_kind (manual/silver), source namespace,
initial_selected_index, confidence (larger is better),
failed and failure_reason. Candidate arrays must represent matched starts.
Failed records may omit transform evidence; their identity fields are required.
Methods must cover identical case sets. --expected-manifest fixes the complete
evaluation cohort (default split: test); --expected-method can be repeated to
declare all methods, including those with no outputs. Missing outputs raise
unless --missing-as-failure is explicit; they then count as failures, not D.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.journal.evaluation import evaluate_records
from task2reg.journal.data import load_journal_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reference-kind', choices=['manual', 'silver'],
                        help='Evaluate only this reference type; mixed types cannot be pooled')
    parser.add_argument('--source', help='Evaluate only this source namespace')
    parser.add_argument('--expected-manifest', type=Path,
                        help='Journal manifest defining every expected case, without loading NPZ data')
    parser.add_argument('--split', choices=['val', 'test', 'external_test'],
                        help='Manifest evaluation split (default: test); requires --expected-manifest')
    parser.add_argument('--expected-method', action='append', dest='expected_methods',
                        help='Expected method name; repeat for every method in the comparison')
    parser.add_argument('--missing-as-failure', action='store_true',
                        help='Count omitted outputs as failures; requires --expected-manifest')
    parser.add_argument('--success-threshold-mm', type=float, default=2.)
    args = parser.parse_args()
    if args.split is not None and args.expected_manifest is None:
        parser.error('--split requires --expected-manifest')
    if args.missing_as_failure and args.expected_manifest is None:
        parser.error('--missing-as-failure requires --expected-manifest')
    payload = json.loads(args.input.read_text())
    if not isinstance(payload, dict) or payload.get('schema_version') != 1 or not isinstance(payload.get('records'), list):
        parser.error('input must contain schema_version: 1 and a records list')
    records = payload['records']
    if args.reference_kind is not None:
        records = [row for row in records if row.get('reference_kind', 'unspecified') == args.reference_kind]
    if args.source is not None:
        records = [row for row in records if row.get('source', '') == args.source]
    expected = None
    if args.expected_manifest is not None:
        try:
            manifest = load_journal_manifest(args.expected_manifest)
        except ValueError as exc:
            parser.error(str(exc))
        selected_split = args.split or 'test'
        expected = [row for row in manifest if row['split'] == selected_split
                    and (args.reference_kind is None or row['reference_kind'] == args.reference_kind)
                    and (args.source is None or row['source'] == args.source)]
        if not expected:
            parser.error('No expected cohort records match the requested split/reference-kind/source filters')
        manifest_splits = {(row['source'], row['case_id'], row['jaw']): row['split'] for row in manifest}
        # Prediction exports need not repeat split metadata. Keep unknown IDs
        # so strict cohort validation rejects them instead of filtering them out.
        records = [row for row in records if manifest_splits.get(
            (row.get('source', ''), row.get('case_id'), row.get('jaw')), selected_split) == selected_split]
    if not records and not (args.missing_as_failure and args.expected_methods):
        parser.error('No records match the requested reference-kind/source filters')
    try:
        report = evaluate_records(records, bootstrap_samples=args.bootstrap_samples,
                                  seed=args.seed, success_threshold_mm=args.success_threshold_mm,
                                  expected_cohort=expected, expected_methods=args.expected_methods,
                                  missing_as_failure=args.missing_as_failure)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"Evaluated {len(report['cases'])} case/method records; wrote {args.output}")


if __name__ == '__main__':
    main()
