#!/usr/bin/env python3
"""Evaluate {schema_version: 1, records: [...]} using fixed IOS anchors.

Record keys: patient_id, case_id, jaw, method, anchors[N,3],
reference_transform[4,4], initial_candidates[K,4,4], final_candidates[K,4,4],
selected_index; optional reference_kind (manual/silver), source namespace,
initial_selected_index, confidence (larger is better),
failed and failure_reason. Candidate arrays must represent matched starts.
Failed records may omit transform evidence; their identity fields are required.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.journal.evaluation import evaluate_records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reference-kind', choices=['manual', 'silver'],
                        help='Evaluate only this reference type; mixed types cannot be pooled')
    parser.add_argument('--source', help='Evaluate only this source namespace')
    parser.add_argument('--success-threshold-mm', type=float, default=2.)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    if not isinstance(payload, dict) or payload.get('schema_version') != 1 or not isinstance(payload.get('records'), list):
        parser.error('input must contain schema_version: 1 and a records list')
    records = payload['records']
    if args.reference_kind is not None:
        records = [row for row in records if row.get('reference_kind', 'unspecified') == args.reference_kind]
    if args.source is not None:
        records = [row for row in records if row.get('source', '') == args.source]
    if not records:
        parser.error('No records match the requested reference-kind/source filters')
    report = evaluate_records(records, bootstrap_samples=args.bootstrap_samples,
                              seed=args.seed, success_threshold_mm=args.success_threshold_mm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"Evaluated {len(report['cases'])} case/method records; wrote {args.output}")


if __name__ == '__main__':
    main()
