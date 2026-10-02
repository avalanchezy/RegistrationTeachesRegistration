"""Fixed-anchor registration diagnostics with patient-level inference.

Distance D uses fixed IOS anchors under predicted and reference transforms. It is
not independent anatomical landmark error. Error statistics explicitly condition
on valid predictions; failures are retained in failure rates and risk coverage.
"""
from itertools import combinations

import numpy as np

from .verification import _points, _transforms, pose_displacement


_ERROR_METRICS = ('initial_selected_d_mm', 'final_selected_d_mm', 'initial_oracle_d_mm',
                  'final_oracle_d_mm', 'initial_regret_mm', 'final_regret_mm', 'refinement_delta_mm',
                  'initial_rotation_deg', 'final_rotation_deg')
_RATE_METRICS = ('failure_rate', 'registration_failure_rate', 'invalid_output_rate', 'success_rate',
                 'failure_rate_1mm', 'failure_rate_2mm', 'failure_rate_3mm')


def _case_metrics(record, success_threshold_mm):
    row = {key: str(record[key]) for key in ('patient_id', 'case_id', 'jaw', 'method')}
    row['source'] = record.get('source', '')
    row['reference_kind'] = record.get('reference_kind', 'unspecified')
    row.update({key: None for key in _ERROR_METRICS})
    row.update(failed=False, failure_reason=None, success=False, confidence=None,
               registration_failed=True, parity_mismatch=None,
               initial_parity_mismatch=None, final_parity_mismatch=None)
    confidence = record.get('confidence')
    if confidence is not None:
        try:
            if np.isfinite(float(confidence)):
                row['confidence'] = float(confidence)
        except (ValueError, TypeError):
            pass
    if record.get('failed', False):
        row.update(failed=True, failure_reason=str(record.get('failure_reason', 'reported failure')))
        return row
    try:
        anchors = _points(record.get('anchors'), 'anchors')
        reference = _transforms(np.asarray(record.get('reference_transform'))[None], 'reference_transform')[0]
        initial = _transforms(record.get('initial_candidates'), 'initial_candidates')
        final = _transforms(record.get('final_candidates'), 'final_candidates')
        if len(initial) != len(final):
            raise ValueError('initial/final candidates must be matched starts')
        selected = record.get('selected_index')
        initial_selected = record.get('initial_selected_index', selected)
        for name, index in [('selected_index', selected), ('initial_selected_index', initial_selected)]:
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(initial):
                raise ValueError(f'{name} must index the matched candidate list')
        d_initial = [pose_displacement(t, reference, anchors) for t in initial]
        d_final = [pose_displacement(t, reference, anchors) for t in final]
        reference_parity = np.sign(np.linalg.det(reference[:3, :3]))
        for stage, candidate in [('initial', initial[initial_selected]), ('final', final[selected])]:
            mismatch = bool(np.sign(np.linalg.det(candidate[:3, :3])) != reference_parity)
            row[f'{stage}_parity_mismatch'] = mismatch
            if not mismatch:
                relative_rotation = candidate[:3, :3] @ reference[:3, :3].T
                row[f'{stage}_rotation_deg'] = float(np.degrees(np.arccos(
                    np.clip((np.trace(relative_rotation) - 1) / 2, -1., 1.))))
        row['parity_mismatch'] = row['final_parity_mismatch']
        row['registration_failed'] = bool(row['parity_mismatch'] or d_final[selected] > success_threshold_mm)
        row.update(initial_selected_d_mm=d_initial[initial_selected], final_selected_d_mm=d_final[selected],
                   initial_oracle_d_mm=min(d_initial), final_oracle_d_mm=min(d_final),
                   initial_regret_mm=d_initial[initial_selected] - min(d_initial),
                   final_regret_mm=d_final[selected] - min(d_final),
                   refinement_delta_mm=d_final[selected] - d_initial[selected],
                   success=not row['registration_failed'])
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        row.update(failed=True, failure_reason=f'invalid_evidence: {exc}')
    return row


def _bootstrap(values, samples, rng):
    """Values contain exactly one observation per patient, never one per jaw."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(mean=None, ci95=[None, None], n_patients=0)
    means = np.empty(samples)
    # Chunking bounds memory even for a large cohort/bootstrap count.
    for start in range(0, samples, 256):
        size = min(256, samples - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[start:start + size] = values[indices].mean(axis=1)
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist(), n_patients=len(values))


def _patient_rows(rows):
    result = []
    for patient in sorted({row['patient_id'] for row in rows}):
        cases = [row for row in rows if row['patient_id'] == patient]
        item = dict(patient_id=patient, n_cases=len(cases), n_failed_cases=sum(row['registration_failed'] for row in cases),
                    n_invalid_cases=sum(row['failed'] for row in cases),
                    invalid_output_rate=float(np.mean([row['failed'] for row in cases])),
                    registration_failure_rate=float(np.mean([row['registration_failed'] for row in cases])),
                    failure_rate=float(np.mean([row['registration_failed'] for row in cases])),
                    success_rate=float(np.mean([row['success'] for row in cases])))
        for threshold in (1, 2, 3):
            item[f'failure_rate_{threshold}mm'] = float(np.mean([
                row['failed'] or row['parity_mismatch'] or row['final_selected_d_mm'] > threshold
                for row in cases]))
        for metric in _ERROR_METRICS:
            values = [row[metric] for row in cases if row[metric] is not None]
            item[metric] = float(np.mean(values)) if values else None
        result.append(item)
    return result


def _risk_coverage(rows):
    # Missing confidence is retained at the end, never silently discarded.
    ranked = sorted(rows, key=lambda row: (-(row['confidence'] if row['confidence'] is not None else -np.inf),
                                            row['patient_id'], row['source'], row['case_id'], row['jaw']))
    curve = []
    for count in range(1, len(ranked) + 1):
        selected = ranked[:count]
        patients = _patient_rows(selected)
        valid_errors = [row['final_selected_d_mm'] for row in patients if row['final_selected_d_mm'] is not None]
        curve.append(dict(coverage=count / len(ranked), n_cases=count, n_patients=len(patients),
                          n_failed_cases=sum(row['registration_failed'] for row in selected),
                          n_invalid_cases=sum(row['failed'] for row in selected),
                          confidence_threshold=selected[-1]['confidence'],
                          risk=float(np.mean([1 - row['success_rate'] for row in patients])),
                          mean_d_mm=float(np.mean(valid_errors)) if valid_errors else None))
    return curve


def evaluate_records(records, *, bootstrap_samples=2000, seed=0, success_threshold_mm=2.):
    """Evaluate JSON-compatible observations; schema is documented by the CLI.

    Each method/source/case/jaw has one record and a globally reconciled patient_id.
    Patient statistics average jaws/cases first and bootstrap patients with
    replacement. Paired comparisons use the common patient cohort; D comparisons
    additionally require valid predictions for both methods on matched cases.
    Failure rates and threshold risk include failed or invalid predictions.
    Each method must use one reference_kind; mixed manual/silver/unspecified
    evidence must be filtered into separate reports before evaluation.
    """
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError('bootstrap_samples must be a positive integer')
    if not np.isfinite(success_threshold_mm) or success_threshold_mm < 0:
        raise ValueError('success_threshold_mm must be finite and nonnegative')
    records = list(records)
    if not records:
        raise ValueError('records must be nonempty')
    seen = set()
    identity_patients = {}
    reference_evidence = {}
    method_reference_kinds = {}
    identity_reference_kinds = {}
    for record in records:
        for key in ('patient_id', 'case_id', 'jaw', 'method'):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f'{key} must be an explicit nonempty string')
        source = record.get('source', '')
        if not isinstance(source, str):
            raise ValueError('source must be a string when present')
        identity = (source, record['case_id'])
        if identity in identity_patients and identity_patients[identity] != record['patient_id']:
            raise ValueError('case has inconsistent patient identity across jaws/methods')
        identity_patients[identity] = record['patient_id']
        key = (record['method'], source, record['case_id'], record['jaw'])
        if key in seen:
            raise ValueError(f'duplicate method/source/case/jaw record: {key}')
        seen.add(key)
        # Compare physical evidence before aggregation: identity alone cannot
        # make two methods paired when their anchors or reference differ.
        evidence_key = (source, record['case_id'], record['jaw'])
        reference_kind = record.get('reference_kind', 'unspecified')
        if reference_kind not in {'manual', 'silver', 'unspecified'}:
            raise ValueError('reference_kind must be manual, silver, or unspecified')
        method = record['method']
        if method in method_reference_kinds and method_reference_kinds[method] != reference_kind:
            raise ValueError(f'Mixed reference_kind values within method {method}; filter before evaluation')
        method_reference_kinds[method] = reference_kind
        if evidence_key in identity_reference_kinds and identity_reference_kinds[evidence_key] != reference_kind:
            raise ValueError(f'Inconsistent reference_kind across methods for {evidence_key}')
        identity_reference_kinds[evidence_key] = reference_kind
        previous = reference_evidence.setdefault(evidence_key, {})
        for name in ('anchors', 'reference_transform'):
            value = record.get(name)
            if value is None:
                # Failure records may omit geometric evidence. Their failures
                # remain paired while their unavailable D cannot enter a pair.
                continue
            if name in previous:
                try:
                    current = np.asarray(value, dtype=float)
                    prior = np.asarray(previous[name], dtype=float)
                    matches = (current.shape == prior.shape and
                               np.allclose(current, prior, rtol=1e-7, atol=1e-6))
                except (ValueError, TypeError):
                    matches = False
                if not matches:
                    raise ValueError(f'inconsistent reference evidence {name} for {evidence_key}')
            else:
                previous[name] = value
    cases = [_case_metrics(record, success_threshold_mm) for record in records]
    cases.sort(key=lambda row: (row['method'], row['patient_id'], row['source'], row['case_id'], row['jaw']))
    methods = sorted({row['method'] for row in cases})
    rng = np.random.default_rng(seed)
    report = dict(schema_version=1, metric='D: mean fixed-anchor displacement in millimetres',
                  bootstrap=dict(unit='patient', samples=bootstrap_samples, seed=seed, confidence=.95),
                  success_threshold_mm=success_threshold_mm,
                  error_statistics='D conditional on valid rigid outputs; rotation conditional on matching parity.',
                  failure_rate_definition='Registration failure: invalid output, parity mismatch, or final D above threshold. failure_rate aliases registration_failure_rate.',
                  risk_definition='Patient mean of jaw/case failure or final D above threshold; coverage is case fraction.',
                  cases=cases, patients={}, methods={}, paired_comparisons=[], risk_coverage={})
    for method in methods:
        rows = [row for row in cases if row['method'] == method]
        patients = _patient_rows(rows)
        report['patients'][method] = patients
        summary = dict(reference_kind=method_reference_kinds[method],
                       n_cases=len(rows), n_patients=len(patients),
                       n_failed_cases=sum(row['registration_failed'] for row in rows),
                       n_invalid_cases=sum(row['failed'] for row in rows),
                       n_missing_confidence=sum(row['confidence'] is None for row in rows))
        for metric in (*_ERROR_METRICS, *_RATE_METRICS):
            summary[metric] = _bootstrap([row[metric] for row in patients if row[metric] is not None], bootstrap_samples, rng)
        report['methods'][method] = summary
        report['risk_coverage'][method] = _risk_coverage(rows)
    for a, b in combinations(methods, 2):
        lookup_a = {(row['patient_id'], row['source'], row['case_id'], row['jaw']): row for row in cases if row['method'] == a}
        lookup_b = {(row['patient_id'], row['source'], row['case_id'], row['jaw']): row for row in cases if row['method'] == b}
        common = sorted(lookup_a.keys() & lookup_b.keys())
        patients = sorted({key[0] for key in common})
        comparison = dict(method_a=a, method_b=b, n_common_patients=len(patients), n_common_cases=len(common))
        for metric in ('final_selected_d_mm', 'initial_selected_d_mm', 'final_regret_mm', 'registration_failure_rate', 'invalid_output_rate', 'failure_rate'):
            deltas = []
            for patient in patients:
                patient_deltas = []
                for key in common:
                    if key[0] != patient:
                        continue
                    row_metric = ('failed' if metric == 'invalid_output_rate' else
                                  'registration_failed' if metric in {'failure_rate', 'registration_failure_rate'} else metric)
                    value_a = lookup_a[key][row_metric]
                    value_b = lookup_b[key][row_metric]
                    if value_a is not None and value_b is not None:
                        patient_deltas.append(float(value_b) - float(value_a))
                if patient_deltas:
                    deltas.append(float(np.mean(patient_deltas)))
            comparison[f'{metric}_b_minus_a'] = _bootstrap(deltas, bootstrap_samples, rng)
        report['paired_comparisons'].append(comparison)
    return report
