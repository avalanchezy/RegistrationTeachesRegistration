"""Oracle-centered convergence diagnostics, never deployment performance.

The reference defines diagnostic starts and post-hoc fixed-anchor D only. Model
queries and refinement receive no reference transform or anatomical labels.
Physical starts and point subsets depend on identity/seed, never a checkpoint or
manifest ordering. All numerical failures stay in the trial denominator.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from .data import load_case, load_journal_manifest
from .engine import _selection_history, encode_case, tensor
from .evaluation import _bootstrap
from .field import refine_transform, registration_energy, transform_points
from .inference import load_field
from .pose_supervision import gradient_alignment_diagnostic
from .runtime import canonical_hash, load_checkpoint, manifest_fingerprint, runtime_metadata, set_seed, sha256_file, write_json
from .verification import _points, _transforms, pose_displacement


_DIAGNOSTIC = ('Reference-centered starts diagnose field convergence; they are oracle initializations, '
               'not deployment results. D is fixed-anchor displacement, not anatomical landmark TRE.')
_ERROR_METRICS = ('initial_d_mm', 'final_d_mm', 'delta_d_mm', 'initial_energy', 'final_energy', 'delta_energy',
                  'initial_gradient_cosine')
_RATE_FIELDS = dict(failure_rate='failed', success_rate='success', improvement_rate='improved',
                    deterioration_rate='worsened', unchanged_rate='unchanged', initial_success_rate='initial_success')
_CONDITIONAL_RATES = ('capture_rate', 'initial_gradient_descent_rate')


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _identity(record):
    keys = ('patient_id', 'source', 'case_id', 'jaw')
    if any(not isinstance(record.get(key), str) or not record[key].strip() for key in keys):
        raise ValueError('patient_id, source, case_id and jaw must be explicit nonempty strings')
    return {key: record[key] for key in keys}


def _seed(record, seed, purpose, repeat=0):
    digest = canonical_hash(dict(version=1, identity=_identity(record), seed=_integer(seed, 'seed'),
                                 purpose=purpose, repeat=repeat))
    return int(digest[:16], 16)


def _grid(values, name, maximum=None):
    values = [float(value) for value in values]
    if not values or any(not np.isfinite(value) or value < 0 or
                         (maximum is not None and value > maximum) for value in values):
        raise ValueError(f'{name} must be nonempty finite nonnegative magnitudes' +
                         (f' <= {maximum}' if maximum is not None else ''))
    if len(set(values)) != len(values):
        raise ValueError(f'{name} must not contain duplicate magnitudes')
    return sorted(values)


def generate_perturbations(record, reference_transform, anchors, *,
                           rotation_degrees=(0., 5., 10., 20., 40.),
                           translation_mm=(0., 1., 2., 4., 8.), repeats=5, seed=0):
    """Proper left increments with exact angle and centroid displacement.

    Rotation axes and translation directions are independent unit normals for
    each identity/repeat, shared across all magnitudes. Translation is a direct
    physical vector after rotation, not the translation coordinate of a twist
    exponential (whose norm changes with rotation). Either reference parity is
    retained. ``anchors`` may be replaced with IOS points by explicit callers.
    """
    reference = _transforms(np.asarray(reference_transform)[None], 'reference_transform')[0]
    anchors = _points(anchors, 'anchors')
    rotations = _grid(rotation_degrees, 'rotation_degrees', 180.)
    translations = _grid(translation_mm, 'translation_mm')
    repeats = _integer(repeats, 'repeats', 1)
    center = (anchors @ reference[:3, :3].T + reference[:3, 3]).mean(0)
    result = []
    for rotation_deg in rotations:
        for translation in translations:
            for repeat in range(repeats):
                derived_seed = _seed(record, seed, 'perturbation', repeat)
                rng = np.random.Generator(np.random.PCG64(derived_seed))
                axis, direction = rng.normal(size=(2, 3))
                axis /= np.linalg.norm(axis)
                direction /= np.linalg.norm(direction)
                x, y, z = axis
                skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                angle = np.radians(rotation_deg)
                rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
                increment = np.eye(4)
                increment[:3, :3] = rotation
                increment[:3, 3] = center - rotation @ center + translation * direction
                initial = increment @ reference
                trial_id = canonical_hash(dict(identity=_identity(record), seed=int(seed),
                                               rotation_deg=rotation_deg, translation_mm=translation,
                                               repeat=repeat))
                result.append(dict(trial_id=trial_id, rotation_deg=rotation_deg,
                                   translation_mm=translation, repeat=repeat,
                                   generator_seed=derived_seed, rotation_axis=axis.tolist(),
                                   translation_direction=direction.tolist(), center_world_mm=center.tolist(),
                                   left_increment=increment.tolist(), initial_transform=initial.tolist()))
    return result


def _point_indices(record, count, budget, seed):
    budget = _integer(budget, 'point_budget', 1)
    rng = np.random.Generator(np.random.PCG64(_seed(record, seed, 'point_subset')))
    return np.sort(rng.choice(count, min(count, budget), replace=False)).tolist()


def _settings(refinement_steps, learning_rate, success_threshold_mm, improvement_tolerance_mm):
    _integer(refinement_steps, 'refinement_steps')
    for name, value, positive in [('learning_rate', learning_rate, True),
                                  ('success_threshold_mm', success_threshold_mm, False),
                                  ('improvement_tolerance_mm', improvement_tolerance_mm, False)]:
        if not np.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')


def benchmark_field(query_fn, *, points, anchors, reference_transform, affine, input_shape,
                    record, method, rotation_degrees=(0., 5., 10., 20., 40.),
                    translation_mm=(0., 1., 2., 4., 8.), repeats=5, seed=0,
                    refinement_steps=20, learning_rate=.25, point_budget=4096,
                    success_threshold_mm=2., improvement_tolerance_mm=1e-4, device='cpu'):
    """Run the actual inference optimizer against an arbitrary differentiable field.

    ``query_fn`` receives only batched transformed physical points. This pure
    runner supports analytic fields as well as both learned field heads. Every
    trial is retained, including invalid energies/gradients and runtime errors.
    """
    _settings(refinement_steps, learning_rate, success_threshold_mm, improvement_tolerance_mm)
    if not isinstance(method, str) or not method.strip():
        raise ValueError('method must be a nonempty string')
    if record.get('reference_kind') not in {'manual', 'silver'}:
        raise ValueError('reference_kind must be manual or silver')
    points, anchors = _points(points, 'points'), _points(anchors, 'anchors')
    reference = _transforms(np.asarray(reference_transform)[None], 'reference_transform')[0]
    starts = generate_perturbations(record, reference, anchors, rotation_degrees=rotation_degrees,
                                   translation_mm=translation_mm, repeats=repeats, seed=seed)
    indices = _point_indices(record, len(points), point_budget, seed)
    sampled_points, affine_tensor = tensor(points[indices], device)[None], tensor(affine, device)[None]
    evidence_hash = canonical_hash(dict(anchors=anchors.tolist(), reference_transform=reference.tolist()))
    rows = []
    for start in starts:
        initial = np.asarray(start['initial_transform'])
        initial_d = pose_displacement(initial, reference, anchors)
        row = _identity(record) | dict(method=method, reference_kind=record['reference_kind']) | start
        row.update(point_indices_sha256=canonical_hash(indices), reference_evidence_sha256=evidence_hash,
                   initial_d_mm=initial_d, final_d_mm=None, delta_d_mm=None,
                   initial_energy=None, final_energy=None, delta_energy=None, final_transform=None,
                   initial_success=initial_d <= success_threshold_mm,
                   failed=False, failure_reason=None, success=False, improved=False,
                   worsened=False, unchanged=False, initial_gradient_cosine=None,
                   initial_gradient_dot=None, initial_gradient_energy_norm=None,
                   initial_gradient_target_norm=None, initial_gradient_descent=None,
                   initial_gradient_failure_reason=None)
        try:
            initial_tensor = tensor(initial, device)[None]
            with torch.no_grad():
                moved = transform_points(sampled_points, initial_tensor)
                energy, _ = registration_energy(query_fn(moved), moved, affine_tensor, input_shape)
                value = float(energy[0])
                if not np.isfinite(value):
                    raise FloatingPointError('nonfinite initial field energy')
                row['initial_energy'] = value
            try:
                direction = gradient_alignment_diagnostic(query_fn, sampled_points, initial_tensor,
                                                           tensor(reference, device)[None], tensor(anchors, device)[None],
                                                           affine_tensor, input_shape)
                row.update({f'initial_gradient_{key}': value for key, value in direction.items()})
            except (FloatingPointError, RuntimeError, ValueError, OverflowError) as exc:
                # Measurement failure must not alter the optimizer's outcome,
                # including an otherwise valid zero-step registration.
                row['initial_gradient_failure_reason'] = f'{type(exc).__name__}: {exc}'
            result = refine_transform(query_fn, sampled_points, initial_tensor,
                                      affine_tensor, input_shape, steps=refinement_steps,
                                      learning_rate=learning_rate)
            final = _transforms(result['transform'].detach().cpu().numpy(), 'final_transform')[0]
            if np.sign(np.linalg.det(final[:3, :3])) != np.sign(np.linalg.det(reference[:3, :3])):
                raise FloatingPointError('refinement changed reference parity')
            initial_energy, final_energy = [float(result[key][0]) for key in ('initial_energy', 'final_energy')]
            if not np.isfinite([initial_energy, final_energy]).all():
                raise FloatingPointError('nonfinite field energy')
            final_d = pose_displacement(final, reference, anchors)
            delta = final_d - initial_d
            row.update(final_transform=final.tolist(), initial_energy=initial_energy, final_energy=final_energy,
                       delta_energy=final_energy - initial_energy, final_d_mm=final_d, delta_d_mm=delta,
                       success=final_d <= success_threshold_mm, improved=delta < -improvement_tolerance_mm,
                       worsened=delta > improvement_tolerance_mm, unchanged=abs(delta) <= improvement_tolerance_mm)
        except (FloatingPointError, RuntimeError, ValueError, OverflowError) as exc:
            row.update(failed=True, failure_reason=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    return rows


def _gradient_counts(rows):
    return dict(n_gradient_comparable_trials=sum(r.get('initial_gradient_descent') is not None for r in rows),
                n_gradient_cosine_trials=sum(r.get('initial_gradient_cosine') is not None for r in rows),
                n_gradient_diagnostic_failures=sum(r.get('initial_gradient_failure_reason') is not None for r in rows))


def _patient_summary(rows):
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[row['patient_id']].append(row)
    patients = []
    for patient, trials in sorted(by_patient.items()):
        by_case = defaultdict(list)
        for row in trials:
            by_case[(row['source'], row['case_id'], row['jaw'])].append(row)
        case_values = []
        for case_trials in by_case.values():
            values = {}
            for metric in _ERROR_METRICS:
                valid = [row[metric] for row in case_trials if row.get(metric) is not None]
                values[metric] = float(np.mean(valid)) if valid else None
            for metric, field in _RATE_FIELDS.items():
                values[metric] = float(np.mean([row[field] for row in case_trials]))
            eligible = [row for row in case_trials if not row['initial_success']]
            values['capture_rate'] = float(np.mean([row['success'] for row in eligible])) if eligible else None
            directions = [row['initial_gradient_descent'] for row in case_trials
                          if row.get('initial_gradient_descent') is not None]
            values['initial_gradient_descent_rate'] = float(np.mean(directions)) if directions else None
            case_values.append(values)
        item = dict(patient_id=patient, n_cases=len(by_case), n_trials=len(trials),
                    n_failed_trials=sum(row['failed'] for row in trials),
                    n_capture_eligible_trials=sum(not row['initial_success'] for row in trials), **_gradient_counts(trials))
        for metric in (*_ERROR_METRICS, *_RATE_FIELDS, *_CONDITIONAL_RATES):
            values = [case[metric] for case in case_values if case[metric] is not None]
            item[metric] = float(np.mean(values)) if values else None
        patients.append(item)
    return patients


def summarize_landscape(trials, *, bootstrap_samples=2000, seed=0):
    """Average repeated starts per case, then cases per patient; bootstrap patients.

    Error/energy means condition on available finite results. Failure/success
    rates include all starts. Capture conditions on initial D above the success
    threshold, retaining failed refinements. Paired D includes only matched
    starts with valid outputs for both methods; paired failure uses all matches.
    """
    samples = _integer(bootstrap_samples, 'bootstrap_samples', 1)
    _integer(seed, 'seed')
    rows = list(trials)
    if not rows:
        raise ValueError('trials must be nonempty')
    if len({row['reference_kind'] for row in rows}) != 1:
        raise ValueError('Mixed reference_kind: manual and silver must have separate reports')
    identity_patients, paired_evidence, seen = {}, {}, set()
    cohorts = defaultdict(set)
    for row in rows:
        _identity(row)
        identity = (row['source'], row['case_id'])
        if identity in identity_patients and identity_patients[identity] != row['patient_id']:
            raise ValueError('inconsistent patient identity across cases/methods')
        identity_patients[identity] = row['patient_id']
        pair_key = (row['source'], row['case_id'], row['jaw'], row['rotation_deg'], row['translation_mm'], row['repeat'])
        cohorts[row['method']].add(pair_key)
        key = (row['method'], *pair_key)
        if key in seen:
            raise ValueError('duplicate method/case/grid/repeat trial')
        seen.add(key)
        previous = paired_evidence.setdefault(pair_key, row)
        for name in ('initial_d_mm', 'initial_transform', 'point_indices_sha256', 'reference_evidence_sha256'):
            a, b = previous.get(name), row.get(name)
            if a is None or b is None:
                raise ValueError(f'missing paired initial evidence: {name}')
            if name in {'reference_evidence_sha256', 'point_indices_sha256'}:
                matches = a == b
            elif a is None or b is None:
                matches = a is None and b is None
            else:
                aa, bb = np.asarray(a), np.asarray(b)
                matches = aa.shape == bb.shape and np.allclose(aa, bb, rtol=1e-7, atol=1e-6)
            if not matches:
                raise ValueError(f'inconsistent paired initial evidence: {name}')
    if any(cohort != next(iter(cohorts.values())) for cohort in cohorts.values()):
        raise ValueError('All methods must have the same trial cohort; explicitly filter a common cohort before summarizing')
    rows.sort(key=lambda row: (row['method'], row['rotation_deg'], row['translation_mm'],
                              row['patient_id'], row['source'], row['case_id'], row['jaw'], row['repeat']))
    rng = np.random.default_rng(seed)
    report = dict(schema_version=1, interpretation=_DIAGNOSTIC, trials=rows, grid=[], patients=[],
                  paired_comparisons=[], bootstrap=dict(unit='patient', samples=samples, seed=seed, confidence=.95),
                  error_statistics='D/energy means conditional on available valid results; matched pairs use joint validity.',
                  failure_definition='Numerical or invalid-output failures; retained in all-start failure and success denominators.',
                  capture_definition='Final success among initially outside-threshold starts, with numerical failures counted unsuccessful.',
                  gradient_definition='Actual autograd energy gradient versus smooth anchor-RMS gradient (smoothing=1 mm), '
                                      'both rotation blocks divided by optimizer radius. Positive dot predicts infinitesimal '
                                      'unclipped target descent. Direction rates use finite gradients with nonzero target only; '
                                      'zero energy gradient counts stalled. Cosine also requires nonzero energy gradient. '
                                      'GT stationary targets and diagnostic failures are excluded, with explicit trial counts.',
                  aggregation='Mean repeats within source/case/jaw, then equal-weight cases within patient; bootstrap patients.')
    cells = sorted({(row['method'], row['rotation_deg'], row['translation_mm']) for row in rows})
    for method, rotation, translation in cells:
        selected = [row for row in rows if (row['method'], row['rotation_deg'], row['translation_mm']) == (method, rotation, translation)]
        patients = _patient_summary(selected)
        cell = dict(method=method, rotation_deg=rotation, translation_mm=translation,
                    reference_kind=selected[0]['reference_kind'], n_trials=len(selected), n_patients=len(patients),
                    n_cases=sum(p['n_cases'] for p in patients), n_failed_trials=sum(r['failed'] for r in selected),
                    n_capture_eligible_trials=sum(not r['initial_success'] for r in selected), **_gradient_counts(selected))
        for metric in (*_ERROR_METRICS, *_RATE_FIELDS, *_CONDITIONAL_RATES):
            cell[metric] = _bootstrap([p[metric] for p in patients if p[metric] is not None], samples, rng)
        report['grid'].append(cell)
        report['patients'].extend(dict(method=method, rotation_deg=rotation, translation_mm=translation) | p for p in patients)
    for a, b in combinations(sorted({row['method'] for row in rows}), 2):
        for rotation, translation in sorted({(row['rotation_deg'], row['translation_mm']) for row in rows}):
            def lookup(method):
                return {(r['patient_id'], r['source'], r['case_id'], r['jaw'], r['repeat']): r for r in rows
                        if (r['method'], r['rotation_deg'], r['translation_mm']) == (method, rotation, translation)}
            aa, bb = lookup(a), lookup(b)
            common = sorted(aa.keys() & bb.keys())
            if not common:
                continue
            pair = dict(method_a=a, method_b=b, rotation_deg=rotation, translation_mm=translation,
                        n_common_trials=len(common), n_common_patients=len({key[0] for key in common}))
            for metric, field in [('final_d_mm', 'final_d_mm'), ('delta_d_mm', 'delta_d_mm'),
                                  ('failure_rate', 'failed'), ('success_rate', 'success'),
                                  ('improvement_rate', 'improved'), ('deterioration_rate', 'worsened'),
                                  ('initial_gradient_cosine', 'initial_gradient_cosine'),
                                  ('initial_gradient_descent_rate', 'initial_gradient_descent')]:
                patient_cases = defaultdict(lambda: defaultdict(list))
                for key in common:
                    va, vb = aa[key].get(field), bb[key].get(field)
                    if va is not None and vb is not None:
                        patient_cases[key[0]][key[1:4]].append(float(vb) - float(va))
                values = [float(np.mean([np.mean(v) for v in cases.values()])) for cases in patient_cases.values()]
                pair[f'{metric}_b_minus_a'] = _bootstrap(values, samples, rng)
            report['paired_comparisons'].append(pair)
    return report


def _benchmark_model_case(model, case, record, method, settings):
    # Function scope releases this context and query before another case/model.
    inference_case = {key: case[key] for key in ('image', 'affine', 'points', 'anchors')}
    device = settings['device']
    try:
        with torch.no_grad():
            context = encode_case(model, inference_case, device)
        affine = tensor(case['affine'], device)[None]
        jaw = torch.tensor([int(record['jaw'] == 'lower')], device=device)

        def query(points):
            return model.registration_query(context, points, affine, jaw)
    except (FloatingPointError, RuntimeError, ValueError, OverflowError) as exc:
        reason = f'field encoding failed: {type(exc).__name__}: {exc}'

        def query(points, reason=reason):
            raise RuntimeError(reason)
    return benchmark_field(query, points=case['points'], anchors=case['anchors'],
                           reference_transform=case['transform'], affine=case['affine'],
                           input_shape=case['image'].shape, record=record, method=method, **settings)


def benchmark_landscape(manifest, checkpoints, output_dir, *, split='test', source=None,
                        reference_kind=None, data_root=None, device='cuda',
                        rotation_degrees=(0., 5., 10., 20., 40.), translation_mm=(0., 1., 2., 4., 8.),
                        repeats=5, refinement_steps=20, learning_rate=.25, point_budget=4096,
                        seed=0, bootstrap_samples=2000, success_threshold_mm=2., improvement_tolerance_mm=1e-4):
    """Evaluate named checkpoints under one replayable, reference-centered protocol."""
    _settings(refinement_steps, learning_rate, success_threshold_mm, improvement_tolerance_mm)
    rotations = _grid(rotation_degrees, 'rotation_degrees', 180.)
    translations = _grid(translation_mm, 'translation_mm')
    for name, value, minimum in [('repeats', repeats, 1), ('point_budget', point_budget, 1),
                                 ('bootstrap_samples', bootstrap_samples, 1), ('seed', seed, 0)]:
        _integer(value, name, minimum)
    if split not in {'val', 'test', 'external_test'}:
        raise ValueError('convergence diagnosis requires val, test or external_test')
    if not checkpoints or any(not isinstance(name, str) or not name.strip() for name in checkpoints):
        raise ValueError('checkpoints must map explicit method names to paths')
    if reference_kind not in {None, 'manual', 'silver'}:
        raise ValueError('reference_kind filter must be manual or silver')
    records = load_journal_manifest(manifest, data_root=data_root)
    selected = [r for r in records if r['split'] == split and (source is None or r['source'] == source)
                and (reference_kind is None or r['reference_kind'] == reference_kind)]
    if not selected:
        raise ValueError('no records match the requested cohort')
    kinds = {r['reference_kind'] for r in selected}
    if not kinds <= {'manual', 'silver'} or len(kinds) != 1:
        raise ValueError('Select exactly one reference_kind: manual and silver must not be pooled')
    selected.sort(key=lambda r: tuple(r[key] for key in ('patient_id', 'source', 'case_id', 'jaw')))
    settings = dict(rotation_degrees=rotations, translation_mm=translations, repeats=repeats,
                    seed=seed, refinement_steps=refinement_steps, learning_rate=learning_rate,
                    point_budget=point_budget, success_threshold_mm=success_threshold_mm,
                    improvement_tolerance_mm=improvement_tolerance_mm, device=device)
    set_seed(seed)
    checkpoint_metadata = {}
    for method, path in sorted(checkpoints.items()):
        saved = load_checkpoint(path, 'cpu')
        if any(r['patient_id'] in set(saved['training_patient_ids']) for r in selected):
            raise ValueError('evaluation patient was used to train the checkpoint')
        if any(r.get('content_hash') in set(saved.get('training_content_hashes', [])) for r in selected):
            raise ValueError('evaluation image content was used to train the checkpoint')
        if split == 'external_test' and {r['source'] for r in selected} & set(saved.get('training_sources', [])):
            raise ValueError('external test source was used to train the checkpoint')
        try:
            selection_patients, selection_hashes, selection_sources = _selection_history(saved, method)
            selection_known = True
        except ValueError:
            if split != 'val':
                raise
            selection_patients = selection_hashes = selection_sources = None
            selection_known = False
        if split != 'val':
            if {r['patient_id'] for r in selected} & selection_patients:
                raise ValueError('evaluation patient was used in checkpoint selection')
            if {r.get('content_hash') for r in selected} & selection_hashes:
                raise ValueError('evaluation image content was used in checkpoint selection')
            if split == 'external_test' and {r['source'] for r in selected} & selection_sources:
                raise ValueError('external test source was used in checkpoint selection')
        checkpoint_metadata[method] = dict(path=str(Path(path).resolve()), sha256=sha256_file(path),
                                           config=saved['config'], training_patient_ids=saved['training_patient_ids'],
                                           training_content_hashes=saved.get('training_content_hashes', []),
                                           training_sources=saved.get('training_sources', []),
                                           provenance_schema_version=saved.get('provenance_schema_version'),
                                           selection_provenance_known=selection_known,
                                           selection_patient_ids=sorted(selection_patients) if selection_known else None,
                                           selection_content_hashes=sorted(selection_hashes) if selection_known else None,
                                           selection_sources=sorted(selection_sources) if selection_known else None)
        del saved
    rows, payloads = [], []
    # One checkpoint and one encoder context at a time: checkpoint ordering must
    # not steal another method's GPU memory budget or manufacture OOM failures.
    for method, path in sorted(checkpoints.items()):
        model, saved = load_field(path, device)
        del saved
        for record in selected:
            case = load_case(record)
            if method == sorted(checkpoints)[0]:
                payloads.append(_identity(record) | dict(reference_kind=record['reference_kind'],
                                npz_path=record['npz_path'], payload_sha256=sha256_file(record['npz_path']),
                                point_indices=_point_indices(record, len(case['points']), point_budget, seed)))
            rows.extend(_benchmark_model_case(model, case, record, method, settings))
        del model
    report = summarize_landscape(rows, bootstrap_samples=bootstrap_samples, seed=seed)
    report['protocol'] = dict(diagnostic_only=True, deployment_result=False, interpretation=_DIAGNOSTIC,
                              settings=settings, split=split, source=source, reference_kind=next(iter(kinds)),
                              cohort_role='descriptive_validation' if split == 'val' else 'held_out_diagnostic',
                              manifest_path=str(Path(manifest).resolve()), manifest_sha256=sha256_file(manifest),
                              manifest_fingerprint=manifest_fingerprint(sorted(records, key=lambda r: (
                                  r['source'], r['case_id'], r['jaw']))), records=payloads,
                              checkpoints=checkpoint_metadata, runtime=runtime_metadata(),
                              rng='NumPy PCG64 seeded by SHA256 of protocol version/identity/seed/purpose/repeat',
                              rotation_center='centroid of all fixed IOS anchors under reference transform',
                              perturbation='proper left rotation about center, followed by exact-mm centroid translation',
                              reference_use='diagnostic starts and post-hoc D only; encoder/refiner receive no reference or labels',
                              optimizer='task2reg.journal.field.refine_transform; inference backtracking; float32 geometry',
                              energy=dict(coverage_weight=1., outside_weight=2., coverage_threshold_mm=2.,
                                          coverage_temperature_mm=.25, robust_epsilon_mm=.1),
                              selection_note='Validation may be used for checkpoint selection; this diagnostic is not independent certification.')
    write_json(Path(output_dir) / 'report.json', report)
    return report
