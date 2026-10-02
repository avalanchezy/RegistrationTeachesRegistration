"""Independent-reference audits of a fixed registration verifier.

Development results are descriptive. A frozen audit evaluates one policy on new
patients; its binomial bound assumes independent, representative patient draws.
It is neither a clinical safety certificate nor a guarantee under source shift.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from .data import load_case, load_journal_manifest
from .inference import load_field, record_directory
from .pseudo import _jsonable, verify_case
from .runtime import canonical_hash, manifest_fingerprint, runtime_metadata, sha256_file, write_json
from .verification import VerificationConfig, pose_displacement


def binomial_upper(errors, count, alpha=.05):
    """One-sided Clopper-Pearson upper bound; no accepted patients means no evidence."""
    if (not isinstance(errors, (int, np.integer)) or not isinstance(count, (int, np.integer))
            or not 0 <= errors <= count or not 0 < alpha < 1):
        raise ValueError("require integer 0 <= errors <= count and 0 < alpha < 1")
    return 1. if count == 0 or errors == count else float(beta.ppf(1 - alpha, errors + 1, count - errors))


def hidden_case(case):
    """Allowlist deployable inputs so future extra reference arrays cannot leak."""
    return {key: case[key] for key in ("image", "affine", "points", "anchors", "candidates") if key in case}


def judge_result(result, reference, anchors, *, success_threshold_mm=2.):
    """Attach truth only after the verifier has completed its decisions."""
    row = dict(accepted=bool(result["accepted"]), reasons=result.get("reasons", []),
               metrics=result.get("metrics", {}), d_mm=None, parity_mismatch=None,
               registration_failed=True, verification_failed=False)
    try:
        transform = result.get("transform")
        if transform is None:
            raise ValueError("verifier returned no transform")
        distance = pose_displacement(transform, reference, anchors)
        mismatch = bool(np.sign(np.linalg.det(np.asarray(transform)[:3, :3])) !=
                        np.sign(np.linalg.det(np.asarray(reference)[:3, :3])))
        row.update(d_mm=distance, parity_mismatch=mismatch,
                   registration_failed=bool(mismatch or distance > success_threshold_mm))
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        row.update(verification_failed=True, failure_reason=str(exc))
    return row


def summarize_audit(rows, *, alpha=.05, risk_target=.05):
    """Complete available jaws per case; any erroneous exported jaw fails its patient."""
    if not rows or not 0 < risk_target < 1:
        raise ValueError("nonempty audit and 0 < risk_target < 1 required")
    if {r["reference_kind"] for r in rows} != {"manual"}:
        raise ValueError("verification risk audit requires independent manual references")
    identities, cases = set(), {}
    for row in rows:
        identity = row["source"], row["case_id"], row["jaw"]
        if identity in identities:
            raise ValueError("duplicate audit record")
        identities.add(identity)
        cases.setdefault(identity[:2], []).append(row)
    accepted, patients = [], {}
    for group in cases.values():
        if len({r["patient_id"] for r in group}) != 1:
            raise ValueError("inconsistent patient identity within case")
        patient = group[0]["patient_id"]
        item = patients.setdefault(patient, dict(patient_id=patient, accepted=False, incorrect=False))
        if all(r["accepted"] for r in group):
            accepted.append(group)
            item["accepted"] = True
            item["incorrect"] |= any(r["registration_failed"] for r in group)
    retained = [p for p in patients.values() if p["accepted"]]
    errors = sum(p["incorrect"] for p in retained)
    upper = binomial_upper(errors, len(retained), alpha)
    accepted_jaws = [r for group in accepted for r in group]
    return dict(n_jaws=len(rows), n_cases=len(cases), n_patients=len(patients),
                n_accepted_cases=len(accepted), n_accepted_jaws=len(accepted_jaws),
                n_accepted_patients=len(retained), n_incorrect_accepted_patients=errors,
                case_coverage=len(accepted) / len(cases), patient_coverage=len(retained) / len(patients),
                accepted_patient_risk=errors / len(retained) if retained else None,
                accepted_jaw_error_fraction=float(np.mean([r["registration_failed"] for r in accepted_jaws])) if accepted_jaws else None,
                n_verification_failures=sum(r["verification_failed"] for r in rows),
                risk_upper_bound=upper, confidence_level=1 - alpha, risk_target=risk_target,
                meets_risk_target=bool(retained and upper <= risk_target),
                patients=sorted(patients.values(), key=lambda p: p["patient_id"]),
                risk_event="any incorrect accepted jaw among all accepted cases of one patient",
                bound_assumptions="one fixed policy; independent representative patients; complete case roster; no adaptive audit reuse")


def check_audit_cohort(selected, saved, *, purpose, split, policy=None):
    if purpose not in {"development", "frozen"}:
        raise ValueError("purpose must be development or frozen")
    if purpose == "development" and split != "val":
        raise ValueError("development audit uses val; do not tune on locked test sources")
    if purpose == "frozen" and split not in {"test", "external_test"}:
        raise ValueError("frozen audit requires test or external_test")
    patients = {r["patient_id"] for r in selected}
    hashes = {r["content_hash"] for r in selected if r.get("content_hash")}
    sources = {r["source"] for r in selected}
    if patients & set(saved["training_patient_ids"]) or hashes & set(saved.get("training_content_hashes", [])):
        raise ValueError("audit patient/content was used for training")
    if split == "external_test" and sources & set(saved.get("training_sources", [])):
        raise ValueError("locked external source was used for training")
    if purpose == "frozen":
        if (saved.get("provenance_schema_version") != 2 or
                not all(isinstance(saved.get(k), list) for k in ("selection_patient_ids", "selection_content_hashes", "selection_sources"))):
            raise ValueError("frozen audit requires explicit checkpoint selection provenance v2")
        if patients & set(saved["selection_patient_ids"]) or hashes & set(saved["selection_content_hashes"]):
            raise ValueError("audit patient/content was used for checkpoint selection")
        if split == "external_test" and sources & set(saved["selection_sources"]):
            raise ValueError("locked external source was used for checkpoint selection")
        if not isinstance(policy, dict) or not all(isinstance(policy.get(k), list) for k in
                ("development_patient_ids", "development_content_hashes", "development_sources")):
            raise ValueError("frozen audit requires explicit policy development provenance")
        if patients & set(policy["development_patient_ids"]) or hashes & set(policy["development_content_hashes"]):
            raise ValueError("audit patient/content was used for policy development")
        if split == "external_test" and sources & set(policy["development_sources"]):
            raise ValueError("locked external source was used for policy development")


def _settings(*, starts, sectors, refinement_steps, point_budget, learning_rate, seed):
    if (starts < sectors or sectors < 2 or point_budget < 2 * sectors or
            refinement_steps < 0 or not np.isfinite(learning_rate) or learning_rate <= 0 or seed < 0):
        raise ValueError("invalid verification run settings")
    return dict(starts=starts, sectors=sectors, refinement_steps=refinement_steps,
                point_budget=point_budget, learning_rate=learning_rate, seed=seed)


def freeze_policy(selected_audit, output_path, *, additional_development_audits=()):
    """Freeze one chosen policy; list all explored development audits for exposure tracking."""
    paths = [Path(selected_audit), *map(Path, additional_development_audits)]
    audits = [json.loads(path.read_text()) for path in paths]
    if any(a.get("format") != "rtr-verification-audit-v1" or a.get("purpose") != "development"
           or a.get("status") != "complete" for a in audits):
        raise ValueError("only complete development audits may define a frozen policy")
    chosen = audits[0]
    policy = dict(format="rtr-frozen-verification-policy-v1", frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  protocol=chosen["protocol"], development_audit_sha256=[sha256_file(path) for path in paths])
    for key in ("patient_ids", "content_hashes", "sources"):
        policy["development_" + key] = sorted({value for audit in audits for value in audit["cohort"][key]})
    policy["policy_sha256"] = canonical_hash(policy)
    if Path(output_path).exists():
        raise FileExistsError("frozen policies are immutable; use a new versioned path")
    write_json(output_path, policy)
    return policy


def audit_verifier(manifest, checkpoint, output_dir, *, config=None, data_root=None, device="cuda",
                   split="val", purpose="development", policy_path=None, source=None,
                   starts=8, sectors=4, refinement_steps=20, point_budget=4096,
                   learning_rate=.25, seed=0, success_threshold_mm=2., alpha=.05, risk_target=.05):
    if not np.isfinite(success_threshold_mm) or success_threshold_mm <= 0:
        raise ValueError("success threshold must be positive and finite")
    if not 0 < risk_target < 1:
        raise ValueError("risk_target must lie strictly between zero and one")
    binomial_upper(0, 0, alpha)
    config = VerificationConfig() if config is None else config
    settings = _settings(starts=starts, sectors=sectors, refinement_steps=refinement_steps,
                         point_budget=point_budget, learning_rate=learning_rate, seed=seed)
    records = load_journal_manifest(manifest, data_root)
    selected = [r for r in records if r["split"] == split and (source is None or r["source"] == source)]
    if not selected or any(r["reference_kind"] != "manual" for r in selected):
        raise ValueError("nonempty cohort with manual references required; silver is not independent truth")
    model, saved = load_field(checkpoint, device)
    metadata = runtime_metadata()
    protocol = dict(checkpoint_sha256=sha256_file(checkpoint), gate_config=asdict(config), run_settings=settings,
                    success_threshold_mm=success_threshold_mm, alpha=alpha, risk_target=risk_target,
                    verifier_source_sha256=metadata["source_sha256"])
    policy = json.loads(Path(policy_path).read_text()) if policy_path else None
    if purpose == "frozen":
        if not policy or policy.get("format") != "rtr-frozen-verification-policy-v1":
            raise ValueError("provide a frozen policy made from a development audit")
        if canonical_hash({k: v for k, v in policy.items() if k != "policy_sha256"}) != policy.get("policy_sha256"):
            raise ValueError("frozen policy hash mismatch")
        if policy["protocol"] != protocol:
            raise ValueError("checkpoint, code, gates or run settings changed since policy freeze")
    elif policy is not None:
        raise ValueError("policy is only used for frozen audits")
    check_audit_cohort(selected, saved, purpose=purpose, split=split, policy=policy)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("use an empty audit directory; preserve previous evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort = dict(patient_ids=sorted({r["patient_id"] for r in selected}),
                  content_hashes=sorted({r["content_hash"] for r in selected if r.get("content_hash")}),
                  sources=sorted({r["source"] for r in selected}), manifest_fingerprint=manifest_fingerprint(selected))
    report = dict(format="rtr-verification-audit-v1", purpose=purpose, split=split, protocol=protocol,
                  policy_sha256=policy.get("policy_sha256") if policy else None,
                  cohort=cohort, runtime=metadata, records=[], status="running")
    write_json(output_dir / "audit.json", report)
    for record in sorted(selected, key=lambda r: (r["source"], r["case_id"], r["jaw"])):
        # Invalid inputs abort clearly rather than disappearing from a risk denominator.
        case = load_case(record)
        identity = {k: record[k] for k in ("source", "case_id", "patient_id", "jaw", "reference_kind")}
        try:
            result = verify_case(model, hidden_case(case),
                                 {k: identity[k] for k in ("source", "case_id", "patient_id", "jaw")},
                                 device, config=config, **settings)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            result = dict(accepted=False, reasons=[f"verification_failure: {type(exc).__name__}: {exc}"],
                          transform=None, metrics={})
        judged = judge_result(result, case["transform"], case["anchors"], success_threshold_mm=success_threshold_mm)
        row = identity | judged
        # Store frozen decision evidence independently of the later truth comparison.
        write_json(output_dir / "evidence" / (record_directory(record) + ".json"), _jsonable(result))
        report["records"].append(row)
        write_json(output_dir / "audit.json", report)
        print(f"{record['case_id']} {record['jaw']}: accepted={row['accepted']} D_mm={row['d_mm']}", flush=True)
    report.update(status="complete", summary=summarize_audit(report["records"], alpha=alpha, risk_target=risk_target))
    report["by_source"] = {s: summarize_audit([r for r in report["records"] if r["source"] == s], alpha=alpha, risk_target=risk_target)
                           for s in cohort["sources"]}
    report["supports_frozen_risk_target"] = purpose == "frozen" and report["summary"]["meets_risk_target"]
    report["interpretation"] = ("One frozen policy on an independent declared cohort; sampling assumptions still apply."
                                if purpose == "frozen" else "Descriptive development audit; no independent risk certification.")
    write_json(output_dir / "audit.json", _jsonable(report))
    return report
