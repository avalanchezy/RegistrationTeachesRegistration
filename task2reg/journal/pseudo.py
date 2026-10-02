"""Frozen-teacher multistart verification and weighted spatial pseudo export."""
from dataclasses import asdict
import os
from pathlib import Path

import numpy as np
import torch

from .data import load_case, load_journal_manifest, validate_protocol
from .engine import candidate_energies, encode_case, tensor
from .field import centered_increment, refine_transform, roi_outside_distance, transform_points
from .inference import load_field, record_directory, score_and_refine
from .runtime import sha256_file, write_json
from .verification import VerificationConfig, transform_medoid, verify_registration


def spatial_sectors(points, count=4):
    """Equal-count contiguous bins along the crown's first PCA axis.

    These are reproducible geometric sectors, not anatomical tooth labels.
    Use a fixed upstream crown crop; full meshes can include irrelevant gingiva.
    """
    points = np.asarray(points, dtype=float)
    if count < 2 or len(points) < 2 * count or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("at least two points per sector and finite Nx3 points required")
    centered = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    axis = axes[0]
    if axis[np.argmax(abs(axis))] < 0:
        axis = -axis
    order = np.argsort(centered @ axis, kind="stable")
    sectors = np.empty(len(points), dtype=np.int64)
    for sector, indices in enumerate(np.array_split(order, count)):
        sectors[indices] = sector
    return sectors


def verify_case(model, case, record, device, *, config=None, starts=8, sectors=4,
                refinement_steps=20, point_budget=4096, learning_rate=.25, seed=0):
    if starts < sectors or sectors < 2 or point_budget < sectors * 2:
        raise ValueError("starts must cover every sector and point_budget must be >=2*sectors")
    config = VerificationConfig() if config is None else config
    if "candidates" not in case or not len(case["candidates"]):
        raise ValueError("verification requires cached initial candidates")
    candidates = case["candidates"]
    rng = np.random.default_rng(seed)
    subset = rng.choice(len(case["points"]), min(point_budget, len(case["points"])), replace=False)
    points_np = case["points"][subset]
    partition = spatial_sectors(points_np, sectors)
    points = tensor(points_np, device)[None]
    affine = tensor(case["affine"], device)[None]
    jaw = torch.tensor([int(record["jaw"] == "lower")], device=device)
    with torch.no_grad():
        context = encode_case(model, case, device)
    shape = case["image"].shape
    refined, heldout = [], []
    for run in range(starts):
        hold = partition == run % sectors
        fitting = points[:, ~hold]
        # Different subsamples and small physical perturbations make repeated
        # starts useful even when the input candidate cache contains few rows.
        fit_indices = rng.choice(fitting.shape[1], max(3, int(.8 * fitting.shape[1])), replace=False)
        fitting = fitting[:, fit_indices]
        # Rank raw starts on fitting points only. A full-surface-refined start
        # would already contain information from the held-out sector.
        with torch.no_grad():
            fitting_energies, _ = candidate_energies(model, context, fitting, tensor(candidates, device), affine, jaw, shape)
            order = torch.argsort(fitting_energies).cpu().numpy()
        initial = tensor(candidates[order[run % len(order)]], device)[None]
        twist = tensor(np.r_[rng.normal(0, .005, 3), rng.normal(0, .1, 3)], device)[None]
        initial = centered_increment(initial, twist, transform_points(fitting, initial).mean(dim=1))
        result = refine_transform(lambda p: model.query(context, p, affine, jaw), fitting, initial,
                                  affine, shape, steps=refinement_steps, learning_rate=learning_rate)
        transform = result["transform"]
        refined.append(transform[0].cpu().numpy())
        with torch.no_grad():
            query = transform_points(points[:, hold], transform)
            distances = model.query(context, query, affine, jaw)
            # Out-of-volume held-out points cannot appear to be a good fit.
            distances = distances + roi_outside_distance(query, affine, shape)
            heldout.extend(distances[0].cpu().tolist())
    refined = np.asarray(refined)
    # Full-surface refinement is used only after holdout trajectories are frozen,
    # for the independent task of scanning the entire pool for ambiguity.
    rows = score_and_refine(model, case, record, device,
                            refinement_steps=refinement_steps, learning_rate=learning_rate,
                            point_budget=point_budget, seed=seed)
    ambiguity_candidates = np.asarray([r["transform"] for r in rows])
    medoid = refined[transform_medoid(refined, case["anchors"])]
    all_points = tensor(case["points"], device)[None]
    with torch.no_grad():
        query = transform_points(all_points, tensor(medoid, device)[None])
        distances = model.query(context, query, affine, jaw)[0].cpu().numpy()
        outside = roi_outside_distance(query, affine, shape)[0].cpu().numpy() > 1e-5
        pool = np.concatenate((candidates, ambiguity_candidates, refined), axis=0)
        energies, _ = candidate_energies(model, context, points, tensor(pool, device), affine, jaw, shape)
    result = verify_registration(refined, case["points"], anchors=case["anchors"], distances=distances,
                                 heldout_distances=np.asarray(heldout),
                                 sector_ids=spatial_sectors(case["points"], sectors), outside_mask=outside,
                                 candidate_transforms=pool, candidate_energies=energies.cpu().numpy(), config=config)
    result["run_settings"] = dict(starts=starts, sectors=sectors, refinement_steps=refinement_steps,
                                  point_budget=point_budget, learning_rate=learning_rate, seed=seed)
    return result


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def assemble_pseudo_manifest(records, evidence, output_dir, *, teacher_id,
                             excluded_patient_ids, max_cases=0, supervision_radius_mm=2.):
    """Replace complete accepted case groups; rejection never creates labels."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if max_cases < 0:
        raise ValueError("max_cases must be nonnegative")
    evidence_map = {(r["source"], r["case_id"], r["jaw"]): e for r, e in evidence}
    grouped = {}
    for row in records:
        if row["split"] == "unlabeled":
            grouped.setdefault((row["source"], row["case_id"]), []).append(row)
    accepted = [key for key, group in grouped.items()
                if all(evidence_map.get((r["source"], r["case_id"], r["jaw"]), {}).get("accepted", False) for r in group)]
    accepted.sort(key=lambda key: (max(evidence_map[(r["source"], r["case_id"], r["jaw"])]["metrics"]["u95_mm"]
                                       for r in grouped[key]), key))
    if max_cases:
        accepted = accepted[:max_cases]
    accepted = set(accepted)
    output = []
    for row in records:
        copied = dict(row)
        key = row["source"], row["case_id"], row["jaw"]
        if row["split"] == "unlabeled" and key[:2] in accepted:
            case = load_case(row)
            result = evidence_map[key]
            case.pop("label", None)
            case["transform"] = np.asarray(result["transform"], dtype=np.float64)
            case["point_weights"] = np.asarray(result["point_weights"], dtype=np.float32)
            destination = output_dir / "data" / (record_directory(row) + ".npz")
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, **case)
            copied.update(split="pseudo", reference_kind="pseudo", pseudo_kind="field",
                          npz_path=str(destination), supervision_radius_mm=supervision_radius_mm,
                          pseudo_provenance={"teacher_id": teacher_id, "excluded_patient_ids": list(excluded_patient_ids)})
        copied["npz_path"] = os.path.relpath(copied["npz_path"], output_dir)
        output.append(copied)
    validate_protocol(output)
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, {"schema_version": 1, "records": output})
    write_json(output_dir / "verification.json", {"teacher_id": teacher_id,
               "accepted_cases": len(accepted), "evaluated_jaws": len(evidence),
               "records": [{"case_id": r["case_id"], "source": r["source"], "jaw": r["jaw"],
                            "exported": (r["source"], r["case_id"]) in accepted,
                            **_jsonable(e)} for r, e in evidence]})
    return manifest_path


def export_verified_pseudo(manifest, checkpoint, output_dir, *, data_root=None, device="cuda",
                           config=None, starts=8, sectors=4, refinement_steps=20,
                           point_budget=4096, learning_rate=.25, seed=0, max_cases=0):
    records = load_journal_manifest(manifest, data_root)
    targets = [r for r in records if r["split"] == "unlabeled"]
    if not targets:
        raise ValueError("no unlabeled records; locked evaluation sources cannot generate pseudo supervision")
    model, saved = load_field(checkpoint, device)
    evaluation_patients = {r["patient_id"] for r in records if r["split"] in {"val", "test", "external_test"}}
    excluded = set(saved["excluded_patient_ids"])
    if not evaluation_patients.issubset(excluded) or evaluation_patients & set(saved["training_patient_ids"]):
        raise ValueError("teacher does not exclude all current evaluation patients")
    external_sources = {r["source"] for r in records if r["split"] == "external_test"}
    if external_sources & set(saved.get("training_sources", [])):
        raise ValueError("teacher used a locked external source")
    evaluation_hashes = {r["content_hash"] for r in records if r["split"] in {"val", "test", "external_test"} and r.get("content_hash")}
    if evaluation_hashes & set(saved.get("training_content_hashes", [])):
        raise ValueError("teacher used evaluation image content")
    evidence = []
    for index, row in enumerate(targets):
        result = verify_case(model, load_case(row), row, device, config=config,
                             starts=starts, sectors=sectors, refinement_steps=refinement_steps,
                             point_budget=point_budget, learning_rate=learning_rate, seed=seed + index)
        evidence.append((row, result))
        print(f"{row['case_id']} {row['jaw']}: {'accepted' if result['accepted'] else ', '.join(result['reasons'])}", flush=True)
    return assemble_pseudo_manifest(records, evidence, output_dir, teacher_id=sha256_file(checkpoint),
                                    excluded_patient_ids=sorted(excluded), max_cases=max_cases)
