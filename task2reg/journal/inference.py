"""Field scoring/refinement of cached candidates, with no reference-bank route."""
from pathlib import Path
import hashlib
import re

import numpy as np
import torch

from .config import TrainingConfig
from .data import load_case, load_journal_manifest
from .field import RegistrationField, refine_transform
from .engine import _selection_history, candidate_energies, encode_case, tensor
from .runtime import load_checkpoint, sha256_file, write_json


def record_directory(record):
    raw = f"{record['source']}:{record['case_id']}:{record['jaw']}"
    stem = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw)[:100]
    return stem + "_" + hashlib.sha256(raw.encode()).hexdigest()[:10]


def load_field(path, device):
    saved = load_checkpoint(path, "cpu")
    config = TrainingConfig(**saved["config"])
    model = RegistrationField(config.base_channels, config.mode, config.truncation_mm, config.energy_mode).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    model.requires_grad_(False)
    return model, saved


def score_and_refine(model, case, record, device, *, refinement_steps=20,
                     learning_rate=.25, point_budget=4096, seed=0):
    if "candidates" not in case or not len(case["candidates"]):
        raise ValueError(f"{record['case_id']} has no candidate initializations")
    rng = np.random.default_rng(seed)
    subset = rng.choice(len(case["points"]), min(point_budget, len(case["points"])), replace=False)
    points = tensor(case["points"][subset], device)[None]
    affine = tensor(case["affine"], device)[None]
    jaw = torch.tensor([int(record["jaw"] == "lower")], device=device)
    with torch.no_grad():
        context = encode_case(model, case, device)
    initial = tensor(case["candidates"], device)
    rows = []
    for index in range(len(initial)):
        result = refine_transform(lambda p: getattr(model, "registration_query", model.query)(context, p, affine, jaw), points,
                                  initial[index:index+1], affine, case["image"].shape,
                                  steps=refinement_steps, learning_rate=learning_rate)
        rows.append({"candidate_id": f"cached:{index}", "initial_transform": initial[index].cpu().tolist(),
                     "transform": result["transform"][0].detach().cpu().tolist(),
                     "initial_field_energy": float(result["initial_energy"][0]),
                     "field_energy": float(result["final_energy"][0]),
                     "initial_coverage": float(result["initial_coverage"][0]),
                     "field_coverage": float(result["final_coverage"][0]),
                     "field_outside_fraction": float(result["final_outside_fraction"][0]),
                     "method": "journal_field_refinement"})
    return rows


def predict(manifest, checkpoint, output_dir, *, split="test", device="cuda",
            data_root=None, refinement_steps=20, learning_rate=.25,
            point_budget=4096, seed=0, evaluate=False):
    if refinement_steps < 0 or point_budget < 1:
        raise ValueError("refinement_steps must be nonnegative and point_budget positive")
    records = load_journal_manifest(Path(manifest), data_root=data_root)
    selected = [r for r in records if r["split"] == split]
    if not selected:
        raise ValueError(f"no records in split {split}")
    model, saved = load_field(checkpoint, device)
    used = set(saved["training_patient_ids"])
    if split in {"val", "test", "external_test"} and any(r["patient_id"] in used for r in selected):
        raise ValueError("evaluation patient was used to train the checkpoint")
    if split in {"val", "test", "external_test"} and any(r.get("content_hash") in set(saved.get("training_content_hashes", [])) for r in selected):
        raise ValueError("evaluation image content was used to train the checkpoint")
    if split == "external_test" and {r["source"] for r in selected} & set(saved.get("training_sources", [])):
        raise ValueError("external test source was used to train the checkpoint")
    if split in {"test", "external_test"}:
        selection_patients, selection_hashes, selection_sources = _selection_history(saved, "evaluation checkpoint")
        if {r["patient_id"] for r in selected} & selection_patients:
            raise ValueError("evaluation patient was used for checkpoint selection")
        if {r["content_hash"] for r in selected if r.get("content_hash")} & selection_hashes:
            raise ValueError("evaluation image content was used for checkpoint selection")
        if split == "external_test" and {r["source"] for r in selected} & selection_sources:
            raise ValueError("external test source was used for checkpoint selection")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, evaluation = [], []
    for record in selected:
        case = load_case(record)
        row = {k: record[k] for k in ("case_id", "source", "patient_id", "jaw", "reference_kind")}
        if evaluate and (record["reference_kind"] not in {"manual", "silver"} or "transform" not in case):
            raise ValueError("evaluation requires an explicit manual/silver reference")
        try:
            candidates = score_and_refine(model, case, record, device,
                                          refinement_steps=refinement_steps, learning_rate=learning_rate,
                                          point_budget=point_budget, seed=seed)
            energies = np.asarray([c["field_energy"] for c in candidates])
            if not np.all(np.isfinite(energies)):
                raise FloatingPointError(f"nonfinite candidate energy: {record['case_id']}")
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            row.update(failed=True, failure_reason=f"{type(exc).__name__}: {exc}", confidence=None)
            rows.append(row)
            if evaluate:
                evaluation.append(row | {"method": "journal_field"})
            write_json(output_dir / record_directory(record) / "failure.json", row)
            continue
        best = int(np.argmin(energies))
        initial_best = int(np.argmin([c["initial_field_energy"] for c in candidates]))
        ordered = np.sort(energies)
        confidence = float((ordered[1] - ordered[0]) / (abs(ordered[0]) + 1e-6)) if len(ordered) > 1 else 0.
        directory = output_dir / record_directory(record)
        write_json(directory / "candidates.json", candidates)
        row.update(candidates=candidates, selected_index=best, initial_selected_index=initial_best,
                   transform=candidates[best]["transform"], confidence=confidence,
                   failed=False,
                   confidence_definition="relative energy gap; uncalibrated")
        rows.append(row)
        if evaluate:
            evaluation.append({k: row[k] for k in ("case_id", "source", "patient_id", "jaw", "selected_index", "initial_selected_index", "confidence")}
                              | {"method": "journal_field", "reference_kind": record["reference_kind"],
                                 "anchors": case["anchors"].tolist(),
                                 "reference_transform": case["transform"].tolist(),
                                 "initial_candidates": case["candidates"].tolist(),
                                 "final_candidates": [c["transform"] for c in candidates]})
    output = {"schema_version": 1, "checkpoint_sha256": sha256_file(checkpoint),
              "split": split, "refinement_steps": refinement_steps, "point_budget": point_budget,
              "seed": seed, "records": rows}
    write_json(output_dir / "predictions.json", output)
    if evaluate:
        write_json(output_dir / "evaluation_input.json", {"schema_version": 1, "records": evaluation})
    return output
