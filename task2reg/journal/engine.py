"""Single-device field training with strict splits and resumable state."""
import json
from pathlib import Path
import random

import numpy as np
import torch

from task2reg.crown_network import crown_localizer_loss, normalize_hu
from .config import TrainingConfig
from .data import load_journal_manifest, load_case, sample_queries
from .field import (RegistrationField, eikonal_loss, fixed_point_error,
                    pairwise_rank_loss, refine_transform, registration_energy,
                    transform_points, weighted_huber_loss)
from .runtime import (atomic_checkpoint, float_context, load_checkpoint,
                      manifest_fingerprint, restore_rng, rng_state, runtime_metadata,
                      set_seed, write_json, restore_scaler, sha256_file)


def tensor(value, device):
    return torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32, device=device)


def encode_case(model, case, device, *, amp=False, jitter_hu=0., rng=None):
    image = case["image"].astype(np.float32)
    if jitter_hu:
        image = image + float(rng.normal(0, jitter_hu))
    image = tensor(normalize_hu(image), device)[None, None]
    with torch.autocast(device_type=torch.device(device).type, enabled=amp):
        context = model.encode(image)
    return float_context(context)


def candidate_energies(model, context, points, candidates, affine, jaw, shape, *, chunk_size=8):
    count = candidates.shape[0]
    if count < 1 or chunk_size < 1:
        raise ValueError("candidate count and chunk_size must be positive")
    if count > chunk_size:
        chunks = [candidate_energies(model, context, points, candidates[start:start+chunk_size],
                                     affine, jaw, shape, chunk_size=chunk_size)
                  for start in range(0, count, chunk_size)]
        return (torch.cat([item[0] for item in chunks]),
                {key: torch.cat([item[1][key] for item in chunks]) for key in chunks[0][1]})
    moved = transform_points(points.expand(count, -1, -1), candidates)
    distances = model.query(context, moved.reshape(1, -1, 3), affine, jaw).reshape(count, -1)
    energies, diagnostics = registration_energy(distances, moved, affine.expand(count, -1, -1), shape)
    return energies, diagnostics


def case_loss(model, case, record, config, device, rng, epoch, *, training):
    context = encode_case(model, case, device,
                          amp=config.amp and torch.device(device).type == "cuda",
                          jitter_hu=config.intensity_jitter_hu if training else 0., rng=rng)
    affine = tensor(case["affine"], device)[None]
    jaw = torch.tensor([int(record["jaw"] == "lower")], device=device)
    shape = case["image"].shape
    loss = context["logits"].sum() * 0
    metrics = {}
    pseudo = record["split"] == "pseudo"
    support_only = pseudo and record.get("pseudo_kind", "field") == "support"
    if not pseudo and config.support_weight and "label" not in case:
        raise ValueError("support loss requires a manual label volume; explicitly set support_weight=0 for field-only training")
    if "label" in case and (not pseudo or support_only) and config.support_weight:
        support, _ = crown_localizer_loss(context["logits"],
                                         torch.as_tensor(case["label"], device=device).long()[None])
        loss = loss + config.support_weight * support
        metrics["support"] = float(support.detach())
    if support_only:
        if "label" not in case:
            raise ValueError("legacy support pseudo case requires a label volume")
        return loss, metrics
    queries = sample_queries(case, config.queries, rng, truncation_mm=config.truncation_mm)
    points_world = tensor(queries["points_world"], device)[None]
    targets = tensor(queries["targets"], device)[None]
    weights = tensor(queries["weights"], device)[None]
    prediction = model.query(context, points_world, affine, jaw)
    reconstruction = weighted_huber_loss(prediction, targets, weights)
    loss = loss + config.field_weight * reconstruction
    metrics["field"] = float(reconstruction.detach())
    if config.eikonal_weight and training and not pseudo:
        query = points_world[:, :config.eikonal_queries].detach().requires_grad_(True)
        values = model.query(context, query, affine, jaw)
        regularizer = eikonal_loss(values, query, targets[:, :config.eikonal_queries])
        loss = loss + config.eikonal_weight * regularizer
        metrics["eikonal"] = float(regularizer.detach())
    # A pseudo transform is never a manual target for ranking or pose supervision.
    if not pseudo and (config.rank_weight or config.pose_weight):
        if "candidates" not in case or len(case["candidates"]) < 2:
            raise ValueError("ranking/pose supervision requires at least two cached OOF or geometry-only candidates")
        anchors = tensor(case.get("anchors", case["points"]), device)[None]
        reference = tensor(case["transform"], device)[None]
        candidates = tensor(case["candidates"], device)
        errors = fixed_point_error(candidates, reference.expand(len(candidates), -1, -1),
                                   anchors.expand(len(candidates), -1, -1))
        # Stratify by actual displacement, then randomize within bins. Include
        # the exact GT only during supervised learning, never in inference.
        if training:
            groups = [torch.where((errors >= low) & (errors < high))[0].cpu().numpy()
                      for low, high in ((0,.5),(.5,2),(2,5),(5,15),(15,float("inf")))]
            queues = [list(rng.permutation(g)) for g in groups]
            selected = []
            while any(queues) and len(selected) < max(1, config.max_candidates - 1):
                for queue in queues:
                    if queue and len(selected) < max(1, config.max_candidates - 1):
                        selected.append(queue.pop())
            candidates = torch.cat((reference, candidates[selected]), dim=0)
            errors = fixed_point_error(candidates, reference.expand(len(candidates), -1, -1),
                                       anchors.expand(len(candidates), -1, -1))
        subset = rng.choice(len(case["points"]), min(config.candidate_points, len(case["points"])), replace=False)
        points = tensor(case["points"][subset], device)[None]
        energies, _ = candidate_energies(model, context, points, candidates, affine, jaw, shape)
        ranking = pairwise_rank_loss(energies, errors)
        ramp = min(1., max(0., (epoch - config.rank_warmup_epochs + 1) / max(1, config.rank_ramp_epochs)))
        loss = loss + config.rank_weight * ramp * ranking
        metrics["rank"] = float(ranking.detach())
        metrics["selected_D_mm"] = float(errors[energies.detach().argmin()])
        if config.pose_weight and training and epoch >= config.rank_warmup_epochs:
            eligible = torch.where((errors > .5) & (errors < 15.))[0]
            if len(eligible):
                index = int(eligible[int(rng.integers(len(eligible)))])
                refined = refine_transform(lambda p: model.query(context, p, affine, jaw), points,
                                           candidates[index:index+1], affine, shape,
                                           steps=config.pose_steps,
                                           learning_rate=config.refinement_learning_rate,
                                           differentiable=True)
                pose = fixed_point_error(refined["transform"], reference, anchors).mean()
                loss = loss + config.pose_weight * pose
                metrics["pose"] = float(pose.detach())
    return loss, metrics


def _resume_contract(config):
    return {k: v for k, v in config.as_dict().items() if k not in {"epochs", "device"}}


def select_pseudo_records(rows, max_cases):
    keys = sorted({(r["source"], r["case_id"]) for r in rows})
    chosen = set(keys[:max_cases] if max_cases else keys)
    return [r for r in rows if (r["source"], r["case_id"]) in chosen]


@torch.no_grad()
def validation_score(model, case, record, config, device):
    """One fixed patient-aggregated criterion, independent of training schedules."""
    context = encode_case(model, case, device)
    affine = tensor(case["affine"], device)[None]
    jaw = torch.tensor([int(record["jaw"] == "lower")], device=device)
    rng = np.random.default_rng(config.seed + 100000)
    if config.validation_metric == "field_loss":
        q = sample_queries(case, config.queries, rng, config.truncation_mm)
        pred = model.query(context, tensor(q["points_world"], device)[None], affine, jaw)
        return float(weighted_huber_loss(pred, tensor(q["targets"], device)[None], tensor(q["weights"], device)[None]))
    if "candidates" not in case or not len(case["candidates"]):
        raise ValueError("selected_D_mm validation requires cached validation candidates")
    selected = rng.choice(len(case["points"]), min(config.candidate_points, len(case["points"])), replace=False)
    points = tensor(case["points"][selected], device)[None]
    candidates = tensor(case["candidates"], device)
    energies, _ = candidate_energies(model, context, points, candidates, affine, jaw, case["image"].shape)
    best = candidates[energies.argmin()][None]
    return float(fixed_point_error(best, tensor(case["transform"], device)[None], tensor(case["anchors"], device)[None])[0])


def train(manifest, output_dir, config, *, data_root=None, resume=None, initialize_support=None, initialize_field=None,
          initialization_provenance=None):
    records = load_journal_manifest(Path(manifest), data_root=data_root)
    train_rows = [r for r in records if r["split"] == "train"]
    val_rows = [r for r in records if r["split"] == "val"]
    pseudo_rows = [r for r in records if r["split"] == "pseudo"]
    if not train_rows or not val_rows:
        raise ValueError("training requires nonempty patient-disjoint train and val splits")
    if config.ssl_strategy == "none":
        pseudo_rows = []
    else:
        expected = "support" if config.ssl_strategy == "legacy_support" else "field"
        pseudo_rows = [r for r in pseudo_rows if r.get("pseudo_kind", "field") == expected]
        if not pseudo_rows:
            raise ValueError(f"{config.ssl_strategy} requires matching pseudo records")
        # Budget is in paired cases, never individual jaws. Keep both jaws.
        pseudo_rows = select_pseudo_records(pseudo_rows, config.max_pseudo_cases)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; choose --device cpu for a smoke run")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume and (output_dir / "last.pt").exists():
        raise FileExistsError("run already exists; use --resume or a fresh output directory")
    fingerprint = manifest_fingerprint(records)
    set_seed(config.seed)
    model = RegistrationField(config.base_channels, config.mode, config.truncation_mm).to(device)
    initialization = None
    if initialize_support or initialize_field:
        if resume or (initialize_support and initialize_field):
            raise ValueError("resume, initialize_field and initialize_support are mutually exclusive")
        if initialize_field:
            initial_checkpoint = load_checkpoint(initialize_field)
            for name in ("base_channels", "mode", "truncation_mm"):
                if getattr(config, name) != initial_checkpoint["config"][name]:
                    raise ValueError(f"field initialization architecture differs: {name}")
            initialization = {key: initial_checkpoint.get(key, []) for key in
                              ("training_patient_ids", "training_sources", "training_content_hashes", "excluded_patient_ids")}
        else:
            if initialization_provenance is None:
                raise ValueError("support initialization requires audited initialization provenance")
            initialization = json.loads(Path(initialization_provenance).read_text())
        required = {r["patient_id"] for r in records if r["split"] in {"val", "test", "external_test"}}
        if not required.issubset(set(initialization.get("excluded_patient_ids", []))):
            raise ValueError("initialization provenance does not exclude all evaluation patients")
        for key in ("training_patient_ids", "training_sources", "training_content_hashes"):
            if key not in initialization or not isinstance(initialization[key], list):
                raise ValueError(f"initialization provenance requires {key}")
        if required & set(initialization["training_patient_ids"]):
            raise ValueError("initialization provenance includes evaluation patients")
        if {r["source"] for r in records if r["split"] == "external_test"} & set(initialization["training_sources"]):
            raise ValueError("initialization used a locked external source")
        if {r.get("content_hash") for r in records if r["split"] in {"val", "test", "external_test"}} & set(initialization["training_content_hashes"]):
            raise ValueError("initialization used evaluation image content")
        initialization["checkpoint_sha256"] = sha256_file(initialize_field or initialize_support)
        if initialize_field:
            model.load_state_dict(initial_checkpoint["model"], strict=True)
        else:
            old = torch.load(initialize_support, map_location="cpu", weights_only=False)
            model.backbone.load_state_dict(old.get("state_dict", old), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    start, global_step, best = 0, 0, float("inf")
    if resume:
        saved = load_checkpoint(resume, device)
        if _resume_contract(config) != _resume_contract(TrainingConfig(**saved["config"])):
            raise ValueError("resume configuration changed; initialize a new experiment instead")
        if fingerprint != saved["manifest_fingerprint"]:
            raise ValueError("resume manifest or payload changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_scaler(scaler, saved["scaler"])
        restore_rng(saved["rng"])
        start, global_step, best = saved["epoch"] + 1, saved["global_step"], saved["best_validation_loss"]
        initialization = saved.get("initialization")
    initial_provenance = initialization or {}
    train_patients = sorted({r["patient_id"] for r in train_rows + pseudo_rows} | set(initial_provenance.get("training_patient_ids", [])))
    train_sources = sorted({r["source"] for r in train_rows + pseudo_rows} | set(initial_provenance.get("training_sources", [])))
    train_hashes = sorted({r["content_hash"] for r in train_rows + pseudo_rows if r.get("content_hash")} | set(initial_provenance.get("training_content_hashes", [])))
    metadata = runtime_metadata()
    write_json(output_dir / "config.json", config.as_dict())
    write_json(output_dir / "provenance.json", {**metadata, "manifest_fingerprint": fingerprint,
               "train_patient_ids": train_patients, "training_sources": train_sources,
               "training_content_hashes": train_hashes, "initialization": initialization,
               "excluded_patient_ids": sorted({r["patient_id"] for r in records if r["split"] in {"val","test","external_test"}}),
               "pseudo_cases": len({(r["source"], r["case_id"]) for r in pseudo_rows}),
               "pseudo_records": len(pseudo_rows), "validation_criterion": config.validation_metric})
    # Every epoch uses an explicit seed; resume reproduces query draws and order.
    for epoch in range(start, config.epochs):
        rng = np.random.default_rng(config.seed + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        order = rng.permutation(len(train_rows))
        epoch_values = []
        component_values = {}
        skipped_steps = 0
        for position, index in enumerate(order):
            row = train_rows[int(index)]
            loss, metrics = case_loss(model, load_case(row), row, config, device, rng, epoch, training=True)
            if pseudo_rows and epoch >= config.pseudo_warmup_epochs:
                pseudo_row = pseudo_rows[int(rng.integers(len(pseudo_rows)))]
                pseudo_loss, _ = case_loss(model, load_case(pseudo_row), pseudo_row, config, device, rng, epoch, training=True)
                loss = loss + config.pseudo_weight * pseudo_loss
                metrics["pseudo"] = float(pseudo_loss.detach())
            for name, value in metrics.items():
                component_values.setdefault(name, []).append(value)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at {row['case_id']} epoch {epoch}")
            group_size = min(config.accumulation_steps, len(order) - (position // config.accumulation_steps) * config.accumulation_steps)
            scaler.scale(loss / group_size).backward()
            epoch_values.append(float(loss.detach()))
            if (position + 1) % config.accumulation_steps == 0 or position + 1 == len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=not scaler.is_enabled())
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() < previous_scale:
                    skipped_steps += 1
                else:
                    global_step += 1
        model.eval()
        patients = {}
        with torch.no_grad():
            for i, row in enumerate(val_rows):
                score = validation_score(model, load_case(row), row, config, device)
                patients.setdefault(row["patient_id"], []).append(score)
        validation = float(np.mean([np.mean(v) for v in patients.values()]))
        if not np.isfinite(validation):
            raise FloatingPointError("nonfinite validation loss")
        improved = validation < best
        best = min(validation, best)
        report = {"epoch": epoch, "global_step": global_step,
                  "train_loss": float(np.mean(epoch_values)), "validation_loss": validation,
                  "best_validation_loss": best, "validation_metric": config.validation_metric,
                  "amp_skipped_steps": skipped_steps,
                  "loss_components": {name: float(np.mean(values)) for name, values in component_values.items()}}
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report, allow_nan=False) + "\n")
        saved = {"format": "rtr-journal-field-v1", "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "rng": rng_state(), "config": config.as_dict(), "epoch": epoch,
                 "global_step": global_step, "best_validation_loss": best,
                 "manifest_fingerprint": fingerprint, "runtime": metadata,
                 "training_patient_ids": train_patients, "training_sources": train_sources,
                 "training_content_hashes": train_hashes, "initialization": initialization,
                 "excluded_patient_ids": sorted({r["patient_id"] for r in records if r["split"] in {"val","test","external_test"}})}
        atomic_checkpoint(saved, output_dir / "last.pt")
        if improved:
            atomic_checkpoint(saved, output_dir / "best.pt")
        print(json.dumps(report), flush=True)
    return {"epochs_completed": max(start, config.epochs), "global_step": global_step,
            "checkpoint": str(output_dir / "last.pt")}


def predict(*args, **kwargs):
    from .inference import predict as run
    return run(*args, **kwargs)
