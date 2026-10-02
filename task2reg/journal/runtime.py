"""Reproducibility and checkpoint helpers; artifacts remain local to a run."""
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess

import numpy as np
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def manifest_fingerprint(records):
    """Paths can relocate; actual payloads and experimental roles cannot change."""
    rows = []
    for row in records:
        clean = {k: v for k, v in row.items() if k != "npz_path" and not k.startswith("_")}
        clean["payload_sha256"] = sha256_file(row["npz_path"])
        rows.append(clean)
    return canonical_hash(rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_checkpoint(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def runtime_metadata():
    root = Path(__file__).resolve().parents[2]
    repository = root
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
            "git_revision": revision,
            "source_sha256": {str(p.relative_to(repository)): sha256_file(p)
                              for p in sorted((root / "task2reg").rglob("*.py"))}}


def restore_scaler(scaler, saved_state):
    # CPU/non-AMP runs save {}; moving to CUDA starts a new loss-scale history.
    if scaler.is_enabled() and saved_state:
        scaler.load_state_dict(saved_state)


def load_checkpoint(path, device="cpu"):
    # Checkpoints include optimizer/RNG Python objects. Load only trusted local artifacts.
    value = torch.load(path, map_location=device, weights_only=False)
    if value.get("format") != "rtr-journal-field-v1":
        raise ValueError("not a journal registration-field checkpoint")
    return value


def float_context(context):
    if isinstance(context, torch.Tensor):
        return context.float()
    if isinstance(context, dict):
        return {k: float_context(v) for k, v in context.items()}
    if isinstance(context, (tuple, list)):
        return type(context)(float_context(v) for v in context)
    return context
