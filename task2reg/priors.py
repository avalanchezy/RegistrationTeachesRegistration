from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .data import CaseRecord


def proper_protocol_rotation(linear: np.ndarray) -> np.ndarray:
    """Remove the STS export reflection while preserving its known axis."""
    linear = np.asarray(linear, dtype=np.float64)
    reflection = np.diag((-1.0, 1.0, 1.0)) if np.linalg.det(linear) < 0 else np.eye(3)
    return linear @ reflection


@dataclass(frozen=True)
class RotationPrior:
    mean_rotation: np.ndarray
    training_angles_deg: np.ndarray

    def angle_deg(self, linear: np.ndarray) -> float:
        candidate = proper_protocol_rotation(linear)
        relative = self.mean_rotation.T @ candidate
        return float(np.degrees(Rotation.from_matrix(relative).magnitude()))

    def centered_initialization(self, source: np.ndarray, target: np.ndarray, chirality: int) -> np.ndarray:
        if chirality not in (-1, 1):
            raise ValueError("chirality must be -1 or +1")
        reflection = np.diag((-1.0, 1.0, 1.0)) if chirality < 0 else np.eye(3)
        linear = self.mean_rotation @ reflection
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = linear
        transform[:3, 3] = target.mean(axis=0) - linear @ source.mean(axis=0)
        return transform

    def json_dict(self) -> dict:
        quantiles = np.quantile(self.training_angles_deg, (0.5, 0.9, 0.95, 1.0))
        return {
            "mean_rotation": self.mean_rotation.tolist(),
            "training_count": int(len(self.training_angles_deg)),
            "angle_deg_median": float(quantiles[0]),
            "angle_deg_p90": float(quantiles[1]),
            "angle_deg_p95": float(quantiles[2]),
            "angle_deg_max": float(quantiles[3]),
        }


def fit_rotation_prior(records: list[CaseRecord], jaw: str, excluded_cases: set[str] | None = None) -> RotationPrior:
    excluded_cases = excluded_cases or set()
    rotations = []
    for record in records:
        if record.split != "Train-Labeled" or record.jaw != jaw or not record.transform_path:
            continue
        if record.case_id in excluded_cases:
            continue
        transform = np.load(record.transform_path)
        rotations.append(proper_protocol_rotation(transform[:3, :3]))
    if len(rotations) < 3:
        raise ValueError(f"Need at least three labeled {jaw} transforms to fit a rotation prior")
    stacked = np.stack(rotations)
    mean = Rotation.from_matrix(stacked).mean().as_matrix()
    relative = np.stack([mean.T @ rotation for rotation in stacked])
    angles = np.degrees(Rotation.from_matrix(relative).magnitude())
    return RotationPrior(mean, angles)


def save_rotation_priors(priors: dict[str, RotationPrior], path: Path) -> None:
    payload = {jaw: prior.json_dict() for jaw, prior in priors.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_rotation_priors(path: Path) -> dict[str, RotationPrior]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    priors = {}
    for jaw, values in payload.items():
        # Stored quantiles are audit metadata; only the learned mean is needed at inference.
        priors[jaw] = RotationPrior(
            mean_rotation=np.asarray(values["mean_rotation"], dtype=np.float64),
            training_angles_deg=np.empty(0, dtype=np.float64),
        )
    return priors
