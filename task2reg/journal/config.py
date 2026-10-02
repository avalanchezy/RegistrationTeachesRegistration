"""Explicit, serializable settings for journal experiments."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class TrainingConfig:
    mode: str = "implicit"
    energy_mode: str = "distance"
    base_channels: int = 24
    truncation_mm: float = 8.0
    epochs: int = 120
    queries: int = 16384
    candidate_points: int = 1024
    max_candidates: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    support_weight: float = 1.0
    field_weight: float = 1.0
    rank_weight: float = 0.25
    rank_warmup_epochs: int = 20
    rank_ramp_epochs: int = 10
    eikonal_weight: float = 0.0
    eikonal_queries: int = 1024
    pose_weight: float = 0.0
    pose_steps: int = 2
    refinement_learning_rate: float = 0.25
    secant_weight: float = 0.0
    secant_points: int = 512
    secant_cached_candidates: int = 2
    secant_reference_perturbations: int = 2
    secant_perturbation_mm: float = 4.0
    secant_step_mm: float = 0.5
    secant_smoothing_mm: float = 1.0
    secant_local_minimum_weight: float = 0.25
    ssl_strategy: str = "none"
    pseudo_weight: float = 0.25
    pseudo_warmup_epochs: int = 0
    max_pseudo_cases: int = 0
    accumulation_steps: int = 1
    gradient_clip: float = 5.0
    intensity_jitter_hu: float = 50.0
    validation_metric: str = "selected_D_mm"
    validation_refinement_steps: int = 20
    validation_point_budget: int = 4096
    validation_refinement_learning_rate: float = 0.25
    device: str = "cuda"
    amp: bool = True
    seed: int = 20261002

    def __post_init__(self):
        if self.mode not in {"implicit", "dense"}:
            raise ValueError("mode must be implicit or dense")
        if self.energy_mode not in {"distance", "task"}:
            raise ValueError("energy_mode must be distance or task")
        if self.ssl_strategy not in {"none", "legacy_support", "verified_field"}:
            raise ValueError("unknown ssl_strategy")
        if self.validation_metric not in {"selected_D_mm", "refined_selected_D_mm", "field_loss"}:
            raise ValueError("validation_metric must be selected_D_mm, refined_selected_D_mm or field_loss")
        for name in ("base_channels", "epochs", "queries", "candidate_points",
                     "max_candidates", "accumulation_steps", "eikonal_queries", "pose_steps", "validation_point_budget", "secant_points"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.base_channels % 4:
            raise ValueError("base_channels must be divisible by four for GroupNorm")
        for name in ("truncation_mm", "learning_rate", "gradient_clip", "refinement_learning_rate",
                     "validation_refinement_learning_rate", "secant_perturbation_mm", "secant_step_mm", "secant_smoothing_mm"):
            if not 0 < getattr(self, name) < float("inf"):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "support_weight", "field_weight", "rank_weight",
                     "eikonal_weight", "pose_weight", "pseudo_weight", "intensity_jitter_hu", "secant_weight", "secant_local_minimum_weight"):
            if not 0 <= getattr(self, name) < float("inf"):
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("rank_warmup_epochs", "rank_ramp_epochs", "pseudo_warmup_epochs", "max_pseudo_cases",
                     "validation_refinement_steps", "secant_cached_candidates", "secant_reference_perturbations"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.secant_weight and self.secant_cached_candidates + self.secant_reference_perturbations == 0:
            raise ValueError("secant supervision requires cached candidates or reference perturbations")
        if self.support_weight + self.field_weight + self.rank_weight + self.pose_weight + self.secant_weight <= 0:
            raise ValueError("at least one training loss must be enabled")

    @classmethod
    def from_json(cls, path: Path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def as_dict(self):
        return asdict(self)
