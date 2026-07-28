from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .crown_localizer import fixed_world_grid, resample_hu, save_registration_labels
from .crown_network import (
    CrownLocalizerUNet,
    apply_inplane_transform,
    inplane_tta_transforms,
    invert_inplane_transform,
    normalize_hu,
    probabilities_to_labels,
)


@dataclass(frozen=True)
class CrownJawPostprocessConfig:
    minimum_probability: float | None = None
    minimum_component_voxels: int | None = None
    maximum_components: int | None = None
    minimum_hu: float | None = None


@dataclass(frozen=True)
class CrownPostprocessConfig:
    minimum_probability: float = 0.40
    minimum_component_voxels: int = 4
    maximum_components: int = 0
    minimum_hu: float = -1000.0
    grid_size: int = 128
    spacing_mm: float = 1.25
    upper: CrownJawPostprocessConfig | None = None
    lower: CrownJawPostprocessConfig | None = None

    @classmethod
    def load(cls, path: Path | None) -> "CrownPostprocessConfig":
        if path is None or not path.is_file():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "best_geometry" in payload:
            payload = payload["best_geometry"]
        aliases = {
            "threshold": "minimum_probability",
            "minimum_component_voxels": "minimum_component_voxels",
            "maximum_components": "maximum_components",
            "minimum_hu": "minimum_hu",
            "grid_size": "grid_size",
            "spacing_mm": "spacing_mm",
        }
        values = {
            destination: payload[source]
            for source, destination in aliases.items()
            if source in payload
        }
        jaws = payload.get("jaws", {})
        # Sweep summaries use ``jaws`` as an evaluated-jaw count. Deployment
        # configs use the same key for optional per-jaw overrides.
        if isinstance(jaws, (int, float)):
            jaws = {}
        if not isinstance(jaws, dict):
            raise ValueError("Crown postprocess 'jaws' must be an object")

        def jaw_config(name: str) -> CrownJawPostprocessConfig | None:
            jaw_payload = jaws.get(name)
            if jaw_payload is None:
                return None
            if not isinstance(jaw_payload, dict):
                raise ValueError(f"Crown postprocess jaw {name} must be an object")
            jaw_values = {
                destination: jaw_payload[source]
                for source, destination in aliases.items()
                if source in jaw_payload
                and destination
                in {
                    "minimum_probability",
                    "minimum_component_voxels",
                    "maximum_components",
                    "minimum_hu",
                }
            }
            return CrownJawPostprocessConfig(**jaw_values)

        values["upper"] = jaw_config("upper")
        values["lower"] = jaw_config("lower")
        return cls(**values)

    def resolved_jaw(self, jaw: str) -> CrownJawPostprocessConfig:
        override = self.upper if jaw == "upper" else self.lower if jaw == "lower" else None
        if jaw not in {"upper", "lower"}:
            raise ValueError(f"Unknown jaw: {jaw}")
        return CrownJawPostprocessConfig(
            minimum_probability=(
                self.minimum_probability
                if override is None or override.minimum_probability is None
                else override.minimum_probability
            ),
            minimum_component_voxels=(
                self.minimum_component_voxels
                if override is None or override.minimum_component_voxels is None
                else override.minimum_component_voxels
            ),
            maximum_components=(
                self.maximum_components
                if override is None or override.maximum_components is None
                else override.maximum_components
            ),
            minimum_hu=(
                self.minimum_hu
                if override is None or override.minimum_hu is None
                else override.minimum_hu
            ),
        )

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class CrownLocalizerEnsemble:
    """Persistent, optionally multi-branch crown localizer for submission inference."""

    def __init__(
        self,
        model_dir: Path,
        config: CrownPostprocessConfig | None = None,
        device: str = "cuda",
        fold_indices: tuple[int, ...] | None = None,
        tta_mode: str = "none",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.config = config or CrownPostprocessConfig()
        selected_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        if selected_device == "cuda":
            capability = torch.cuda.get_device_capability()
            requested_arch = f"sm_{capability[0]}{capability[1]}"
            compiled_arches = set(torch.cuda.get_arch_list())
            if compiled_arches and requested_arch not in compiled_arches:
                selected_device = "cpu"
                print(
                    "Crown localizer falling back to CPU because this PyTorch build "
                    f"does not include {requested_arch}",
                    flush=True,
                )
        self.device = torch.device(selected_device)
        inplane_tta_transforms(tta_mode)
        self.tta_mode = tta_mode
        branch_config = self.model_dir / "ensemble.json"
        if branch_config.is_file():
            payload = json.loads(branch_config.read_text(encoding="utf-8"))
            branches = payload.get("branches")
            if not isinstance(branches, list) or not branches:
                raise ValueError(f"No branches configured in {branch_config}")
            blend_mode = str(payload.get("mode", "arithmetic"))
        else:
            branches = [{"name": "default", "path": ".", "weight": 1.0}]
            blend_mode = "arithmetic"
        if blend_mode not in {"arithmetic", "geometric"}:
            raise ValueError(f"Unknown crown ensemble blend mode: {blend_mode}")

        requested_folds = set(fold_indices) if fold_indices is not None else None
        model_paths: list[Path] = []
        model_weights: list[float] = []
        branch_metadata: list[dict[str, object]] = []
        branch_model_indices: list[tuple[int, ...]] = []
        branch_weights: list[float] = []
        root = self.model_dir.resolve()
        raw_branch_weights = []
        raw_branch_class_weights = []
        parsed_branches = []
        for index, branch in enumerate(branches):
            if not isinstance(branch, dict):
                raise ValueError(f"Invalid branch {index} in {branch_config}")
            name = str(branch.get("name", f"branch_{index}"))
            relative_path = Path(str(branch.get("path", ".")))
            branch_dir = (self.model_dir / relative_path).resolve()
            if branch_dir != root and root not in branch_dir.parents:
                raise ValueError(f"Crown branch escapes model directory: {relative_path}")
            weight = float(branch.get("weight", 1.0))
            if not np.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"Invalid weight for crown branch {name}: {weight}")
            available = sorted(
                int(path.name.split("_", 1)[1])
                for path in branch_dir.glob("fold_*")
                if (path / "best.pt").is_file()
            )
            branch_folds = branch.get("fold_indices")
            if branch_folds is not None:
                configured = {int(fold) for fold in branch_folds}
                available = [fold for fold in available if fold in configured]
            if requested_folds is not None:
                available = [fold for fold in available if fold in requested_folds]
            if not available:
                raise FileNotFoundError(
                    f"No crown-localizer fold checkpoints for branch {name} under "
                    f"{branch_dir}"
                )
            parsed_branches.append((name, branch_dir, weight, tuple(available)))
            raw_branch_weights.append(weight)
            configured_weights = branch.get("weights")
            if configured_weights is None:
                class_weights = (weight, weight, weight)
            else:
                if not isinstance(configured_weights, dict):
                    raise ValueError(f"Class weights for branch {name} must be an object")
                class_weights = tuple(
                    float(configured_weights.get(class_name, weight))
                    for class_name in ("background", "upper", "lower")
                )
                if any(not np.isfinite(value) or value <= 0.0 for value in class_weights):
                    raise ValueError(
                        f"Invalid class weights for crown branch {name}: {class_weights}"
                    )
            raw_branch_class_weights.append(class_weights)

        weight_sum = float(sum(raw_branch_weights))
        class_weight_array = np.asarray(raw_branch_class_weights, dtype=np.float64)
        class_weight_array /= np.sum(class_weight_array, axis=0, keepdims=True)
        for branch_index, (name, branch_dir, raw_weight, available) in enumerate(parsed_branches):
            branch_weight = raw_weight / weight_sum
            per_model_weight = branch_weight / len(available)
            first_model = len(model_paths)
            for fold in available:
                model_paths.append(branch_dir / f"fold_{fold}" / "best.pt")
                model_weights.append(per_model_weight)
            branch_model_indices.append(
                tuple(range(first_model, first_model + len(available)))
            )
            branch_weights.append(branch_weight)
            branch_metadata.append(
                {
                    "name": name,
                    "path": str(branch_dir),
                    "weight": branch_weight,
                    "class_weights": class_weight_array[branch_index].tolist(),
                    "fold_indices": available,
                }
            )

        self.branch_metadata = tuple(branch_metadata)
        self.branch_model_indices = tuple(branch_model_indices)
        self.branch_weights = np.asarray(branch_weights, dtype=np.float64)
        self.branch_class_weights = class_weight_array
        self.class_specific_weights = not np.allclose(
            class_weight_array,
            self.branch_weights[:, None],
        )
        self.blend_mode = blend_mode
        self.model_weights = np.asarray(model_weights, dtype=np.float64)
        self.model_paths = tuple(model_paths)
        self.fold_indices = tuple(
            int(path.parent.name.split("_", 1)[1]) for path in self.model_paths
        )
        self.models = [self._load_model(path) for path in self.model_paths]

    def _load_model(self, checkpoint_path: Path) -> CrownLocalizerUNet:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        model = CrownLocalizerUNet(base_channels=int(checkpoint["base_channels"]))
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(self.device).eval()

    def predict_image(
        self, image: nib.spatialimages.SpatialImage
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
        shape = (int(self.config.grid_size),) * 3
        affine = fixed_world_grid(image, shape, float(self.config.spacing_mm))
        image_hu = resample_hu(image, shape, affine)
        inputs = torch.from_numpy(normalize_hu(image_hu)[None, None]).to(self.device)
        fold_probabilities = []
        for model in self.models:
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda",
            ):
                view_probabilities = []
                for rotation, flip in inplane_tta_transforms(self.tta_mode):
                    transformed = apply_inplane_transform(inputs, rotation, flip)
                    probabilities = torch.softmax(model(transformed), dim=1)
                    view_probabilities.append(
                        invert_inplane_transform(probabilities, rotation, flip)
                    )
                fold_probabilities.append(
                    torch.stack(view_probabilities).mean(dim=0)[0].float().cpu().numpy()
                )
        stack = np.stack(fold_probabilities)
        branch_probabilities = np.stack(
            [np.mean(stack[list(indices)], axis=0) for indices in self.branch_model_indices]
        )
        expanded_class_weights = self.branch_class_weights.reshape(
            self.branch_class_weights.shape
            + (1,) * (branch_probabilities.ndim - self.branch_class_weights.ndim)
        )
        if self.blend_mode == "arithmetic":
            probabilities = np.sum(
                branch_probabilities * expanded_class_weights,
                axis=0,
            )
            if self.class_specific_weights:
                epsilon = np.finfo(np.float32).tiny
                probabilities /= np.maximum(
                    np.sum(probabilities, axis=0, keepdims=True), epsilon
                )
        else:
            epsilon = np.finfo(np.float32).tiny
            log_probabilities = np.sum(
                expanded_class_weights
                * np.log(np.clip(branch_probabilities, epsilon, 1.0)),
                axis=0,
            )
            log_probabilities -= np.max(log_probabilities, axis=0, keepdims=True)
            probabilities = np.exp(log_probabilities)
            probabilities /= np.maximum(
                np.sum(probabilities, axis=0, keepdims=True), epsilon
            )
        upper_config = self.config.resolved_jaw("upper")
        lower_config = self.config.resolved_jaw("lower")
        labels = probabilities_to_labels(
            probabilities,
            minimum_probability=(
                float(upper_config.minimum_probability),
                float(lower_config.minimum_probability),
            ),
            minimum_component_voxels=(
                int(upper_config.minimum_component_voxels),
                int(lower_config.minimum_component_voxels),
            ),
            maximum_components=(
                int(upper_config.maximum_components),
                int(lower_config.maximum_components),
            ),
            image_hu=image_hu,
            minimum_hu=(
                float(upper_config.minimum_hu),
                float(lower_config.minimum_hu),
            ),
        )
        foreground = labels > 0
        if np.any(foreground) and len(stack) > 1:
            differences = stack[:, 1:] - probabilities[None, 1:]
            variance = np.tensordot(
                self.model_weights, differences * differences, axes=(0, 0)
            )
            disagreement = float(np.mean(np.sqrt(variance)[:, foreground]))
        else:
            disagreement = 0.0
        statistics: dict[str, float | int] = {
            "folds": len(self.models),
            "branches": len(self.branch_metadata),
            "tta_views": len(inplane_tta_transforms(self.tta_mode)),
            "upper_voxels": int(np.sum(labels == 1)),
            "lower_voxels": int(np.sum(labels == 2)),
            "ensemble_disagreement": disagreement,
        }
        return labels, probabilities, affine, statistics

    def predict_path(
        self,
        image_path: Path,
        output_path: Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
        result = self.predict_image(nib.load(str(image_path)))
        if output_path is not None:
            save_registration_labels(result[0], result[2], output_path)
        return result
