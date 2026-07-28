import json

import nibabel as nib
import numpy as np
import torch
from torch import nn

from task2reg.crown_inference import CrownLocalizerEnsemble, CrownPostprocessConfig
from task2reg.crown_network import CrownLocalizerUNet


class ConstantProbabilities(nn.Module):
    def __init__(self, probabilities: tuple[float, float, float]) -> None:
        super().__init__()
        self.register_buffer(
            "logits",
            torch.log(torch.tensor(probabilities, dtype=torch.float32)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shape = (inputs.shape[0], 3, *inputs.shape[2:])
        return self.logits.view(1, 3, 1, 1, 1).expand(shape)


def _write_checkpoint(path) -> None:
    path.parent.mkdir(parents=True)
    model = CrownLocalizerUNet(base_channels=1)
    torch.save({"state_dict": model.state_dict(), "base_channels": 1}, path)


def test_weighted_branches_blend_probabilities(tmp_path) -> None:
    _write_checkpoint(tmp_path / "supervised" / "fold_0" / "best.pt")
    _write_checkpoint(tmp_path / "semisupervised" / "fold_0" / "best.pt")
    (tmp_path / "ensemble.json").write_text(
        json.dumps(
            {
                "branches": [
                    {"name": "supervised", "path": "supervised", "weight": 0.25},
                    {
                        "name": "semisupervised",
                        "path": "semisupervised",
                        "weight": 0.75,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    ensemble = CrownLocalizerEnsemble(
        tmp_path,
        CrownPostprocessConfig(
            minimum_probability=0.20,
            minimum_component_voxels=1,
            maximum_components=0,
            minimum_hu=-1000.0,
            grid_size=16,
            spacing_mm=1.0,
        ),
        device="cpu",
    )
    ensemble.models = [
        ConstantProbabilities((0.1, 0.8, 0.1)),
        ConstantProbabilities((0.1, 0.1, 0.8)),
    ]
    image = nib.Nifti1Image(np.zeros((16, 16, 16), dtype=np.float32), np.eye(4))

    labels, probabilities, _, statistics = ensemble.predict_image(image)

    np.testing.assert_allclose(
        probabilities[:, 8, 8, 8], (0.1, 0.275, 0.625), atol=1e-6
    )
    assert np.all(labels == 2)
    assert statistics["branches"] == 2
    assert statistics["folds"] == 2


def test_single_directory_keeps_uniform_fold_weights(tmp_path) -> None:
    _write_checkpoint(tmp_path / "fold_0" / "best.pt")
    _write_checkpoint(tmp_path / "fold_1" / "best.pt")

    ensemble = CrownLocalizerEnsemble(tmp_path, device="cpu")

    np.testing.assert_allclose(ensemble.model_weights, (0.5, 0.5))
    assert len(ensemble.branch_metadata) == 1


def test_geometric_branches_combine_branch_means(tmp_path) -> None:
    for branch in ("first", "second"):
        _write_checkpoint(tmp_path / branch / "fold_0" / "best.pt")
        _write_checkpoint(tmp_path / branch / "fold_1" / "best.pt")
    (tmp_path / "ensemble.json").write_text(
        json.dumps(
            {
                "mode": "geometric",
                "branches": [
                    {"name": "first", "path": "first", "weight": 0.5},
                    {"name": "second", "path": "second", "weight": 0.5},
                ],
            }
        ),
        encoding="utf-8",
    )
    ensemble = CrownLocalizerEnsemble(
        tmp_path,
        CrownPostprocessConfig(
            minimum_probability=0.10,
            minimum_component_voxels=1,
            maximum_components=0,
            minimum_hu=-1000.0,
            grid_size=16,
            spacing_mm=1.0,
        ),
        device="cpu",
    )
    ensemble.models = [
        ConstantProbabilities((0.8, 0.1, 0.1)),
        ConstantProbabilities((0.6, 0.2, 0.2)),
        ConstantProbabilities((0.1, 0.8, 0.1)),
        ConstantProbabilities((0.2, 0.6, 0.2)),
    ]
    image = nib.Nifti1Image(np.zeros((16, 16, 16), dtype=np.float32), np.eye(4))

    _, probabilities, _, _ = ensemble.predict_image(image)

    first_mean = np.asarray((0.7, 0.15, 0.15))
    second_mean = np.asarray((0.15, 0.7, 0.15))
    expected = np.sqrt(first_mean * second_mean)
    expected /= expected.sum()
    np.testing.assert_allclose(probabilities[:, 8, 8, 8], expected, atol=1e-6)


def test_class_specific_branch_weights_are_normalized(tmp_path) -> None:
    _write_checkpoint(tmp_path / "first" / "fold_0" / "best.pt")
    _write_checkpoint(tmp_path / "second" / "fold_0" / "best.pt")
    (tmp_path / "ensemble.json").write_text(
        json.dumps(
            {
                "branches": [
                    {
                        "name": "first",
                        "path": "first",
                        "weight": 0.5,
                        "weights": {"background": 0.5, "upper": 0.75, "lower": 0.25},
                    },
                    {
                        "name": "second",
                        "path": "second",
                        "weight": 0.5,
                        "weights": {"background": 0.5, "upper": 0.25, "lower": 0.75},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    ensemble = CrownLocalizerEnsemble(
        tmp_path,
        CrownPostprocessConfig(
            minimum_probability=0.10,
            minimum_component_voxels=1,
            maximum_components=0,
            minimum_hu=-1000.0,
            grid_size=16,
            spacing_mm=1.0,
        ),
        device="cpu",
    )
    ensemble.models = [
        ConstantProbabilities((0.1, 0.8, 0.1)),
        ConstantProbabilities((0.1, 0.1, 0.8)),
    ]
    image = nib.Nifti1Image(np.zeros((16, 16, 16), dtype=np.float32), np.eye(4))

    _, probabilities, _, _ = ensemble.predict_image(image)

    expected = np.asarray((0.1, 0.625, 0.625))
    expected /= expected.sum()
    np.testing.assert_allclose(probabilities[:, 8, 8, 8], expected, atol=1e-6)
