import nibabel as nib
import numpy as np

from task2reg.crown_inference import CrownPostprocessConfig
from task2reg.crown_localizer import (
    build_crown_labels,
    fixed_world_grid,
    labels_to_registration_ids,
    world_to_grid,
)
from task2reg.crown_network import (
    CrownLocalizerUNet,
    crown_localizer_loss,
    probabilities_to_labels,
)
from task2reg.surfaces import (
    crown_guided_cbct_surface_candidates,
    crown_probability_surface_candidates,
)
from scripts.train_crown_localizer import CrownDataset, PseudoWeightedEpochSampler


def test_fixed_grid_is_centered_in_world_coordinates():
    affine = np.eye(4)
    affine[:3, 3] = (10.0, -4.0, 7.0)
    image = nib.Nifti1Image(np.zeros((20, 30, 40), dtype=np.int16), affine)
    grid = fixed_world_grid(image, (16, 16, 16), 2.0)
    source_center = affine[:3, 3] + 0.5 * (np.asarray(image.shape) - 1.0)
    grid_center = grid[:3, 3] + 0.5 * 15.0 * 2.0
    np.testing.assert_allclose(grid_center, source_center)


def test_crown_rasterization_keeps_jaws_separate():
    shape = (32, 32, 32)
    image = np.full(shape, 1000.0, dtype=np.float32)
    affine = np.eye(4)
    upper = np.asarray([[10.0, 10.0, 9.0], [11.0, 10.0, 9.0]])
    lower = np.asarray([[20.0, 20.0, 23.0], [21.0, 20.0, 23.0]])
    labels, coverage = build_crown_labels(
        image,
        {"upper": upper, "lower": lower},
        affine,
        spacing_mm=1.0,
        radius_mm=2.0,
    )
    assert coverage == {"upper": 1.0, "lower": 1.0}
    assert labels[10, 10, 9] == 1
    assert labels[20, 20, 23] == 2
    assert set(np.unique(labels_to_registration_ids(labels))) == {0, 1, 17}
    np.testing.assert_allclose(world_to_grid(upper, affine), upper)


def test_crown_network_shapes_and_probability_postprocessing():
    import torch

    model = CrownLocalizerUNet(base_channels=4)
    logits, shapes = model(torch.zeros((1, 1, 32, 32, 32)), return_shapes=True)
    assert logits.shape == (1, 3, 32, 32, 32)
    assert shapes["bottleneck"] == (1, 64, 2, 2, 2)

    probabilities = np.zeros((3, 16, 16, 16), dtype=np.float32)
    probabilities[0] = 0.1
    probabilities[1, 2:6, 2:6, 2:6] = 0.8
    probabilities[2, 10:14, 10:14, 10:14] = 0.9
    labels = probabilities_to_labels(
        probabilities,
        minimum_probability=0.5,
        minimum_component_voxels=8,
        image_hu=np.full((16, 16, 16), 1000.0),
    )
    assert np.sum(labels == 1) == 64
    assert np.sum(labels == 2) == 64

    target = torch.zeros((1, 16, 16, 16), dtype=torch.long)
    target[:, 4:6, 4:6, 4:6] = 1
    tolerant_loss, parts = crown_localizer_loss(
        torch.zeros((1, 3, 16, 16, 16), requires_grad=True),
        target,
        surface_tolerance_voxels=2,
    )
    assert torch.isfinite(tolerant_loss)
    assert parts["surface_tolerance_voxels"] == 2.0


def test_crown_probability_candidates_preserve_affine_and_jaw(tmp_path):
    probabilities = np.full((3, 16, 16, 16), 0.05, dtype=np.float32)
    probabilities[0] = 0.10
    probabilities[1, 2:6, 3:7, 4:8] = 0.90
    probabilities[2, 10:14, 9:13, 8:12] = 0.85
    affine = np.eye(4)
    affine[:3, :3] *= 1.25
    path = tmp_path / "case.npz"
    np.savez_compressed(path, probabilities=probabilities, affine=affine)

    upper = crown_probability_surface_candidates(
        path,
        "upper",
        probability_thresholds=(0.5,),
        voxel_counts=(),
        minimum_component_voxels=8,
    )
    lower = crown_probability_surface_candidates(
        path,
        "lower",
        probability_thresholds=(0.5,),
        voxel_counts=(),
        minimum_component_voxels=8,
    )
    assert len(upper) == len(lower) == 1
    assert len(upper[0].points) == len(lower[0].points) == 64
    np.testing.assert_allclose(upper[0].points.min(axis=0), np.asarray([2, 3, 4]) * 1.25)


def test_crown_guidance_extracts_native_cbct_surface(tmp_path):
    affine = np.eye(4)
    volume = np.zeros((32, 32, 32), dtype=np.int16)
    volume[8:18, 8:18, 8:18] = 1200
    labels = np.zeros_like(volume, dtype=np.uint8)
    labels[10:16, 10:16, 10:16] = 1
    volume_path = tmp_path / "volume.nii.gz"
    labels_path = tmp_path / "labels.nii.gz"
    nib.save(nib.Nifti1Image(volume, affine), volume_path)
    nib.save(nib.Nifti1Image(labels, affine), labels_path)

    candidates = crown_guided_cbct_surface_candidates(
        volume_path,
        labels_path,
        "upper",
        thresholds=(500.0,),
        guidance_radii_mm=(4.0,),
    )
    assert len(candidates) == 1
    assert candidates[0].metadata["threshold"] == 500.0
    assert len(candidates[0].points) >= 64


def test_crown_dataset_reads_weighted_pseudo_labels(tmp_path):
    image_path = tmp_path / "unlabeled.npz"
    pseudo_path = tmp_path / "pseudo.npz"
    image = np.zeros((16, 16, 16), dtype=np.int16)
    labels = np.zeros_like(image, dtype=np.uint8)
    labels[4:8, 4:8, 4:8] = 1
    np.savez_compressed(image_path, image=image, affine=np.eye(4))
    np.savez_compressed(pseudo_path, label=labels, affine=np.eye(4))
    dataset = CrownDataset(
        [image_path],
        augment=False,
        cache=True,
        seed=1,
        label_overrides={image_path: pseudo_path},
        sample_weights={image_path: 0.2},
    )
    _, loaded_labels, _, weight = dataset[0]
    assert int(np.sum(loaded_labels.numpy() == 1)) == 64
    assert weight == 0.2


def test_pseudo_weighted_sampler_preserves_labeled_epoch_and_effective_mass():
    sampler = PseudoWeightedEpochSampler(
        labeled_count=24,
        pseudo_count=40,
        pseudo_weight=0.10,
        seed=7,
    )
    indices = list(sampler)
    assert len(sampler) == len(indices) == 28
    assert sorted(index for index in indices if index < 24) == list(range(24))
    assert sum(index >= 24 for index in indices) == 4
    assert len(set(index for index in indices if index >= 24)) == 4


def test_crown_postprocess_config_accepts_sweep_summary(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(
        '{"best_geometry":{"threshold":0.35,"minimum_component_voxels":12,'
        '"maximum_components":2,"minimum_hu":0}}',
        encoding="utf-8",
    )
    config = CrownPostprocessConfig.load(path)
    assert config.minimum_probability == 0.35
    assert config.minimum_component_voxels == 12
    assert config.maximum_components == 2
    assert config.minimum_hu == 0


def test_crown_postprocess_config_ignores_sweep_jaw_count(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(
        '{"best_geometry":{"threshold":0.25,"minimum_component_voxels":4,'
        '"maximum_components":0,"minimum_hu":-1000,"jaws":60}}',
        encoding="utf-8",
    )
    config = CrownPostprocessConfig.load(path)
    assert config.minimum_probability == 0.25
    assert config.upper is None
    assert config.lower is None


def test_crown_postprocess_config_accepts_jaw_specific_overrides(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(
        '{"threshold":0.4,"minimum_component_voxels":4,'
        '"maximum_components":0,"minimum_hu":-1000,'
        '"jaws":{"upper":{"threshold":0.5,"minimum_component_voxels":12},'
        '"lower":{"threshold":0.3}}}',
        encoding="utf-8",
    )
    config = CrownPostprocessConfig.load(path)
    upper = config.resolved_jaw("upper")
    lower = config.resolved_jaw("lower")
    assert upper.minimum_probability == 0.5
    assert upper.minimum_component_voxels == 12
    assert lower.minimum_probability == 0.3
    assert lower.minimum_component_voxels == 4


def test_probability_postprocessing_accepts_per_jaw_parameters():
    probabilities = np.zeros((3, 4, 4, 4), dtype=np.float32)
    probabilities[0] = 0.10
    probabilities[1, :2] = 0.45
    probabilities[2, 2:] = 0.45
    labels = probabilities_to_labels(
        probabilities,
        minimum_probability=(0.40, 0.50),
        minimum_component_voxels=(1, 1),
        maximum_components=(0, 0),
        minimum_hu=(-1000.0, -1000.0),
    )
    assert int(np.sum(labels == 1)) == 32
    assert int(np.sum(labels == 2)) == 0
