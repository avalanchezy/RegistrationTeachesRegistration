import torch

from task2reg.crown_network import (
    apply_inplane_transform,
    inplane_tta_transforms,
    invert_inplane_transform,
)


def test_d4_transforms_round_trip() -> None:
    tensor = torch.arange(2 * 3 * 5 * 7 * 4).reshape(2, 3, 5, 7, 4)
    transforms = inplane_tta_transforms("d4")
    assert len(transforms) == 8
    for rotation, flip in transforms:
        transformed = apply_inplane_transform(tensor, rotation, flip)
        restored = invert_inplane_transform(transformed, rotation, flip)
        assert torch.equal(restored, tensor)


def test_none_transform_is_identity() -> None:
    assert inplane_tta_transforms("none") == ((0, False),)
