from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import label as connected_components
from torch import nn
from torch.nn import functional as functional


def inplane_tta_transforms(mode: str) -> tuple[tuple[int, bool], ...]:
    if mode == "none":
        return ((0, False),)
    if mode == "d4":
        return tuple((rotation, flip) for rotation in range(4) for flip in (False, True))
    raise ValueError(f"Unknown crown TTA mode: {mode}")


def apply_inplane_transform(
    tensor: torch.Tensor, rotation: int, flip: bool
) -> torch.Tensor:
    transformed = torch.rot90(tensor, int(rotation), dims=(-3, -2))
    return torch.flip(transformed, dims=(-3,)) if flip else transformed


def invert_inplane_transform(
    tensor: torch.Tensor, rotation: int, flip: bool
) -> torch.Tensor:
    restored = torch.flip(tensor, dims=(-3,)) if flip else tensor
    return torch.rot90(restored, -int(rotation), dims=(-3, -2))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(4, out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class CrownLocalizerUNet(nn.Module):
    def __init__(self, base_channels: int = 8, output_channels: int = 3) -> None:
        super().__init__()
        widths = [base_channels * (2**index) for index in range(5)]
        self.encoder1 = ConvBlock(1, widths[0])
        self.encoder2 = ConvBlock(widths[0], widths[1])
        self.encoder3 = ConvBlock(widths[1], widths[2])
        self.encoder4 = ConvBlock(widths[2], widths[3])
        self.bottleneck = ConvBlock(widths[3], widths[4])
        self.pool = nn.MaxPool3d(2)
        self.up4 = nn.ConvTranspose3d(widths[4], widths[3], 2, stride=2)
        self.decoder4 = ConvBlock(widths[3] * 2, widths[3])
        self.up3 = nn.ConvTranspose3d(widths[3], widths[2], 2, stride=2)
        self.decoder3 = ConvBlock(widths[2] * 2, widths[2])
        self.up2 = nn.ConvTranspose3d(widths[2], widths[1], 2, stride=2)
        self.decoder2 = ConvBlock(widths[1] * 2, widths[1])
        self.up1 = nn.ConvTranspose3d(widths[1], widths[0], 2, stride=2)
        self.decoder1 = ConvBlock(widths[0] * 2, widths[0])
        self.output = nn.Conv3d(widths[0], output_channels, kernel_size=1)

    @staticmethod
    def _match(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if inputs.shape[2:] == skip.shape[2:]:
            return inputs
        return functional.interpolate(
            inputs, size=skip.shape[2:], mode="trilinear", align_corners=False
        )

    def forward(
        self, inputs: torch.Tensor, return_shapes: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        shapes: dict[str, tuple[int, ...]] = {"input": tuple(inputs.shape)}
        first = self.encoder1(inputs)
        shapes["encoder1"] = tuple(first.shape)
        second = self.encoder2(self.pool(first))
        shapes["encoder2"] = tuple(second.shape)
        third = self.encoder3(self.pool(second))
        shapes["encoder3"] = tuple(third.shape)
        fourth = self.encoder4(self.pool(third))
        shapes["encoder4"] = tuple(fourth.shape)
        center = self.bottleneck(self.pool(fourth))
        shapes["bottleneck"] = tuple(center.shape)
        decoded4 = self.decoder4(torch.cat((self._match(self.up4(center), fourth), fourth), dim=1))
        shapes["decoder4"] = tuple(decoded4.shape)
        decoded3 = self.decoder3(torch.cat((self._match(self.up3(decoded4), third), third), dim=1))
        shapes["decoder3"] = tuple(decoded3.shape)
        decoded2 = self.decoder2(torch.cat((self._match(self.up2(decoded3), second), second), dim=1))
        shapes["decoder2"] = tuple(decoded2.shape)
        decoded1 = self.decoder1(torch.cat((self._match(self.up1(decoded2), first), first), dim=1))
        shapes["decoder1"] = tuple(decoded1.shape)
        logits = self.output(decoded1)
        shapes["logits"] = tuple(logits.shape)
        return (logits, shapes) if return_shapes else logits


def normalize_hu(image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(image, torch.Tensor):
        return (torch.clamp(image, -500.0, 3000.0) - 500.0) / 1000.0
    return (np.clip(image, -500.0, 3000.0) - 500.0) / 1000.0


def crown_localizer_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    background_weight: float = 0.05,
    surface_tolerance_voxels: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = torch.tensor(
        [background_weight, 1.0, 1.0], device=logits.device, dtype=logits.dtype
    )
    one_hot = functional.one_hot(labels, num_classes=3).movedim(-1, 1).to(logits.dtype)
    if surface_tolerance_voxels > 0:
        foreground = one_hot[:, 1:]
        soft_foreground = foreground.clone()
        for radius in range(1, surface_tolerance_voxels + 1):
            dilated = functional.max_pool3d(
                foreground,
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            )
            soft_foreground = torch.maximum(
                soft_foreground, dilated * (0.5 ** radius)
            )
        soft_background = 1.0 - torch.amax(soft_foreground, dim=1, keepdim=True)
        soft_targets = torch.cat((soft_background, soft_foreground), dim=1)
        soft_targets = soft_targets / torch.clamp(
            torch.sum(soft_targets, dim=1, keepdim=True), min=1e-6
        )
        weighted_targets = soft_targets * weights.view(1, -1, 1, 1, 1)
        cross_entropy = -torch.sum(
            weighted_targets * functional.log_softmax(logits, dim=1)
        ) / torch.clamp(torch.sum(weighted_targets), min=1.0)
    else:
        cross_entropy = functional.cross_entropy(logits, labels, weight=weights)
    probabilities = torch.softmax(logits, dim=1)
    intersection = torch.sum(probabilities[:, 1:] * one_hot[:, 1:], dim=(0, 2, 3, 4))
    denominator = torch.sum(probabilities[:, 1:] + one_hot[:, 1:], dim=(0, 2, 3, 4))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = 1.0 - torch.mean(dice)
    loss = cross_entropy + dice_loss
    metrics = {
        "cross_entropy": float(cross_entropy.detach().cpu()),
        "dice_loss": float(dice_loss.detach().cpu()),
        "soft_dice_upper": float(dice[0].detach().cpu()),
        "soft_dice_lower": float(dice[1].detach().cpu()),
        "surface_tolerance_voxels": float(surface_tolerance_voxels),
    }
    return loss, metrics


def hard_dice(probabilities: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    prediction = torch.argmax(probabilities, dim=1)
    values = []
    for class_id in (1, 2):
        predicted = prediction == class_id
        target = labels == class_id
        intersection = torch.sum(predicted & target).float()
        denominator = torch.sum(predicted).float() + torch.sum(target).float()
        values.append(float(((2.0 * intersection + 1.0) / (denominator + 1.0)).cpu()))
    return values[0], values[1]


def remove_small_components(
    mask: np.ndarray,
    minimum_voxels: int,
    maximum_components: int = 0,
) -> np.ndarray:
    components, count = connected_components(mask)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(components.reshape(-1))
    keep = np.flatnonzero(sizes >= minimum_voxels)
    keep = keep[keep != 0]
    if maximum_components > 0 and len(keep) > maximum_components:
        order = np.argsort(sizes[keep])[::-1]
        keep = keep[order[:maximum_components]]
    return np.isin(components, keep)


def probabilities_to_labels(
    probabilities: np.ndarray,
    minimum_probability: float | tuple[float, float] = 0.70,
    minimum_component_voxels: int | tuple[int, int] = 12,
    maximum_components: int | tuple[int, int] = 1,
    image_hu: np.ndarray | None = None,
    minimum_hu: float | tuple[float, float] = 150.0,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape[0] != 3:
        raise ValueError(f"Expected 3 probability channels, got {probabilities.shape}")
    winner = np.argmax(probabilities, axis=0)
    def class_value(value, class_id: int):
        if np.isscalar(value):
            return value
        values = tuple(value)
        if len(values) != 2:
            raise ValueError(f"Expected upper/lower values, got {values}")
        return values[class_id - 1]

    masks = {}
    for class_id in (1, 2):
        class_minimum_hu = float(class_value(minimum_hu, class_id))
        intensity_mask = (
            np.ones(probabilities.shape[1:], dtype=bool)
            if image_hu is None
            else np.asarray(image_hu) >= class_minimum_hu
        )
        mask = (
            (winner == class_id)
            & (
                probabilities[class_id]
                >= float(class_value(minimum_probability, class_id))
            )
            & intensity_mask
        )
        masks[class_id] = remove_small_components(
            mask,
            int(class_value(minimum_component_voxels, class_id)),
            int(class_value(maximum_components, class_id)),
        )
    labels = np.zeros(probabilities.shape[1:], dtype=np.uint8)
    labels[masks[1]] = 1
    labels[masks[2]] = 2
    return labels


@dataclass(frozen=True)
class ModelConfiguration:
    base_channels: int = 8
    output_channels: int = 3
