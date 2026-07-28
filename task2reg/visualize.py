from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from .geometry import transform_points


def _sample(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(points) <= count:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=count, replace=False)]


def save_registration_figure(
    output: Path,
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray | None = None,
    title: str = "",
) -> None:
    target_plot = _sample(target, 6000, 1)
    source_plot = _sample(source, 6000, 2)
    coarse = transform_points(source_plot, initial)
    refined = transform_points(source_plot, prediction)
    truth = transform_points(source_plot, ground_truth) if ground_truth is not None else None
    panels = [
        (source_plot, "IOS source"),
        (target_plot, "CBCT target"),
        (coarse, "Coarse initialization"),
        (refined, "Robust refinement"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for column, (points, name) in enumerate(panels):
        for row, dims in enumerate(((0, 1), (0, 2))):
            ax = axes[row, column]
            if column >= 2:
                ax.scatter(target_plot[:, dims[0]], target_plot[:, dims[1]], s=0.3, c="#8b949e", alpha=0.22)
            ax.scatter(points[:, dims[0]], points[:, dims[1]], s=0.45, c="#d94841" if column >= 2 else "#2675bf", alpha=0.75)
            if column == 3 and truth is not None:
                ax.scatter(truth[:, dims[0]], truth[:, dims[1]], s=0.35, c="#2b9a66", alpha=0.45)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{name} ({'XY' if row == 0 else 'XZ'})", fontsize=10)
            ax.set_xlabel("mm")
            ax.set_ylabel("mm")
    fig.suptitle(title, fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_overlay_ply(output: Path, target: np.ndarray, prediction: np.ndarray, truth: np.ndarray | None = None) -> None:
    point_sets = [target, prediction]
    colors = [np.array([150, 156, 166]), np.array([220, 72, 65])]
    if truth is not None:
        point_sets.append(truth)
        colors.append(np.array([43, 154, 102]))
    points = np.concatenate(point_sets, axis=0)
    color_array = np.concatenate([np.tile(color, (len(group), 1)) for group, color in zip(point_sets, colors)], axis=0)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors = o3d.utility.Vector3dVector(color_array / 255.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output), cloud, write_ascii=False)
