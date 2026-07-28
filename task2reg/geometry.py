from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class RegistrationResult:
    method: str
    chirality: int
    transform_initial: np.ndarray
    transform: np.ndarray
    score_initial: float
    score: float
    median_distance: float
    p90_distance: float
    overlap_2mm: float
    correspondences: int

    def json_dict(self) -> dict:
        out = asdict(self)
        out["transform_initial"] = self.transform_initial.tolist()
        out["transform"] = self.transform.tolist()
        return out


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _homogeneous(linear: np.ndarray, translation: np.ndarray | None = None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear
    if translation is not None:
        transform[:3, 3] = translation
    return transform


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return _homogeneous(rotation, target_center - rotation @ source_center)


def _basis(points: np.ndarray) -> np.ndarray:
    covariance = np.cov(points - points.mean(axis=0), rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return basis


def _signed_permutations() -> list[np.ndarray]:
    matrices = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = base @ np.diag(signs)
            if np.linalg.det(candidate) > 0:
                matrices.append(candidate)
    return matrices


SIGNED_PERMUTATIONS = _signed_permutations()


def fit_score(source_transformed: np.ndarray, target_tree: cKDTree, trim_fraction: float = 0.7) -> dict[str, float]:
    distances, _ = target_tree.query(source_transformed, workers=-1)
    keep = max(32, int(len(distances) * trim_fraction))
    trimmed = np.partition(distances, keep - 1)[:keep]
    return {
        "score": float(np.mean(trimmed)),
        "median": float(np.median(distances)),
        "p90": float(np.quantile(distances, 0.9)),
        "overlap_2mm": float(np.mean(distances <= 2.0)),
    }


def bidirectional_fit_score(
    source_transformed: np.ndarray,
    target: np.ndarray,
    source_trim_fraction: float = 0.7,
    target_trim_fraction: float = 0.35,
    target_weight: float = 0.5,
) -> dict[str, float]:
    target_tree = cKDTree(target)
    source_distances, _ = target_tree.query(source_transformed, workers=-1)
    source_keep = max(32, int(len(source_distances) * source_trim_fraction))
    source_trimmed = np.partition(source_distances, source_keep - 1)[:source_keep]
    source_tree = cKDTree(source_transformed)
    target_distances, _ = source_tree.query(target, workers=-1)
    target_keep = max(32, int(len(target_distances) * target_trim_fraction))
    target_trimmed = np.partition(target_distances, target_keep - 1)[:target_keep]
    source_score = float(np.mean(source_trimmed))
    target_score = float(np.mean(target_trimmed))
    combined = (source_score + target_weight * target_score) / (1.0 + target_weight)
    return {
        "score": float(combined),
        "source_score": source_score,
        "target_score": target_score,
        "target_coverage_1mm": float(np.mean(target_distances <= 1.0)),
        "target_coverage_2mm": float(np.mean(target_distances <= 2.0)),
    }


def pca_initializations(source: np.ndarray, target: np.ndarray, chirality: int) -> list[np.ndarray]:
    reflection = np.eye(3)
    if chirality < 0:
        reflection[0, 0] = -1.0
    reflected = source @ reflection.T
    source_basis = _basis(reflected)
    target_basis = _basis(target)
    source_center = reflected.mean(axis=0)
    target_center = target.mean(axis=0)
    reflection_h = _homogeneous(reflection)
    transforms = []
    for permutation in SIGNED_PERMUTATIONS:
        rotation = target_basis @ permutation @ source_basis.T
        if np.linalg.det(rotation) < 0:
            continue
        proper = _homogeneous(rotation, target_center - rotation @ source_center)
        transforms.append(proper @ reflection_h)
    return transforms


def robust_trimmed_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    distance_schedule: tuple[float, ...] = (8.0, 5.0, 3.0, 2.0, 1.25),
    iterations_per_level: int = 8,
    trim_fraction: float = 0.7,
) -> tuple[np.ndarray, int]:
    target_tree = cKDTree(target)
    transform = initial.copy()
    correspondence_count = 0
    for max_distance in distance_schedule:
        for _ in range(iterations_per_level):
            moved = transform_points(source, transform)
            distances, indices = target_tree.query(moved, workers=-1)
            valid = np.flatnonzero(distances <= max_distance)
            if len(valid) < 32:
                break
            keep = max(32, int(len(valid) * trim_fraction))
            selected = valid[np.argpartition(distances[valid], keep - 1)[:keep]]
            delta = _rigid_fit(moved[selected], target[indices[selected]])
            transform = delta @ transform
            correspondence_count = len(selected)
            motion = np.linalg.norm(delta[:3, 3]) + np.linalg.norm(delta[:3, :3] - np.eye(3))
            if motion < 1e-5:
                break
    return transform, correspondence_count


def stochastic_basin_refinement(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    samples_per_stage: int = 256,
    stages: tuple[tuple[float, float], ...] = ((12.0, 8.0), (6.0, 4.0), (3.0, 2.0)),
    refine_top_k: int = 8,
    seed: int = 2026,
) -> tuple[np.ndarray, int]:
    """Escape a local ICP basin with centered rigid perturbations.

    Rotations are applied around the currently transformed crown center, so a
    few degrees do not create a large artificial translation when IOS scanner
    coordinates are far from the world origin.
    """
    rng = np.random.default_rng(seed)
    target_tree = cKDTree(target)

    def score(transform: np.ndarray) -> float:
        return fit_score(transform_points(source, transform), target_tree)["score"]

    transform = initial.copy()
    correspondences = 0
    for rotation_sigma_deg, translation_sigma_mm in stages:
        center = transform_points(source, transform).mean(axis=0)
        pool = [(score(transform), transform)]
        for _ in range(samples_per_stage):
            rotation = Rotation.from_rotvec(
                rng.normal(size=3) * np.deg2rad(rotation_sigma_deg)
            ).as_matrix()
            delta = np.eye(4)
            delta[:3, :3] = rotation
            delta[:3, 3] = center - rotation @ center + rng.normal(size=3) * translation_sigma_mm
            candidate = delta @ transform
            pool.append((score(candidate), candidate))
        refined = []
        for _, candidate in sorted(pool, key=lambda item: item[0])[:refine_top_k]:
            candidate, count = robust_trimmed_icp(
                source,
                target,
                candidate,
                distance_schedule=(4.0, 2.0, 1.25),
                iterations_per_level=5,
            )
            refined.append((score(candidate), candidate, count))
        _, transform, correspondences = min(refined, key=lambda item: item[0])
    return transform, correspondences


def _open3d_feature_initialization(
    source: np.ndarray,
    target: np.ndarray,
    reflection: np.ndarray,
    method: str,
    voxel_size: float,
    ransac_iterations: int,
    seed: int,
) -> np.ndarray | None:
    import open3d as o3d

    o3d.utility.random.seed(seed)
    reflected = source @ reflection.T

    def prepare(points: np.ndarray):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        down = cloud.voxel_down_sample(voxel_size)
        if len(down.points) < 32:
            return None, None
        down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.5, max_nn=48))
        feature = o3d.pipelines.registration.compute_fpfh_feature(
            down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
        )
        return down, feature

    source_down, source_feature = prepare(reflected)
    target_down, target_feature = prepare(target)
    if source_down is None or target_down is None:
        return None
    threshold = voxel_size * 1.75
    if method == "fgr":
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down,
            target_down,
            source_feature,
            target_feature,
            o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=threshold),
        )
    else:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_feature,
            target_feature,
            True,
            threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_iterations, 0.999),
        )
    reflection_h = _homogeneous(reflection)
    return np.asarray(result.transformation) @ reflection_h


def register_geometry(
    source: np.ndarray,
    target: np.ndarray,
    methods: tuple[str, ...] = ("pca", "fgr", "ransac"),
    allow_reflection: bool = True,
    pca_refine_top_k: int = 4,
    voxel_size: float = 2.0,
    ransac_iterations: int = 50000,
    seed: int = 2026,
    chirality_hint: int | None = None,
) -> list[RegistrationResult]:
    target_tree = cKDTree(target)
    if chirality_hint is not None:
        if chirality_hint not in (-1, 1):
            raise ValueError("chirality_hint must be -1, +1, or None")
        chiralities = (chirality_hint,)
    else:
        chiralities = (1, -1) if allow_reflection else (1,)
    results: list[RegistrationResult] = []

    for chirality in chiralities:
        reflection = np.eye(3)
        if chirality < 0:
            reflection[0, 0] = -1.0
        initializations: list[tuple[str, np.ndarray]] = []
        if "pca" in methods:
            pca = pca_initializations(source, target, chirality)
            ranked = sorted(pca, key=lambda t: fit_score(transform_points(source, t), target_tree)["score"])
            initializations.extend(("pca", transform) for transform in ranked[:pca_refine_top_k])
        for method in ("fgr", "ransac"):
            if method not in methods:
                continue
            try:
                transform = _open3d_feature_initialization(
                    source, target, reflection, method, voxel_size, ransac_iterations, seed + len(results)
                )
            except RuntimeError:
                # Open3D raises when feature matching yields no usable correspondences.
                # This initialization is optional; other initializations remain valid.
                transform = None
            if transform is not None:
                initializations.append((method, transform))

        for method, initial in initializations:
            initial_metrics = fit_score(transform_points(source, initial), target_tree)
            refined, correspondences = robust_trimmed_icp(source, target, initial)
            metrics = fit_score(transform_points(source, refined), target_tree)
            results.append(
                RegistrationResult(
                    method=method,
                    chirality=chirality,
                    transform_initial=initial,
                    transform=refined,
                    score_initial=initial_metrics["score"],
                    score=metrics["score"],
                    median_distance=metrics["median"],
                    p90_distance=metrics["p90"],
                    overlap_2mm=metrics["overlap_2mm"],
                    correspondences=correspondences,
                )
            )
    return sorted(results, key=lambda result: result.score)
