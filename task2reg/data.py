from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class CaseRecord:
    split: str
    case_id: str
    jaw: str
    cbct_path: str
    ios_path: str
    transform_path: str = ""
    complete: bool = True

    @property
    def key(self) -> str:
        return f"{self.split}:{self.case_id}:{self.jaw}"


def _first_existing(folder: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = folder / name
        if path.exists():
            return path.resolve()
    return None


def _split_dir(root: Path, split: str) -> tuple[Path, Path | None]:
    base = root / split
    image_dir = _first_existing(base, ("images", "Images"))
    if image_dir is None:
        raise FileNotFoundError(f"Missing images directory under {base}")
    label_dir = _first_existing(base, ("labels", "Labels"))
    return image_dir, label_dir


def build_manifest(root: Path) -> list[CaseRecord]:
    root = root.resolve()
    records: list[CaseRecord] = []
    for split in ("Train-Labeled", "Train-Unlabeled", "Validation"):
        image_dir, label_dir = _split_dir(root, split)
        for case_dir in sorted(p for p in image_dir.iterdir() if p.is_dir()):
            cbct = _first_existing(case_dir, ("CBCT.nii.gz", "CBCT.nii(1).gz"))
            for jaw in ("upper", "lower"):
                ios = _first_existing(case_dir, (f"{jaw}.stl", f"{jaw}(1).stl"))
                transform = None
                if label_dir is not None:
                    transform = _first_existing(label_dir / case_dir.name, (f"{jaw}_gt.npy",))
                complete = cbct is not None and ios is not None
                records.append(
                    CaseRecord(
                        split=split,
                        case_id=case_dir.name,
                        jaw=jaw,
                        cbct_path=str(cbct or ""),
                        ios_path=str(ios or ""),
                        transform_path=str(transform or ""),
                        complete=complete,
                    )
                )
    return records


def write_manifest(records: list[CaseRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def load_manifest(path: Path) -> list[CaseRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        CaseRecord(
            split=row["split"],
            case_id=row["case_id"],
            jaw=row["jaw"],
            cbct_path=row["cbct_path"],
            ios_path=row["ios_path"],
            transform_path=row.get("transform_path", ""),
            complete=row.get("complete", "True").lower() == "true",
        )
        for row in rows
    ]


def load_ios_points(path: Path, num_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        raise ValueError(f"IOS mesh has no vertices: {path}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(vertices), size=min(num_points, len(vertices)), replace=False)
    return vertices[indices], np.asarray(mesh.bounds, dtype=np.float64)


def ios_pca_side_variants(
    points: np.ndarray,
    fractions: tuple[float, ...] = (0.25, 0.35),
    include_full: bool = True,
) -> list[tuple[str, np.ndarray]]:
    centered = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    thin_axis = vectors[:, np.argmin(values)]
    projection = centered @ thin_axis
    variants: list[tuple[str, np.ndarray]] = [("full", points)] if include_full else []
    for fraction in fractions:
        if not 0.05 <= fraction <= 0.75:
            raise ValueError("IOS crop fractions must be between 0.05 and 0.75")
        low = np.quantile(projection, fraction)
        high = np.quantile(projection, 1.0 - fraction)
        variants.append((f"pca_low_{fraction:.2f}", points[projection <= low]))
        variants.append((f"pca_high_{fraction:.2f}", points[projection >= high]))
    return variants


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]
