from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.data import load_manifest
from task2reg.template_transfer import load_mesh_vertices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose pseudo-label teacher transforms as a candidate run."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pseudo-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="Train-Unlabeled")
    parser.add_argument("--case-ids", nargs="*")
    return parser.parse_args()


def candidate_from_label(payload: dict, centroid: np.ndarray) -> dict[str, object]:
    transform = np.asarray(payload["transform"], dtype=np.float64)
    predicted = float(payload.get("predicted_tre_mm", 0.0))
    return {
        "transform": transform.tolist(),
        "transform_initial": transform.tolist(),
        "selection_score_mm": predicted,
        "registration_score_mm": predicted,
        "source_fit_score_mm": predicted,
        "unsupervised_rank": 1,
        "source_full_centroid": np.asarray(centroid, dtype=np.float64).tolist(),
        "predicted_full_centroid": (
            transform[:3, :3] @ centroid + transform[:3, 3]
        ).tolist(),
        "chirality": int(np.sign(np.linalg.det(transform[:3, :3]))),
        "source_variant": "full",
        "target": "pseudo_teacher",
        "method": str(payload.get("teacher", "pseudo_teacher")),
        "teacher": str(payload.get("teacher", "")),
        "source_teacher_case_id": str(payload.get("source_teacher_case_id", "")),
    }


def main() -> None:
    args = parse_args()
    wanted = {str(case_id).zfill(3) for case_id in (args.case_ids or ())}
    records = {
        (record.case_id, record.jaw): record
        for record in load_manifest(args.manifest)
        if record.split == args.split
        and record.complete
        and (not wanted or record.case_id in wanted)
    }
    labels = json.loads(args.pseudo_labels.read_text(encoding="utf-8"))
    written = []
    rejected = []
    for payload in labels:
        key = (str(payload["case_id"]), str(payload["jaw"]))
        if not payload.get("accepted", 1) or key not in records:
            continue
        try:
            vertices = load_mesh_vertices(Path(records[key].ios_path))
        except (ValueError, RuntimeError) as error:
            rejected.append({"case_id": key[0], "jaw": key[1], "reason": str(error)})
            continue
        row = candidate_from_label(payload, vertices.mean(axis=0))
        case_dir = args.output_dir / f"{key[0]}_{key[1]}"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "candidates.json").write_text(
            json.dumps([row], indent=2), encoding="utf-8"
        )
        written.append(key)
    if not written:
        raise RuntimeError("No pseudo labels matched the requested manifest records")
    summary = {
        "candidate_jaws": len(written),
        "candidate_cases": len({key[0] for key in written}),
        "rejected": rejected,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
