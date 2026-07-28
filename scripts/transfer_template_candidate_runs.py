from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.candidate_learning import load_candidate_groups
from task2reg.data import apply_transform, load_manifest
from task2reg.template_transfer import load_mesh_vertices


TRUTH_FIELDS = {
    "mean_tre_mm",
    "median_tre_mm",
    "p95_tre_mm",
    "translation_error_mm",
    "linear_frobenius_error",
    "relative_linear_frobenius",
    "ground_truth_chirality",
    "chirality_correct",
}


def pseudo_label_key(payload: dict) -> tuple[str, str]:
    return str(payload["case_id"]), str(payload["jaw"])


def source_key_for_label(payload: dict) -> tuple[str, str]:
    source_case = str(payload.get("source_labeled_case_id", "")) or str(
        payload.get("source_teacher_case_id", "")
    )
    if not source_case:
        raise ValueError("Pseudo label has no labeled or teacher source case")
    return source_case, str(payload["jaw"])


def source_teacher_transform(
    source_key: tuple[str, str],
    query_key: tuple[str, str],
    query_transform: np.ndarray,
    labels_by_key: dict[tuple[str, str], dict],
    records: dict,
) -> np.ndarray:
    if source_key == query_key:
        return query_transform
    source_payload = labels_by_key.get(source_key)
    if source_payload is not None:
        return np.asarray(source_payload["transform"], dtype=np.float64)
    source_record = records.get(source_key)
    if source_record is not None and source_record.transform_path:
        return np.load(source_record.transform_path, allow_pickle=False)
    raise ValueError(
        f"No teacher transform is available for {source_key[0]} {source_key[1]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer registration candidates to same-CBCT IOS copies using either "
            "a labeled or pseudo-labeled source teacher."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--pseudo-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-candidates-per-jaw", type=int, default=30)
    args = parser.parse_args()

    records = {(record.case_id, record.jaw): record for record in load_manifest(args.manifest)}
    source_groups = load_candidate_groups(args.source_runs)
    labels = json.loads(args.pseudo_labels.read_text(encoding="utf-8"))
    labels_by_key = {pseudo_label_key(payload): payload for payload in labels}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = []
    for payload in labels:
        key = pseudo_label_key(payload)
        try:
            source_key = source_key_for_label(payload)
        except ValueError:
            continue
        if key not in records or source_key not in source_groups:
            continue
        query_gt = np.asarray(payload["transform"], dtype=np.float64)
        try:
            source_transform = source_teacher_transform(
                source_key,
                key,
                query_gt,
                labels_by_key,
                records,
            )
        except ValueError as error:
            print(f"{key[0]} {key[1]} skipped: {error}", flush=True)
            continue
        query_to_source = np.linalg.inv(source_transform) @ query_gt
        query_vertices = load_mesh_vertices(Path(records[key].ios_path))
        query_centroid = query_vertices.mean(axis=0)
        rows = sorted(
            source_groups[source_key],
            key=lambda row: float(row["selection_score_mm"]),
        )[: args.max_candidates_per_jaw]
        transferred = []
        for source_row in rows:
            source_candidate_run = str(source_row.get("candidate_run", ""))
            row = {
                name: value
                for name, value in source_row.items()
                if name not in TRUTH_FIELDS and name != "candidate_run"
            }
            row["source_candidate_run"] = source_candidate_run
            row["transform"] = (
                np.asarray(source_row["transform"], dtype=np.float64) @ query_to_source
            ).tolist()
            if "transform_initial" in source_row:
                row["transform_initial"] = (
                    np.asarray(source_row["transform_initial"], dtype=np.float64)
                    @ query_to_source
                ).tolist()
            row["source_full_centroid"] = query_centroid.tolist()
            row["predicted_full_centroid"] = apply_transform(
                query_centroid[None], np.asarray(row["transform"], dtype=np.float64)
            )[0].tolist()
            metadata = dict(row.get("target_metadata", {}))
            metadata["source_volume_path"] = metadata.get("volume_path", "")
            metadata["volume_path"] = records[key].cbct_path
            row["target_metadata"] = metadata
            row["template_augmented_from"] = source_key[0]
            transferred.append(row)
        case_dir = args.output_dir / f"{key[0]}_{key[1]}"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "candidates.json").write_text(
            json.dumps(transferred, indent=2), encoding="utf-8"
        )
        audit.append(
            {
                "case_id": key[0],
                "jaw": key[1],
                "source_case_id": source_key[0],
                "source_teacher": payload.get("teacher", ""),
                "candidate_count": len(transferred),
                "query_to_source_determinant": float(np.linalg.det(query_to_source[:3, :3])),
            }
        )
        print(
            f"{key[0]} {key[1]} <- {source_key[0]}: {len(transferred)} candidates",
            flush=True,
        )
    if not audit:
        raise RuntimeError("No exact-template candidate groups were transferred")
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(f"Transferred {len(audit)} jaw groups")


if __name__ == "__main__":
    main()
