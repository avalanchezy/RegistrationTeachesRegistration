from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweep_cross_modal_consensus import midpoint_transform, select_pair
from task2reg.candidate_learning import load_candidate_groups
from task2reg.data import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export label-free pseudo transforms accepted by threshold/ToothSeg "
            "cross-modal agreement."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--threshold-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--toothseg-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="Train-Unlabeled")
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument(
        "--selection", choices=("threshold", "toothseg", "midpoint"), default="toothseg"
    )
    parser.add_argument("--max-disagreement-mm", type=float, required=True)
    parser.add_argument("--anchor-radius-mm", type=float, default=20.0)
    parser.add_argument("--confidence-scale", type=float, default=1.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-toothseg-detected-teeth", type=int, default=0)
    parser.add_argument("--exclude-upper-opposite-axial", action="store_true")
    return parser.parse_args()


def agreement_confidence(
    disagreement_mm: float, gate_mm: float, scale: float = 1.0
) -> float:
    if gate_mm <= 0.0:
        raise ValueError("gate_mm must be positive")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    ratio = max(float(disagreement_mm), 0.0) / float(gate_mm)
    return float(math.exp(-0.5 * (ratio / scale) ** 2))


def selected_transform(
    selection: str,
    threshold: dict,
    toothseg: dict,
    center: np.ndarray,
) -> np.ndarray:
    if selection == "threshold":
        return np.asarray(threshold["transform"], dtype=np.float64)
    if selection == "toothseg":
        return np.asarray(toothseg["transform"], dtype=np.float64)
    if selection == "midpoint":
        return midpoint_transform(
            np.asarray(threshold["transform"], dtype=np.float64),
            np.asarray(toothseg["transform"], dtype=np.float64),
            center,
        )
    raise ValueError(f"Unknown cross-modal selection: {selection}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.max_disagreement_mm <= 0.0:
        raise ValueError("--max-disagreement-mm must be positive")
    if args.min_toothseg_detected_teeth < 0:
        raise ValueError("--min-toothseg-detected-teeth cannot be negative")
    wanted = {str(case_id).zfill(3) for case_id in (args.case_ids or ())}
    records = {
        (record.case_id, record.jaw): record
        for record in load_manifest(args.manifest)
        if record.split == args.split
        and record.complete
        and (not wanted or record.case_id in wanted)
    }
    threshold_groups = load_candidate_groups(args.threshold_runs)
    toothseg_groups = load_candidate_groups(args.toothseg_runs)
    keys = sorted(set(records) & set(threshold_groups) & set(toothseg_groups))
    if not keys:
        raise RuntimeError("No manifest jaws have both threshold and ToothSeg candidates")

    labels: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    quality_eligible_jaws = 0
    for case_id, jaw in keys:
        key = (case_id, jaw)
        toothseg_rows = [
            row
            for row in toothseg_groups[key]
            if int(row.get("target_metadata", {}).get("detected_teeth", 0))
            >= args.min_toothseg_detected_teeth
        ]
        if not toothseg_rows:
            detected = max(
                (
                    int(row.get("target_metadata", {}).get("detected_teeth", 0))
                    for row in toothseg_groups[key]
                ),
                default=0,
            )
            audit.append(
                {
                    "case_id": case_id,
                    "jaw": jaw,
                    "accepted": 0,
                    "reason": "insufficient_toothseg_labels",
                    "toothseg_detected_teeth": detected,
                    "cross_modal_disagreement_mm": "",
                    "confidence": "",
                    "threshold_candidate_run": "",
                    "toothseg_candidate_run": "",
                }
            )
            continue
        quality_eligible_jaws += 1
        try:
            pair_score, disagreement, threshold, toothseg, center = select_pair(
                threshold_groups[key],
                toothseg_rows,
                jaw,
                args.top_k,
                args.rank_weight,
                args.anchor_radius_mm,
                args.exclude_upper_opposite_axial,
            )
        except RuntimeError as error:
            audit.append(
                {
                    "case_id": case_id,
                    "jaw": jaw,
                    "accepted": 0,
                    "reason": str(error),
                    "toothseg_detected_teeth": "",
                    "cross_modal_disagreement_mm": "",
                    "confidence": "",
                    "threshold_candidate_run": "",
                    "toothseg_candidate_run": "",
                }
            )
            continue
        confidence = agreement_confidence(
            disagreement, args.max_disagreement_mm, args.confidence_scale
        )
        detected_teeth = int(
            toothseg.get("target_metadata", {}).get("detected_teeth", 0)
        )
        accepted = (
            disagreement <= args.max_disagreement_mm
            and confidence >= args.min_confidence
        )
        audit.append(
            {
                "case_id": case_id,
                "jaw": jaw,
                "accepted": int(accepted),
                "reason": "" if accepted else "agreement_gate",
                "toothseg_detected_teeth": detected_teeth,
                "cross_modal_disagreement_mm": disagreement,
                "confidence": confidence,
                "threshold_candidate_run": threshold.get("candidate_run", ""),
                "toothseg_candidate_run": toothseg.get("candidate_run", ""),
            }
        )
        if not accepted:
            continue
        transform = selected_transform(args.selection, threshold, toothseg, center)
        record = records[key]
        labels.append(
            {
                "case_id": case_id,
                "jaw": jaw,
                "accepted": 1,
                "confidence": confidence,
                "consensus_count": 2,
                "ios_path": record.ios_path,
                "cbct_path": record.cbct_path,
                "transform": transform.tolist(),
                "predicted_tre_mm": 0.0,
                "teacher": "cross_modal_consensus_teacher",
                "source_labeled_case_id": "",
                "source_teacher_case_id": case_id,
                "selection": args.selection,
                "cross_modal_disagreement_mm": disagreement,
                "cross_modal_pair_score": float(pair_score),
                "toothseg_detected_teeth": detected_teeth,
                "threshold_transform": np.asarray(
                    threshold["transform"], dtype=np.float64
                ).tolist(),
                "toothseg_transform": np.asarray(
                    toothseg["transform"], dtype=np.float64
                ).tolist(),
                "threshold_candidate_run": threshold.get("candidate_run", ""),
                "toothseg_candidate_run": toothseg.get("candidate_run", ""),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pseudo_labels.json").write_text(
        json.dumps(labels, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "audit.csv", audit)
    summary = {
        "eligible_jaws": len(keys),
        "quality_eligible_jaws": quality_eligible_jaws,
        "accepted_jaws": len(labels),
        "accepted_cases": len({str(row["case_id"]) for row in labels}),
        "coverage": len(labels) / len(keys),
        "quality_coverage": (
            len(labels) / quality_eligible_jaws if quality_eligible_jaws else 0.0
        ),
        "top_k": args.top_k,
        "rank_weight": args.rank_weight,
        "selection": args.selection,
        "max_disagreement_mm": args.max_disagreement_mm,
        "anchor_radius_mm": args.anchor_radius_mm,
        "confidence_scale": args.confidence_scale,
        "min_confidence": args.min_confidence,
        "min_toothseg_detected_teeth": args.min_toothseg_detected_teeth,
        "mean_accepted_disagreement_mm": (
            float(np.mean([row["cross_modal_disagreement_mm"] for row in labels]))
            if labels
            else None
        ),
        "mean_accepted_confidence": (
            float(np.mean([row["confidence"] for row in labels])) if labels else None
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
