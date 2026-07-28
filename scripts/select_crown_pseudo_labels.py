from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.data import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select confidence-ranked fold-specific crown pseudo labels."
    )
    parser.add_argument("--prediction-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cbct-hash-cache", type=Path)
    parser.add_argument("--max-per-cbct-group", type=int, default=1)
    parser.add_argument("--top-counts", type=int, nargs="+", default=(40, 80))
    parser.add_argument("--minimum-jaw-voxels", type=int, default=500)
    parser.add_argument("--maximum-jaw-voxels", type=int, default=6000)
    parser.add_argument("--minimum-confidence", type=float, default=0.45)
    parser.add_argument("--maximum-entropy", type=float, default=1.0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fold_id(value: str) -> int:
    parts = str(value).split("_")
    integers = [int(part) for part in parts if part.isdigit()]
    if len(integers) != 1:
        raise ValueError(f"Expected one teacher fold in prediction label {value!r}")
    return integers[0]


def main() -> None:
    args = parse_args()
    case_groups: dict[str, str] = {}
    if args.manifest is not None or args.cbct_hash_cache is not None:
        if args.manifest is None or args.cbct_hash_cache is None:
            raise ValueError(
                "--manifest and --cbct-hash-cache must be provided together"
            )
        cache = json.loads(args.cbct_hash_cache.read_text(encoding="utf-8"))
        hash_by_path = {
            str(Path(path).resolve()).lower(): str(payload["sha256"])
            for path, payload in cache.items()
        }
        for record in load_manifest(args.manifest):
            if record.split != "Train-Unlabeled" or not record.complete:
                continue
            resolved = str(Path(record.cbct_path).resolve()).lower()
            case_groups.setdefault(record.case_id, hash_by_path.get(resolved, record.case_id))
    all_ranked = {}
    audit_rows = []
    for prediction_dir in args.prediction_dirs:
        rows = read_rows(prediction_dir / "metrics.csv")
        if not rows:
            continue
        fold = fold_id(rows[0]["fold"])
        ranked = []
        for row in rows:
            upper = int(row["predicted_upper_voxels"])
            lower = int(row["predicted_lower_voxels"])
            confidence = float(row["foreground_confidence"])
            entropy = float(row["foreground_entropy"])
            volume_valid = (
                args.minimum_jaw_voxels <= upper <= args.maximum_jaw_voxels
                and args.minimum_jaw_voxels <= lower <= args.maximum_jaw_voxels
            )
            accepted = (
                volume_valid
                and confidence >= args.minimum_confidence
                and entropy <= args.maximum_entropy
            )
            volume_penalty = abs(np.log(max(upper, 1) / 2200.0)) + abs(
                np.log(max(lower, 1) / 2200.0)
            )
            score = confidence - 0.35 * entropy - 0.08 * volume_penalty
            label_path = prediction_dir / "label_arrays" / f"{row['case_id']}.npz"
            if accepted and label_path.exists():
                ranked.append((float(score), row["case_id"], label_path))
            audit_rows.append(
                {
                    "fold": fold,
                    "case_id": row["case_id"],
                    "upper_voxels": upper,
                    "lower_voxels": lower,
                    "confidence": confidence,
                    "entropy": entropy,
                    "selection_score": float(score),
                    "accepted_filter": int(accepted and label_path.exists()),
                }
            )
        ranked.sort(key=lambda item: (-item[0], item[1]))
        all_ranked[fold] = ranked

    if not all_ranked:
        raise RuntimeError("No fold-specific pseudo predictions were found")
    summaries = []
    for count in sorted(set(args.top_counts)):
        for fold, ranked in sorted(all_ranked.items()):
            destination = args.output_root / f"top_{count}" / f"fold_{fold}"
            destination.mkdir(parents=True, exist_ok=True)
            selected = []
            group_counts: dict[str, int] = {}
            for item in ranked:
                group = case_groups.get(item[1], item[1])
                if group_counts.get(group, 0) >= args.max_per_cbct_group:
                    continue
                selected.append(item)
                group_counts[group] = group_counts.get(group, 0) + 1
                if len(selected) >= count:
                    break
            for _, case_id, source in selected:
                shutil.copy2(source, destination / f"{case_id}.npz")
            summaries.append(
                {
                    "top_count": count,
                    "fold": fold,
                    "eligible": len(ranked),
                    "selected": len(selected),
                    "minimum_selected_score": selected[-1][0] if selected else None,
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    for filename, table in (("audit.csv", audit_rows), ("summary.csv", summaries)):
        with (args.output_root / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    summary = {
        "folds": len(all_ranked),
        "top_counts": sorted(set(args.top_counts)),
        "selection": summaries,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
