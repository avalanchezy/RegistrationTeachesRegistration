from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.candidate_learning import load_candidate_groups
from task2reg.deployment_ensemble import candidate_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export saved unsupervised registration scores for joint evaluation."
    )
    parser.add_argument("--labeled-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--score-keys",
        nargs="+",
        default=("rank_score_mm", "selection_score_mm"),
    )
    parser.add_argument(
        "--top-per-run",
        type=int,
        default=0,
        help="Export only this many lowest-selection-score candidates per active run.",
    )
    parser.add_argument("--exclude-upper-opposite-axial", action="store_true")
    return parser.parse_args()


def export_indices(
    candidates: list[dict],
    jaw: str,
    top_per_run: int,
    exclude_upper_opposite_axial: bool = False,
) -> list[int]:
    if top_per_run <= 0:
        return list(range(len(candidates)))
    run_names = {
        str(row.get("source_candidate_run", row.get("candidate_run", "")))
        for row in candidates
    }
    return candidate_indices(
        candidates,
        jaw,
        top_per_run * max(len(run_names), 1),
        balance_runs=True,
        exclude_upper_opposite_axial=exclude_upper_opposite_axial,
    )


def main() -> None:
    args = parse_args()
    groups = load_candidate_groups(args.labeled_runs)
    output = []
    for (case_id, jaw), candidates in sorted(groups.items()):
        selected_indices = export_indices(
            candidates,
            jaw,
            args.top_per_run,
            args.exclude_upper_opposite_axial,
        )
        for score_key in args.score_keys:
            if not all(score_key in row for row in candidates):
                continue
            for index in selected_indices:
                row = candidates[index]
                output.append(
                    {
                        "case_id": case_id,
                        "jaw": jaw,
                        "ensemble_method": f"unsupervised_{score_key}",
                        "candidate_index": index,
                        "ensemble_score": float(row[score_key]),
                        "mean_tre_mm": float(row["mean_tre_mm"]),
                        "candidate_run": str(row.get("candidate_run", "")),
                        "top_per_active_run": args.top_per_run,
                    }
                )
    if not output:
        raise RuntimeError("None of the requested unsupervised score keys were available")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(
        f"Wrote {len(output)} scores for {len(groups)} jaws to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
