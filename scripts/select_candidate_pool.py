from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.candidate_learning import load_candidate_groups
from task2reg.deployment_ensemble import candidate_indices


def limited_candidate_pool(rows: list[dict], jaw: str, top_per_run: int) -> list[dict]:
    if top_per_run <= 0:
        return rows
    run_names = {
        str(row.get("source_candidate_run", row.get("candidate_run", "")))
        for row in rows
    }
    budget = top_per_run * max(len(run_names), 1)
    indices = candidate_indices(
        rows,
        jaw,
        budget,
        balance_runs=True,
        exclude_upper_opposite_axial=False,
    )
    return [rows[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the best fixed-score candidate across multiple run directories."
    )
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--score-field",
        choices=("rank_score_mm", "selection_score_mm", "fit_score_mm"),
        default="rank_score_mm",
    )
    parser.add_argument(
        "--top-per-run",
        type=int,
        default=0,
        help="Restrict the oracle and fixed selector to this many candidates per active run.",
    )
    args = parser.parse_args()

    selected_cases = set(args.case_ids)
    groups = load_candidate_groups(args.run_dirs)
    output_rows: list[dict[str, object]] = []
    for (case_id, jaw), rows in sorted(groups.items()):
        if case_id not in selected_cases:
            continue
        rows = limited_candidate_pool(rows, jaw, args.top_per_run)
        selected = min(rows, key=lambda row: float(row[args.score_field]))
        oracle = min(rows, key=lambda row: float(row["mean_tre_mm"]))
        output_rows.append(
            {
                "case_id": case_id,
                "jaw": jaw,
                "score_field": args.score_field,
                "score": float(selected[args.score_field]),
                "mean_tre_mm": float(selected["mean_tre_mm"]),
                "local_unsupervised_rank": int(selected["unsupervised_rank"]),
                "source_variant": selected["source_variant"],
                "target": selected["target"],
                "method": selected["method"],
                "candidate_run": selected.get("candidate_run", ""),
                "candidate_pool_size": len(rows),
                "active_candidate_runs": len(
                    {
                        str(
                            row.get(
                                "source_candidate_run", row.get("candidate_run", "")
                            )
                        )
                        for row in rows
                    }
                ),
                "oracle_mean_tre_mm": float(oracle["mean_tre_mm"]),
                "oracle_local_rank": int(oracle["unsupervised_rank"]),
                "oracle_candidate_run": oracle.get("candidate_run", ""),
                "ground_truth_chirality": int(selected.get("ground_truth_chirality", 0)),
                "predicted_chirality": int(selected.get("predicted_chirality", 0)),
            }
        )
    expected = {(case_id, jaw) for case_id in selected_cases for jaw in ("upper", "lower")}
    observed = {(str(row["case_id"]), str(row["jaw"])) for row in output_rows}
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"Missing candidate groups: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "evaluation.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    values = np.asarray([row["mean_tre_mm"] for row in output_rows], dtype=np.float64)
    oracle_values = np.asarray(
        [row["oracle_mean_tre_mm"] for row in output_rows], dtype=np.float64
    )
    chirality = np.asarray(
        [row["ground_truth_chirality"] == row["predicted_chirality"] for row in output_rows]
    )
    summary = {
        "jaws": len(output_rows),
        "mean_tre_mm": float(values.mean()),
        "median_tre_mm": float(np.median(values)),
        "p90_tre_mm": float(np.quantile(values, 0.9)),
        "oracle_mean_tre_mm": float(oracle_values.mean()),
        "oracle_median_tre_mm": float(np.median(oracle_values)),
        "chirality_accuracy": float(chirality.mean()),
        "top_per_active_run": args.top_per_run,
        "mean_candidate_pool_size": float(
            np.mean([row["candidate_pool_size"] for row in output_rows])
        ),
        "run_dirs": [str(path) for path in args.run_dirs],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
