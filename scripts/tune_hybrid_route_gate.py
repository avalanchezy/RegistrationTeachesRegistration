from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune a leakage-safe ToothSeg/threshold route gate from base OOF predictions."
    )
    parser.add_argument("--toothseg-oof", type=Path, required=True)
    parser.add_argument("--threshold-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold-prediction-caps",
        type=float,
        nargs="+",
        default=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, float("inf")),
    )
    parser.add_argument(
        "--minimum-predicted-advantages",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def threshold_selected(row: dict[str, object], cap: float, advantage: float) -> bool:
    threshold_prediction = float(row["threshold_predicted_tre_mm"])
    toothseg_prediction = float(row["toothseg_predicted_tre_mm"])
    return (
        threshold_prediction <= cap
        and toothseg_prediction - threshold_prediction >= advantage
    )


def selected_error(row: dict[str, object], cap: float, advantage: float) -> float:
    field = "threshold_tre_mm" if threshold_selected(row, cap, advantage) else "toothseg_tre_mm"
    return float(row[field])


def gate_grid(
    rows: list[dict[str, object]], caps: list[float], advantages: list[float]
) -> list[dict[str, object]]:
    result = []
    for cap in caps:
        for advantage in advantages:
            errors = np.asarray(
                [selected_error(row, cap, advantage) for row in rows], dtype=np.float64
            )
            result.append(
                {
                    "threshold_prediction_cap_mm": cap,
                    "minimum_predicted_advantage_mm": advantage,
                    "mean_tre_mm": float(errors.mean()),
                    "median_tre_mm": float(np.median(errors)),
                    "p90_tre_mm": float(np.quantile(errors, 0.9)),
                    "threshold_selections": sum(
                        threshold_selected(row, cap, advantage) for row in rows
                    ),
                }
            )
    return sorted(
        result,
        key=lambda row: (
            row["mean_tre_mm"],
            row["p90_tre_mm"],
            row["threshold_selections"],
            row["minimum_predicted_advantage_mm"],
        ),
    )


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_tre_mm": float(array.mean()),
        "median_tre_mm": float(np.median(array)),
        "p90_tre_mm": float(np.quantile(array, 0.9)),
        "max_tre_mm": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    toothseg = {
        (row["case_id"], row["jaw"]): row for row in read_csv(args.toothseg_oof)
    }
    threshold = {
        (row["case_id"], row["jaw"]): row for row in read_csv(args.threshold_oof)
    }
    if set(toothseg) != set(threshold):
        missing_toothseg = sorted(set(threshold) - set(toothseg))
        missing_threshold = sorted(set(toothseg) - set(threshold))
        raise ValueError(
            f"OOF keys differ; missing ToothSeg={missing_toothseg}, "
            f"missing threshold={missing_threshold}"
        )

    rows: list[dict[str, object]] = []
    for key in sorted(toothseg):
        toothseg_row = toothseg[key]
        threshold_row = threshold[key]
        rows.append(
            {
                "case_id": key[0],
                "jaw": key[1],
                "cbct_group": toothseg_row["cbct_group"],
                "toothseg_predicted_tre_mm": float(toothseg_row["predicted_tre_mm"]),
                "toothseg_tre_mm": float(toothseg_row["mean_tre_mm"]),
                "threshold_predicted_tre_mm": float(threshold_row["predicted_tre_mm"]),
                "threshold_tre_mm": float(threshold_row["mean_tre_mm"]),
            }
        )

    caps = list(dict.fromkeys(args.threshold_prediction_caps))
    advantages = list(dict.fromkeys(args.minimum_predicted_advantages))
    best_by_jaw: dict[str, dict[str, object]] = {}
    all_grid_rows: list[dict[str, object]] = []
    for jaw in ("upper", "lower"):
        jaw_rows = [row for row in rows if row["jaw"] == jaw]
        grid = gate_grid(jaw_rows, caps, advantages)
        best_by_jaw[jaw] = grid[0]
        for row in grid:
            all_grid_rows.append({"jaw": jaw, **row})
    write_csv(args.output_dir / "all_data_gate_grid.csv", all_grid_rows)

    nested_rows: list[dict[str, object]] = []
    for group in sorted({str(row["cbct_group"]) for row in rows}):
        for jaw in ("upper", "lower"):
            training = [
                row
                for row in rows
                if row["jaw"] == jaw and str(row["cbct_group"]) != group
            ]
            validation = [
                row
                for row in rows
                if row["jaw"] == jaw and str(row["cbct_group"]) == group
            ]
            if not validation:
                continue
            gate = gate_grid(training, caps, advantages)[0]
            cap = float(gate["threshold_prediction_cap_mm"])
            advantage = float(gate["minimum_predicted_advantage_mm"])
            for row in validation:
                use_threshold = threshold_selected(row, cap, advantage)
                selected = "threshold" if use_threshold else "toothseg"
                error = selected_error(row, cap, advantage)
                nested_rows.append(
                    {
                        **row,
                        "selected_route": selected,
                        "selected_tre_mm": error,
                        "oracle_tre_mm": min(
                            float(row["toothseg_tre_mm"]),
                            float(row["threshold_tre_mm"]),
                        ),
                        "threshold_prediction_cap_mm": cap,
                        "minimum_predicted_advantage_mm": advantage,
                    }
                )
    write_csv(args.output_dir / "nested_group_oof.csv", nested_rows)

    simple_errors = [
        float(row["threshold_tre_mm"])
        if float(row["threshold_predicted_tre_mm"])
        <= float(row["toothseg_predicted_tre_mm"])
        else float(row["toothseg_tre_mm"])
        for row in rows
    ]
    fixed_errors = [
        selected_error(
            row,
            float(best_by_jaw[str(row["jaw"])]["threshold_prediction_cap_mm"]),
            float(best_by_jaw[str(row["jaw"])]["minimum_predicted_advantage_mm"]),
        )
        for row in rows
    ]
    summary = {
        "cases": len({row["case_id"] for row in rows}),
        "jaws": len(rows),
        "cbct_groups": len({row["cbct_group"] for row in rows}),
        "best_fixed_gate_by_jaw": best_by_jaw,
        "toothseg_only": summarize([float(row["toothseg_tre_mm"]) for row in rows]),
        "threshold_only": summarize([float(row["threshold_tre_mm"]) for row in rows]),
        "simple_min_predicted": summarize(simple_errors),
        "nested_group_gate": summarize(
            [float(row["selected_tre_mm"]) for row in nested_rows]
        ),
        "fixed_all_data_gate": summarize(fixed_errors),
        "oracle_route": summarize(
            [
                min(float(row["toothseg_tre_mm"]), float(row["threshold_tre_mm"]))
                for row in rows
            ]
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
