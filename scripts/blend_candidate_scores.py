from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_multiseed_reranker_ensemble import fractional_ranks
from scripts.sweep_multimodal_reranker import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend two strict OOF candidate score tables by within-group rank."
    )
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-methods", nargs="+", required=True)
    parser.add_argument("--second-methods", nargs="+", required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    return parser.parse_args()


def load_scores(path: Path):
    grouped: dict[tuple[str, str, str], dict[int, dict[str, str]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["ensemble_method"], row["case_id"], row["jaw"])
            grouped[key][int(row["candidate_index"])] = row
    return grouped


def blend_tables(
    first,
    second,
    first_methods: list[str],
    second_methods: list[str],
    alphas: list[float],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    group_keys = sorted({(key[1], key[2]) for key in first})
    for first_method in first_methods:
        for second_method in second_methods:
            for case_id, jaw in group_keys:
                first_rows = first[(first_method, case_id, jaw)]
                second_rows = second[(second_method, case_id, jaw)]
                if set(first_rows) != set(second_rows):
                    raise ValueError(
                        f"Candidate pools differ for {case_id} {jaw}: "
                        f"{first_method} versus {second_method}"
                    )
                indices = np.asarray(sorted(first_rows), dtype=int)
                first_ranks = fractional_ranks(
                    np.asarray(
                        [float(first_rows[int(index)]["ensemble_score"]) for index in indices]
                    )
                )
                second_ranks = fractional_ranks(
                    np.asarray(
                        [float(second_rows[int(index)]["ensemble_score"]) for index in indices]
                    )
                )
                for alpha in alphas:
                    method = (
                        f"{first_method}__{second_method}__a"
                        f"{alpha:.3f}".replace(".", "p")
                    )
                    blended = alpha * first_ranks + (1.0 - alpha) * second_ranks
                    for index, score in zip(indices, blended):
                        source = first_rows[int(index)]
                        output.append(
                            {
                                "case_id": case_id,
                                "jaw": jaw,
                                "ensemble_method": method,
                                "candidate_index": int(index),
                                "ensemble_score": float(score),
                                "mean_tre_mm": float(source["mean_tre_mm"]),
                                "candidate_run": source.get("candidate_run", ""),
                            }
                        )
    return output


def main() -> None:
    args = parse_args()
    first = load_scores(args.first)
    second = load_scores(args.second)
    rows = blend_tables(
        first, second, args.first_methods, args.second_methods, args.alphas
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} blended candidate scores to {args.output}")


if __name__ == "__main__":
    main()
