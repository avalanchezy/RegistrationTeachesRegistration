from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_multiseed_reranker_ensemble import selected_row
from scripts.sweep_joint_pair_reranker import fit_pair_prior
from scripts.sweep_multimodal_reranker import (
    exact_fallback_errors,
    load_exact_errors,
    load_exact_official,
    summarize,
    summarize_official,
    write_csv,
)
from scripts.train_semisupervised_candidate_reranker import (
    cbct_groups_by_case,
    stratified_folds,
)
from task2reg.candidate_learning import (
    enrich_candidate_registration_metrics,
    load_candidate_groups,
)
from task2reg.data import load_manifest
from task2reg.priors import proper_protocol_rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply joint upper/lower constraints to cached multiseed scores."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labeled-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exact-loo", type=Path)
    parser.add_argument("--cbct-hash-cache", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--pair-top-k", type=int, nargs="+", default=(2, 3, 5, 8, 10))
    parser.add_argument(
        "--angle-weights", type=float, nargs="+", default=(0.0, 0.01, 0.025, 0.05, 0.075, 0.1)
    )
    parser.add_argument(
        "--translation-weights", type=float, nargs="+", default=(0.0, 0.025, 0.05, 0.075)
    )
    parser.add_argument("--allow-chirality-mismatch", action="store_true")
    parser.add_argument(
        "--selection-metric", choices=("tre", "official"), default="tre"
    )
    parser.add_argument(
        "--top-independent-methods",
        type=int,
        default=0,
        help="Run the joint grid only for the N best independent exact-fallback policies.",
    )
    return parser.parse_args()


def load_scored_candidates(path: Path, groups):
    candidates: dict[str, dict[str, dict[str, list[dict[str, object]]]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for payload in csv.DictReader(handle):
            key = (payload["case_id"], payload["jaw"])
            index = int(payload["candidate_index"])
            row = groups[key][index]
            if not np.isclose(float(payload["mean_tre_mm"]), float(row["mean_tre_mm"])):
                raise ValueError(f"Candidate score index mismatch for {key} index {index}")
            candidates.setdefault(payload["ensemble_method"], {}).setdefault(
                payload["case_id"], {}
            ).setdefault(payload["jaw"], []).append(
                {
                    "row": row,
                    "prediction_mm": float(payload["ensemble_score"]),
                    "candidate_index": index,
                    "optimization_target": payload.get(
                        "optimization_target", "mean_tre_mm"
                    ),
                }
            )
    for cases in candidates.values():
        for jaws in cases.values():
            for items in jaws.values():
                items.sort(key=lambda item: float(item["prediction_mm"]))
    return candidates


def retained_method_order(
    policies: dict[str, dict[str, dict[str, float]]], selection_metric: str
) -> list[str]:
    if selection_metric == "official":
        return sorted(
            policies,
            key=lambda method: (
                policies[method]["official"][
                    "exact_fallback_official_balanced_error"
                ],
                policies[method]["official"][
                    "exact_fallback_mean_rotation_error_deg"
                ],
                policies[method]["official"][
                    "exact_fallback_mean_translation_error_mm"
                ],
                policies[method]["exact_fallback"]["mean_tre_mm"],
                method,
            ),
        )
    if selection_metric != "tre":
        raise ValueError(f"Unsupported selection metric: {selection_metric}")
    return sorted(
        policies,
        key=lambda method: (
            policies[method]["exact_fallback"]["mean_tre_mm"],
            policies[method]["exact_fallback"]["p90_tre_mm"],
            policies[method]["raw"]["mean_tre_mm"],
            method,
        ),
    )


def pair_prior_by_case(records, groups, case_group, seeds: list[int], folds: int):
    priors: dict[str, list] = {case_id: [] for case_id, _ in groups}
    for seed in seeds:
        split = stratified_folds(groups, folds, seed, case_group)
        for validation_cases in split:
            prior = fit_pair_prior(records, validation_cases)
            for case_id in validation_cases:
                priors[case_id].append(prior)
    if any(len(values) != len(seeds) for values in priors.values()):
        raise RuntimeError("Every case must receive one leakage-free pair prior per seed")
    return priors


def precompute_pair_matrices(upper, lower, priors):
    upper_rotations = np.stack(
        [proper_protocol_rotation(np.asarray(item["row"]["transform"])[:3, :3]) for item in upper]
    )
    lower_rotations = np.stack(
        [proper_protocol_rotation(np.asarray(item["row"]["transform"])[:3, :3]) for item in lower]
    )
    relative = np.einsum("aik,bil->abkl", upper_rotations, lower_rotations)
    angles = []
    translations = []
    upper_translation = np.stack(
        [np.asarray(item["row"]["transform"], dtype=np.float64)[:3, 3] for item in upper]
    )
    lower_translation = np.stack(
        [np.asarray(item["row"]["transform"], dtype=np.float64)[:3, 3] for item in lower]
    )
    for prior in priors:
        trace = np.einsum("ij,abij->ab", prior.relative_rotation, relative)
        cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cosine)))
        delta = (
            upper_translation[:, None, :]
            - lower_translation[None, :, :]
            - prior.relative_translation[None, None, :]
        )
        translations.append(np.linalg.norm(delta, axis=2))
    upper_chirality = np.asarray([int(item["row"].get("chirality", 1)) for item in upper])
    lower_chirality = np.asarray([int(item["row"].get("chirality", 1)) for item in lower])
    return {
        "upper": upper,
        "lower": lower,
        "upper_scores": np.asarray([float(item["prediction_mm"]) for item in upper]),
        "lower_scores": np.asarray([float(item["prediction_mm"]) for item in lower]),
        "angle": np.mean(np.stack(angles), axis=0),
        "translation": np.mean(np.stack(translations), axis=0),
        "chirality_match": upper_chirality[:, None] == lower_chirality[None, :],
    }


def select_precomputed_pair(
    data,
    top_k: int,
    angle_weight: float,
    translation_weight: float,
    allow_chirality_mismatch: bool,
):
    upper_limit = min(top_k, len(data["upper"]))
    lower_limit = min(top_k, len(data["lower"]))
    objective = (
        data["upper_scores"][:upper_limit, None]
        + data["lower_scores"][None, :lower_limit]
        + angle_weight * data["angle"][:upper_limit, :lower_limit]
        + translation_weight * data["translation"][:upper_limit, :lower_limit]
    )
    if not allow_chirality_mismatch:
        objective = np.where(
            data["chirality_match"][:upper_limit, :lower_limit], objective, np.inf
        )
    if not np.isfinite(objective).any() and (
        upper_limit < len(data["upper"]) or lower_limit < len(data["lower"])
    ):
        return select_precomputed_pair(
            data,
            max(len(data["upper"]), len(data["lower"])),
            angle_weight,
            translation_weight,
            allow_chirality_mismatch,
        )
    if not np.isfinite(objective).any():
        raise RuntimeError("No chirality-consistent cached ensemble pair")
    upper_index, lower_index = np.unravel_index(np.argmin(objective), objective.shape)
    return {
        "upper": data["upper"][upper_index],
        "lower": data["lower"][lower_index],
        "objective": float(objective[upper_index, lower_index]),
        "relative_angle_deg": float(data["angle"][upper_index, lower_index]),
        "relative_translation_deviation_mm": float(
            data["translation"][upper_index, lower_index]
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(args.manifest)
    groups = load_candidate_groups(args.labeled_runs)
    enrich_candidate_registration_metrics(groups, records)
    case_group = cbct_groups_by_case(records, {key[0] for key in groups}, args.cbct_hash_cache)
    exact = load_exact_errors(args.exact_loo)
    exact_official = load_exact_official(args.exact_loo)
    scored = load_scored_candidates(args.candidate_scores, groups)
    priors = pair_prior_by_case(records, groups, case_group, args.seeds, args.folds)

    policy_rows = []
    policies = {}
    method_cases = {}
    for method, cases in scored.items():
        independent = []
        method_cases[method] = cases
        for case_id, jaws in sorted(cases.items()):
            if set(jaws) != {"upper", "lower"}:
                continue
            for jaw in ("upper", "lower"):
                item = jaws[jaw][0]
                independent.append(
                    selected_row(
                        case_id,
                        jaw,
                        groups[(case_id, jaw)],
                        int(item["candidate_index"]),
                        float(item["prediction_mm"]),
                        method,
                    )
                )
                independent[-1]["optimization_target"] = item[
                    "optimization_target"
                ]
        policy_rows.extend(independent)
        policies[method] = {
            "raw": summarize([float(row["mean_tre_mm"]) for row in independent]),
            "exact_fallback": summarize(exact_fallback_errors(independent, exact)),
            "official": summarize_official(independent, exact_official),
        }
    write_csv(args.output_dir / "independent_policies.csv", policy_rows)

    retained_methods = retained_method_order(policies, args.selection_metric)
    if args.top_independent_methods > 0:
        retained_methods = retained_methods[: args.top_independent_methods]
    pair_data = {}
    for method in retained_methods:
        pair_data[method] = {}
        for case_id, jaws in sorted(method_cases[method].items()):
            if set(jaws) != {"upper", "lower"}:
                continue
            pair_data[method][case_id] = precompute_pair_matrices(
                jaws["upper"], jaws["lower"], priors[case_id]
            )

    grid = []
    best_key = None
    best_rows = []
    for method, top_k, angle_weight, translation_weight in itertools.product(
        sorted(pair_data), args.pair_top_k, args.angle_weights, args.translation_weights
    ):
        rows_out = []
        for case_id, data in sorted(pair_data[method].items()):
            pair = select_precomputed_pair(
                data,
                top_k,
                angle_weight,
                translation_weight,
                args.allow_chirality_mismatch,
            )
            for jaw in ("upper", "lower"):
                item = pair[jaw]
                row = selected_row(
                    case_id,
                    jaw,
                    groups[(case_id, jaw)],
                    int(item["candidate_index"]),
                    float(item["prediction_mm"]),
                    method,
                )
                row["optimization_target"] = item["optimization_target"]
                row.update(
                    pair_objective=pair["objective"],
                    pair_relative_angle_deg=pair["relative_angle_deg"],
                    pair_translation_deviation_mm=pair[
                        "relative_translation_deviation_mm"
                    ],
                )
                rows_out.append(row)
        raw = summarize([float(row["mean_tre_mm"]) for row in rows_out])
        combined = summarize(exact_fallback_errors(rows_out, exact))
        official = summarize_official(rows_out, exact_official)
        payload = {
            "ensemble_method": method,
            "pair_top_k": top_k,
            "angle_weight_mm_per_deg": angle_weight,
            "translation_weight": translation_weight,
            **raw,
            "exact_fallback_mean_tre_mm": combined["mean_tre_mm"],
            "exact_fallback_median_tre_mm": combined["median_tre_mm"],
            "exact_fallback_p90_tre_mm": combined["p90_tre_mm"],
            "exact_fallback_max_tre_mm": combined["max_tre_mm"],
            **official,
        }
        grid.append(payload)
        if args.selection_metric == "official":
            key = (
                official["exact_fallback_official_balanced_error"],
                official["exact_fallback_mean_rotation_error_deg"],
                official["exact_fallback_mean_translation_error_mm"],
                combined["mean_tre_mm"],
            )
        else:
            key = (
                combined["mean_tre_mm"],
                combined["p90_tre_mm"],
                combined["max_tre_mm"],
                raw["mean_tre_mm"],
            )
        if best_key is None or key < best_key:
            best_key = key
            best_rows = rows_out
    grid.sort(
        key=(
            (lambda row: (
                float(row["exact_fallback_official_balanced_error"]),
                float(row["exact_fallback_mean_rotation_error_deg"]),
                float(row["exact_fallback_mean_translation_error_mm"]),
            ))
            if args.selection_metric == "official"
            else (lambda row: (
                float(row["exact_fallback_mean_tre_mm"]),
                float(row["exact_fallback_p90_tre_mm"]),
                float(row["exact_fallback_max_tre_mm"]),
            ))
        )
    )
    write_csv(args.output_dir / "joint_grid.csv", grid)
    write_csv(args.output_dir / "best_joint_oof.csv", best_rows)
    summary = {
        "cases": len({key[0] for key in groups}),
        "jaws": len(groups),
        "cbct_groups": len(set(case_group.values())),
        "seeds": args.seeds,
        "joint_methods": retained_methods,
        "selection_metric": args.selection_metric,
        "policies": policies,
        "best_joint": grid[0],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
