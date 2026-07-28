from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


POSTPROCESS_FIELDS = (
    "threshold",
    "minimum_component_voxels",
    "maximum_components",
    "minimum_hu",
)


@dataclass
class ProbabilityModel:
    name: str
    metadata: dict[str, object]
    values: dict[tuple[str, tuple[str, ...]], dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nested leave-one-CBCT-payload-group-out selection audit for crown "
            "probability fusion."
        )
    )
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cbct-hash-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric", default="symmetric_chamfer_mm")
    return parser.parse_args()


def cbct_groups(
    manifest: Path, hash_cache_path: Path, cases: list[str]
) -> dict[str, list[str]]:
    hash_cache = json.loads(hash_cache_path.read_text(encoding="utf-8-sig"))
    case_paths: dict[str, str] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "Train-Labeled":
                continue
            case_id = str(row["case_id"])
            cbct_path = str(row["cbct_path"])
            previous = case_paths.setdefault(case_id, cbct_path)
            if previous != cbct_path:
                raise ValueError(f"Case {case_id} maps to multiple CBCT paths")
    grouped: dict[str, list[str]] = {}
    for case_id in cases:
        if case_id not in case_paths:
            raise KeyError(f"Case {case_id} is absent from {manifest}")
        cached = hash_cache.get(case_paths[case_id])
        if not isinstance(cached, dict) or not cached.get("sha256"):
            raise KeyError(f"Missing CBCT hash for {case_paths[case_id]}")
        payload_hash = str(cached["sha256"]).lower()
        grouped.setdefault(payload_hash, []).append(case_id)
    return {
        payload_hash: sorted(group_cases)
        for payload_hash, group_cases in sorted(grouped.items())
    }


def load_models(summary_paths: list[Path], metric: str) -> list[ProbabilityModel]:
    configurations = []
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        configurations.extend(payload.get("configurations", []))
    models = []
    seen = set()
    for configuration in configurations:
        probability_dir_value = configuration.get("probability_dir")
        if not probability_dir_value:
            continue
        probability_dir = Path(str(probability_dir_value))
        key = str(probability_dir.resolve())
        if key in seen:
            continue
        per_case = probability_dir.parent / "postprocess" / "per_case.csv"
        if not per_case.is_file():
            continue
        seen.add(key)
        values: dict[tuple[str, tuple[str, ...]], dict[str, float]] = {}
        with per_case.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                postprocess = tuple(str(row[field]) for field in POSTPROCESS_FIELDS)
                values.setdefault((str(row["jaw"]), postprocess), {})[
                    str(row["case_id"])
                ] = float(row[metric])
        models.append(
            ProbabilityModel(
                name=probability_dir.parent.name,
                metadata={**configuration, "probability_dir": str(probability_dir)},
                values=values,
            )
        )
    if not models:
        raise RuntimeError("No probability models with completed per-case sweeps")
    return models


def mean_for_cases(values: dict[str, float], cases: set[str]) -> float:
    selected = [value for case_id, value in values.items() if case_id in cases]
    return float(np.mean(selected)) if selected else float("inf")


def choose_unified(
    models: list[ProbabilityModel], cases: set[str]
) -> tuple[ProbabilityModel, tuple[str, ...], float]:
    choices = []
    for model in models:
        postprocesses = sorted({key[1] for key in model.values})
        for postprocess in postprocesses:
            jaw_means = [
                mean_for_cases(model.values[(jaw, postprocess)], cases)
                for jaw in ("upper", "lower")
                if (jaw, postprocess) in model.values
            ]
            if len(jaw_means) == 2:
                choices.append((float(np.mean(jaw_means)), model, postprocess))
    score, model, postprocess = min(
        choices,
        key=lambda item: (item[0], model_label(item[1]), item[2]),
    )
    return model, postprocess, score


def choose_jaw_postprocess(
    models: list[ProbabilityModel], cases: set[str]
) -> tuple[ProbabilityModel, dict[str, tuple[str, ...]], float]:
    choices = []
    for model in models:
        selected = {}
        means = []
        for jaw in ("upper", "lower"):
            jaw_choices = [
                (mean_for_cases(values, cases), postprocess)
                for (candidate_jaw, postprocess), values in model.values.items()
                if candidate_jaw == jaw
            ]
            score, postprocess = min(
                jaw_choices,
                key=lambda item: (item[0], item[1]),
            )
            selected[jaw] = postprocess
            means.append(score)
        choices.append((float(np.mean(means)), model, selected))
    score, model, selected = min(
        choices,
        key=lambda item: (
            item[0],
            model_label(item[1]),
            item[2]["upper"],
            item[2]["lower"],
        ),
    )
    return model, selected, score


def choose_jaw_models(
    models: list[ProbabilityModel], cases: set[str]
) -> tuple[dict[str, tuple[ProbabilityModel, tuple[str, ...]]], float]:
    selected = {}
    means = []
    for jaw in ("upper", "lower"):
        choices = []
        for model in models:
            choices.extend(
                (
                    mean_for_cases(values, cases),
                    model,
                    postprocess,
                )
                for (candidate_jaw, postprocess), values in model.values.items()
                if candidate_jaw == jaw
            )
        score, model, postprocess = min(
            choices,
            key=lambda item: (item[0], model_label(item[1]), item[2]),
        )
        selected[jaw] = (model, postprocess)
        means.append(score)
    return selected, float(np.mean(means))


def heldout_score(
    model: ProbabilityModel,
    postprocess_by_jaw: dict[str, tuple[str, ...]],
    case_id: str,
) -> float:
    return float(
        np.mean(
            [
                model.values[(jaw, postprocess_by_jaw[jaw])][case_id]
                for jaw in ("upper", "lower")
            ]
        )
    )


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_mm": float(np.mean(array)),
        "median_mm": float(np.median(array)),
        "p90_mm": float(np.quantile(array, 0.9)),
        "maximum_mm": float(np.max(array)),
    }


def model_label(model: ProbabilityModel) -> str:
    method = str(model.metadata.get("method", model.name))
    weight = model.metadata.get("supervised_weight")
    return f"{method}/supervised_{float(weight):.4f}" if weight is not None else method


def postprocess_payload(values: tuple[str, ...]) -> dict[str, float | int]:
    return {
        "threshold": float(values[0]),
        "minimum_component_voxels": int(float(values[1])),
        "maximum_components": int(float(values[2])),
        "minimum_hu": float(values[3]),
    }


def configuration_key(
    model: ProbabilityModel, postprocess: tuple[str, ...]
) -> str:
    return json.dumps(
        {
            "probability_dir": str(model.metadata["probability_dir"]),
            **postprocess_payload(postprocess),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def configuration_mean(
    model: ProbabilityModel,
    postprocess: tuple[str, ...],
    cases: list[str],
) -> float:
    by_jaw = {"upper": postprocess, "lower": postprocess}
    return float(np.mean([heldout_score(model, by_jaw, case_id) for case_id in cases]))


def main() -> None:
    args = parse_args()
    models = load_models(list(args.summaries), args.metric)
    cases = sorted(
        set.intersection(
            *[
                set(next(iter(model.values.values())))
                for model in models
                if model.values
            ]
        )
    )
    if len(cases) < 2:
        raise RuntimeError(f"Too few common cases for nested selection: {cases}")
    groups = cbct_groups(args.manifest, args.cbct_hash_cache, cases)
    if len(groups) < 2:
        raise RuntimeError(f"Too few independent CBCT groups: {len(groups)}")
    records = []
    selection_counts = {
        "unified": Counter(),
        "unified_configuration": Counter(),
        "jaw_postprocess": Counter(),
        "jaw_model_upper": Counter(),
        "jaw_model_lower": Counter(),
    }
    unified_configurations: dict[
        str, tuple[ProbabilityModel, tuple[str, ...]]
    ] = {}
    for payload_hash, heldout_cases in groups.items():
        training = set(cases) - set(heldout_cases)
        unified_model, unified_postprocess, _ = choose_unified(models, training)
        jaw_pp_model, jaw_postprocess, _ = choose_jaw_postprocess(models, training)
        jaw_models, _ = choose_jaw_models(models, training)
        selection_counts["unified"][model_label(unified_model)] += 1
        unified_key = configuration_key(unified_model, unified_postprocess)
        unified_configurations[unified_key] = (unified_model, unified_postprocess)
        selection_counts["unified_configuration"][unified_key] += 1
        selection_counts["jaw_postprocess"][model_label(jaw_pp_model)] += 1
        selection_counts["jaw_model_upper"][model_label(jaw_models["upper"][0])] += 1
        selection_counts["jaw_model_lower"][model_label(jaw_models["lower"][0])] += 1
        for case_id in heldout_cases:
            unified_score = heldout_score(
                unified_model,
                {"upper": unified_postprocess, "lower": unified_postprocess},
                case_id,
            )
            jaw_pp_score = heldout_score(jaw_pp_model, jaw_postprocess, case_id)
            jaw_model_score = float(
                np.mean(
                    [
                        jaw_models[jaw][0].values[(jaw, jaw_models[jaw][1])][case_id]
                        for jaw in ("upper", "lower")
                    ]
                )
            )
            records.append(
                {
                    "case_id": case_id,
                    "cbct_sha256": payload_hash,
                    "heldout_group_cases": heldout_cases,
                    "unified_model": model_label(unified_model),
                    "unified_postprocess": postprocess_payload(unified_postprocess),
                    "unified_mm": unified_score,
                    "jaw_postprocess_mm": jaw_pp_score,
                    "jaw_model_mm": jaw_model_score,
                    "jaw_postprocess_delta_mm": jaw_pp_score - unified_score,
                    "jaw_model_delta_mm": jaw_model_score - unified_score,
                }
            )

    all_cases = set(cases)
    fixed_unified_model, fixed_unified_pp, fixed_unified_train = choose_unified(
        models, all_cases
    )
    fixed_jaw_pp_model, fixed_jaw_pp, fixed_jaw_pp_train = choose_jaw_postprocess(
        models, all_cases
    )
    fixed_jaw_models, fixed_jaw_models_train = choose_jaw_models(models, all_cases)
    recommended_key = min(
        unified_configurations,
        key=lambda key: (
            -selection_counts["unified_configuration"][key],
            configuration_mean(*unified_configurations[key], cases),
            key,
        ),
    )
    recommended_model, recommended_postprocess = unified_configurations[
        recommended_key
    ]
    recommended_mean = configuration_mean(
        recommended_model, recommended_postprocess, cases
    )
    recommended = {
        **recommended_model.metadata,
        **postprocess_payload(recommended_postprocess),
        "mean_symmetric_chamfer_mm": recommended_mean,
        "nested_selection_count": selection_counts["unified_configuration"][
            recommended_key
        ],
        "nested_selection_fraction": selection_counts["unified_configuration"][
            recommended_key
        ]
        / len(groups),
        "selection_protocol": (
            "modal_nested_leave_one_cbct_payload_group_out_unified_configuration"
        ),
    }
    summary = {
        "metric": args.metric,
        "cases": len(cases),
        "cbct_groups": len(groups),
        "models": len(models),
        "nested": {
            "unified": summarize([row["unified_mm"] for row in records]),
            "jaw_specific_postprocess": summarize(
                [row["jaw_postprocess_mm"] for row in records]
            ),
            "jaw_specific_model_and_postprocess": summarize(
                [row["jaw_model_mm"] for row in records]
            ),
            "mean_jaw_postprocess_delta_mm": float(
                np.mean([row["jaw_postprocess_delta_mm"] for row in records])
            ),
            "mean_jaw_model_delta_mm": float(
                np.mean([row["jaw_model_delta_mm"] for row in records])
            ),
            "selection_counts": {
                name: dict(counter.most_common())
                for name, counter in selection_counts.items()
            },
        },
        "fixed_all_oof_selection": {
            "unified": {
                "model": model_label(fixed_unified_model),
                "postprocess": dict(zip(POSTPROCESS_FIELDS, fixed_unified_pp)),
                "mean_mm": fixed_unified_train,
            },
            "jaw_specific_postprocess": {
                "model": model_label(fixed_jaw_pp_model),
                "postprocess": {
                    jaw: dict(zip(POSTPROCESS_FIELDS, values))
                    for jaw, values in fixed_jaw_pp.items()
                },
                "mean_mm": fixed_jaw_pp_train,
            },
            "jaw_specific_model_and_postprocess": {
                "models": {
                    jaw: model_label(model_and_pp[0])
                    for jaw, model_and_pp in fixed_jaw_models.items()
                },
                "mean_mm": fixed_jaw_models_train,
            },
        },
        "best": recommended,
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
