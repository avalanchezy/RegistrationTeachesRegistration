from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task2reg.data import load_manifest
from task2reg.priors import fit_rotation_prior, save_rotation_priors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-case-ids", nargs="*", default=[])
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    excluded = set(args.exclude_case_ids)
    priors = {
        jaw: fit_rotation_prior(records, jaw, excluded_cases=excluded)
        for jaw in ("upper", "lower")
    }
    save_rotation_priors(priors, args.output)
    print(f"Saved rotation prior: {args.output}")
    for jaw, prior in priors.items():
        values = prior.json_dict()
        print(
            f"{jaw}: n={values['training_count']} median={values['angle_deg_median']:.2f} deg "
            f"p95={values['angle_deg_p95']:.2f} deg max={values['angle_deg_max']:.2f} deg"
        )


if __name__ == "__main__":
    main()
