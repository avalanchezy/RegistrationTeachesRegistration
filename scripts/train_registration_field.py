"""Train the journal extension on prepared, patient-grouped data."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2reg.journal.engine import TrainingConfig, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-support", type=Path)
    parser.add_argument("--initialize-field", type=Path)
    parser.add_argument("--initialization-provenance", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = TrainingConfig.from_json(args.config)
    overrides = {k: getattr(args, k) for k in ("device", "epochs", "seed") if getattr(args, k) is not None}
    config = replace(config, **overrides)
    train(args.manifest, args.output_dir, config, data_root=args.data_root,
          resume=args.resume, initialize_support=args.initialize_support,
          initialize_field=args.initialize_field,
          initialization_provenance=args.initialization_provenance)


if __name__ == "__main__":
    main()
