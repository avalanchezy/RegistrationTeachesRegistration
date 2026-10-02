"""Verify unlabeled registrations and export weighted IOS-derived field targets."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2reg.journal.pseudo import export_verified_pseudo
from task2reg.journal.verification import VerificationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--sectors", type=int, default=4)
    parser.add_argument("--refinement-steps", type=int, default=20)
    parser.add_argument("--point-budget", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    config = VerificationConfig(**json.loads(args.gate_config.read_text()))
    export_verified_pseudo(args.manifest, args.checkpoint, args.output_dir, data_root=args.data_root,
                           device=args.device, config=config, starts=args.starts, sectors=args.sectors,
                           refinement_steps=args.refinement_steps, point_budget=args.point_budget,
                           learning_rate=args.learning_rate, seed=args.seed, max_cases=args.max_cases)


if __name__ == "__main__":
    main()
