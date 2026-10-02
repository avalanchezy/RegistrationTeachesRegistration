"""Score and refine existing geometry candidates with a journal field."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2reg.journal.inference import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--refinement-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=.25)
    parser.add_argument("--point-budget", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    predict(args.manifest, args.checkpoint, args.output_dir, data_root=args.data_root,
            split=args.split, device=args.device, refinement_steps=args.refinement_steps,
            learning_rate=args.learning_rate, point_budget=args.point_budget,
            seed=args.seed, evaluate=args.evaluate)


if __name__ == "__main__":
    main()
