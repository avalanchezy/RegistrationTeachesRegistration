"""Reference-centered field convergence diagnosis; not deployment performance."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2reg.journal.landscape import benchmark_landscape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', action='append', required=True, metavar='METHOD=PATH',
                        help='Repeat for matched dense/implicit checkpoints under one protocol')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--split', choices=['val', 'test', 'external_test'], default='test')
    parser.add_argument('--source')
    parser.add_argument('--reference-kind', choices=['manual', 'silver'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--rotation-degrees', type=float, nargs='+', default=[0., 5., 10., 20., 40.])
    parser.add_argument('--translation-mm', type=float, nargs='+', default=[0., 1., 2., 4., 8.])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--refinement-steps', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=.25)
    parser.add_argument('--point-budget', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--success-threshold-mm', type=float, default=2.)
    parser.add_argument('--improvement-tolerance-mm', type=float, default=1e-4)
    args = vars(parser.parse_args())
    checkpoints = {}
    for value in args.pop('checkpoint'):
        method, separator, path = value.partition('=')
        if not separator or not method.strip() or not path.strip() or method in checkpoints:
            parser.error('--checkpoint needs unique, nonempty METHOD=PATH entries')
        checkpoints[method] = path
    report = benchmark_landscape(checkpoints=checkpoints, **args)
    print(f"Reference-centered diagnostic only: {len(report['trials'])} trials; "
          f"wrote {args['output_dir'] / 'report.json'}")


if __name__ == '__main__':
    main()
