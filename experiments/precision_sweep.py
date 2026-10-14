"""ImageNette coding-precision sweep: tabular rank and accuracy results."""
import argparse
from pathlib import Path
from simcoding.common import save_csv
from simcoding.train import add_training_arguments, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, "imagenette")
    parser.add_argument("--eps2", type=float, default=53.2, help="Base epsilon squared.")
    parser.add_argument("--divisors", type=float, nargs="+", default=[16, 8, 4, 2, 1])
    parser.add_argument("--output", default="outputs/precision_sweep")
    args = parser.parse_args()
    if any(divisor <= 0 for divisor in args.divisors) or len(set(args.divisors)) != len(args.divisors):
        parser.error("divisors must be unique and positive")
    rows = []
    for divisor in args.divisors:
        row = run_training(args, "imagenette", args.eps2 / divisor,
                           Path(args.output) / f"divisor_{divisor:g}", measure_rank=True)
        row = {"setting": "SiMCoding" if divisor == 1 else f"eps2/{divisor:g}", **row}
        rows.append(row)
        save_csv(Path(args.output) / "precision_sweep.csv", rows)


if __name__ == "__main__":
    main()
