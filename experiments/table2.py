"""CIFAR-20 label-noise experiment: SiMCoding results only."""
import argparse
from pathlib import Path
from simcoding.common import save_csv
from simcoding.train import add_training_arguments, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, "cifar20")
    parser.add_argument("--eps2", type=float, default=38.2, help="Epsilon squared.")
    parser.add_argument("--noise-ratios", type=float, nargs="+", default=[0, .1, .2, .3, .4, .5])
    parser.add_argument("--exclude-original-label", action="store_true",
                        help="Force selected labels to change; default follows original uniform resampling.")
    parser.add_argument("--output", default="outputs/table2")
    args = parser.parse_args()
    if any(not 0 <= ratio <= 1 for ratio in args.noise_ratios) or len(set(args.noise_ratios)) != len(args.noise_ratios):
        parser.error("noise-ratios must be unique and between 0 and 1")
    rows = []
    for ratio in args.noise_ratios:
        row = run_training(args, "cifar20", args.eps2,
                           Path(args.output) / f"noise_{ratio:g}", noise=ratio)
        rows.append(row)
        save_csv(Path(args.output) / "table2.csv", rows)


if __name__ == "__main__":
    main()
