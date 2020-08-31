import argparse

import torch

from benchmark.benchmark import Benchmark


def main(args):
    torch.set_num_threads(args.num_threads)

    benchmark = Benchmark.from_json(args.run_args_file)
    df = benchmark.run(verbose=args.verbose)
    if args.verbose:
        print(df)
    df.to_csv(args.csv_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_args_file", help="Path to the args json file", type=str)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--num_threads", help="Number of threads", type=int)
    parser.add_argument("--csv_file", help="Results filename", type=str)
    args = parser.parse_args()

    main(args)
