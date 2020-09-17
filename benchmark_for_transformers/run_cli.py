from argparse import ArgumentParser

import torch

from benchmark_for_transformers.benchmark import Benchmark


def main():
    parser = ArgumentParser()
    parser.add_argument("--run_args_file", help="Path to the args json file", type=str)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--num_threads", help="Number of threads", type=int, default=1)
    parser.add_argument("--csv_file", help="Results filename", type=str, default="results.csv")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)

    benchmark = Benchmark.from_json(args.run_args_file)
    df = benchmark.run(verbose=args.verbose)
    if args.verbose:
        print(df)
    df.to_csv(args.csv_file)


if __name__ == "__main__":
    main()
