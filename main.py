import numpy as np
import os
from time import time
from common import get_parser, get_save_folder, plot_results, save_results
import pickle
from collections import defaultdict
from tqdm import tqdm

from ncvx_network_train import train_model
from solve_problem import solve_problem
from utils import check_irregular


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    if args.learned == "normalized":
        if args.optw is None:
            args.optw = 0
        if args.k is None:
            if args.form == "relaxed":
                args.k = 1
            else:
                args.k = 2
    elif args.learned == "skip":
        if args.optw is None:
            args.optw = 1
    elif args.learned == "plain":
        if args.optw is None:
            args.optw = 0
    else:
        raise NotImplementedError("Invalid model type.")

    return args


def main():
    """
    Three different kinds of models:
    - ReLU, ReLU-normalize, ReLU-skip.
    Each one can be phrased in different "form"s:
    - nonconvex neural network training (gradient descent), convex program, relazed min-norm program.
    """
    args = get_args()
    print(str(args))

    save_folder = get_save_folder(args)
    np.random.seed(args.seed)
    save_weights = args.save_details
    dvec = np.arange(10, args.d + 1, 10)
    nvec = np.arange(10, args.n + 1, 10)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    records = defaultdict(lambda: np.empty((len(nvec), len(dvec), args.sample)))
    runtimes = np.zeros(len(nvec))

    n_iter = enumerate(nvec)
    if not args.quiet:
        n_iter = tqdm(n_iter, position=0, total=len(nvec), leave=True)
    for nidx, n in n_iter:
        t0 = time()
        d_iter = enumerate(dvec)
        if not args.quiet:
            d_iter = tqdm(d_iter, position=1, total=len(dvec), leave=False)
        for didx, d in d_iter:
            if args.learned == "normalized" and n < d:
                continue

            for i in range(args.sample):
                if args.form == "gd":
                    data, metrics = train_model(n, d, args)
                elif args.form == "irregular":
                    prob = check_irregular(n, d, args)
                    data, metrics = {"prob": prob}
                else:
                    data, metrics = solve_problem(n, d, args)

                for prop, value in metrics.items():
                    records[prop][nidx, didx, i] = value

                if save_weights:
                    with open(f"{save_folder}n{n}_d{d}_sample{i}.pkl", "wb") as file:
                        pickle.dump(data, file)
        t1 = time()
        runtimes[nidx] = t1 - t0

    print("done experiment. times = " + str(runtimes.round(2)))
    save_results(save_folder, records)
    plot_results(records, nvec, dvec, save_folder=save_folder)


if __name__ == "__main__":
    main()
