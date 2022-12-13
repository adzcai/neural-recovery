import numpy as np
from time import time
import pickle
from collections import defaultdict
from tqdm import tqdm

from utils import Args, check_folder, save_results
from training.noncvx_network_train import train_model
from plot import plot_results
from solve_problem import solve_problem


def main():
    """
    Three different kinds of models:
    - ReLU, ReLU-normalize, ReLU-skip.
    Each one can be phrased in different "form"s:
    - nonconvex neural network training (gradient descent), convex program, relazed min-norm program.
    """
    args = Args.parse()

    save_folder = args.get_save_folder()
    np.random.seed(args.seed)
    save_weights = args.save_details

    dvec = np.arange(10, args.d + 1, 10)
    nvec = np.arange(10, args.n + 1, 10)

    check_folder(save_folder)

    records = defaultdict(lambda: np.empty((len(nvec), len(dvec), args.sample)))
    runtimes = np.zeros(len(nvec))

    n_iter = enumerate(nvec)
    if not args.quiet:
        n_iter = tqdm(n_iter, desc="n samples", position=0, total=len(nvec), leave=True)
    for nidx, n in n_iter:
        t0 = time()
        d_iter = enumerate(dvec)
        if not args.quiet:
            d_iter = tqdm(d_iter, desc="d dim", position=1, total=len(dvec), leave=False)
        for didx, d in d_iter:
            if args.learned == "normalized" and n < d:
                continue

            for i in range(args.sample):
                if args.form == "gd":
                    data, metrics = train_model(n, d, i, args)
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
    save_results(records, save_folder=save_folder)
    plot_results(records, nvec, dvec, cmap=args.cmap, save_folder=save_folder)


if __name__ == "__main__":
    main()
