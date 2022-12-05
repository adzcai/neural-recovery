import numpy as np
import os
from time import time
from common import get_parser, get_record_properties, get_save_folder, plot_and_save
import pickle
from collections import defaultdict
from tqdm import tqdm

from ncvx_network_train import train_model
from solve_problem import solve_problem
from utils import check_irregular

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    if args.model == "normalize":
        if args.optw is None:
            args.optw = 0
        if args.k is None:
            if args.form == "relaxed":
                args.k = 1
            else:
                args.k = 2
    elif args.model == "skip":
        if args.optw is None:
            args.optw = 1
    elif args.model == "plain":
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
    seed = args.seed
    np.random.seed(seed)
    flag = args.save_details
    dvec = np.arange(10, args.d + 1, 10)
    nvec = np.arange(10, args.n + 1, 10)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    record_properties = get_record_properties(args.planted, args.model)
    print("Recording properties " + str(record_properties))
    records = defaultdict(lambda: np.zeros((len(nvec), len(dvec), args.sample)))
    runtimes = np.zeros(len(nvec))

    n_iter = enumerate(nvec)
    if args.verbose:
        n_iter = tqdm(n_iter, position=0, total=len(nvec), leave=True)
    for nidx, n in n_iter:
        t0 = time()
        d_iter = enumerate(dvec)
        if args.verbose:
            d_iter = tqdm(d_iter, position=1, total=len(dvec), leave=False)
        for didx, d in d_iter:
            if args.model == "normalize" and n < d:
                for prop in record_properties:
                    records[prop][nidx, didx, :] = None
                continue

            for i in range(args.sample):
                if args.form == "gd":
                    data = train_model(n, d, args)
                elif args.form == "irregular":
                    prob = check_irregular(n, d, args)
                    data = {"prob": prob}
                else:
                    data = solve_problem(n, d, args)


                for prop in record_properties:
                    records[prop][nidx, didx, i] = data[prop]

                if flag:
                    with open(f"{save_folder}/n{n}_d{d}_sample{i}.pkl", "wb") as file:
                        pickle.dump(data, file)
        t1 = time()
        runtimes[nidx] = t1 - t0

    print("done experiment. times = " + str(runtimes.round(2)))
    plot_and_save(save_folder, records, record_properties, nvec, dvec)


if __name__ == "__main__":
    main()
