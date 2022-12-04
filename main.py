from typing import Optional
import numpy as np
import os
from time import time
from common import get_parser
import pickle
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

from ReLU_skip.cvx_train_skip import solve_problem as solve_cvx_skip
from ReLU_skip.minnrm_skip import solve_problem as solve_minnorm_skip
from ReLU_normal.cvx_train_normal import solve_problem as solve_cvx_normalize
from ReLU_normal.minnrm_normal import solve_problem as solve_minnorm_normalize
from ReLU_normal.irr_cond_normal import check_irregular
from ncvx_network_train import train_model


def get_fname(
    args, n: Optional[int] = None, d: Optional[int] = None, sample: Optional[int] = None
):
    if n is None:
        n = args.n
    if d is None:
        d = args.d
    if sample is None:
        sample = args.sample

    return "{}_train_{}_n{}_d{}_w{}_X{}_sig{}_sample{}".format(
        args.form,
        args.model,
        n,
        d,
        args.optw,
        args.optx,
        args.sigma,
        sample,
    )


def get_record_properties(model: str, form: str):
    if model == "skip":
        if form == "nonconvex":
            return ["dis_abs", "test_err"]
        elif form == "convex":
            return ["dis_abs", "test_err"]
        elif form == "minnorm":
            return ["dis_abs", "test_err", "recovery"]

    if model == "normalize":
        if form == "nonconvex":
            return ["test_err"]
        elif form == "convex":
            return ["dis_abs", "recovery"]
        elif form == "irregular":
            return ["prob"]

    raise NotImplementedError("Invalid model and form combination.")


def main():
    """
    Three different kinds of models:
    - ReLU, ReLU-normalize, ReLU-skip.
    Each one can be phrased in different "form"s:
    - nonconvex neural network training, convex program, relazed min-norm program.
    """
    parser = get_parser(n_planted=None, optw=None)
    args = parser.parse_args()
    if args.model == "normalize":
        if args.neu is None:
            args.neu = 2
        if args.form == "min_norm" and args.neu is None:
            args.neu = 1
    elif args.model == "skip":
        if args.optw is None:
            args.optw = 1
    else:
        raise NotImplementedError("Invalid model type.")
    print(str(args))

    save_folder = args.save_folder + get_fname(args)
    seed = args.seed
    np.random.seed(seed)
    flag = args.save_details
    dvec = np.arange(10, args.d + 1, 10)
    nvec = np.arange(10, args.n + 1, 10)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    record_properties = get_record_properties(args.model, args.form)
    records = defaultdict(lambda: np.zeros((len(nvec), len(dvec), args.sample)))
    times = np.zeros(len(nvec))

    for nidx, n in tqdm(enumerate(nvec), position=0, leave=True):
        print("n = " + str(n))
        t0 = time()
        for didx, d in tqdm(enumerate(dvec), position=1, leave=False):
            if args.model == "normalize" and n < d:
                for prop in record_properties:
                    records[prop][nidx, didx, :] = None
                continue

            for i in tqdm(range(args.sample), position=2, leave=False):
                if args.model == "skip":
                    if args.form == "convex":
                        data = solve_cvx_skip(n, d, args)
                    elif args.form == "nonconvex":
                        data, model = train_model(n, d, args)
                    elif args.form == "minnorm":
                        data = solve_minnorm_skip(n, d, args)
                
                elif args.model == "normalize":
                    if args.form == "convex":
                        data = solve_cvx_normalize(n, d, args)
                    elif args.form == "nonconvex":
                        data, model = train_model(n, d, args)
                    elif args.form == "minnorm":
                        data = solve_minnorm_normalize(n, d, args)
                    elif args.form == "irregular":
                        records["prob"][nidx, didx, :] = check_irregular(n, d, args)

                for prop in record_properties:
                    records[prop][nidx, didx, i] = data[prop]

                if flag:
                    fname = get_fname(args, n, d, i)
                    with open(f"{save_folder}/n{n}_d{d}_sample{i}.pkl", "wb") as file:
                        pickle.dump(data, file)
                    if args.form == "nonconvex":
                        torch.save(
                            model.state_dict(), save_folder + model.name() + fname
                        )
        t1 = time()
        times[nidx] = t1 - t0

    print("done experiment. times = " + times.round(2).astype(str))

    xgrid, ygrid = np.meshgrid(nvec, dvec)
    fig, ax = plt.subplots(len(record_properties), 1, figsize=(5, 5 * len(record_properties)))
    for i, prop in enumerate(record_properties):
        save_path = save_folder + "/" + prop
        np.save(save_path, records[prop])
        print("Saved " + prop + " to " + save_path + ".npy")
        grid = np.mean(records[prop], axis=2).T

        # if plot phase transition for distance, use extend='max'
        cs = ax[i].contourf(xgrid, ygrid, grid, levels=np.linspace(0, 1), cmap="jet", extend="max")

        # if plot phase transition for probability
        # cs = ax.contourf(X, Y, Z, levels=np.arange(0,1.1,0.1), cmap=cm.jet)
        # if plot the boundary of success with probability 1
        # cs2 = ax.contour(X, Y, Z, levels=[0.9, 1], colors=('k',), linewidths=(2,))

        fig.colorbar(cs, ax=ax[i])
        ax[i].set_xlabel("n")
        ax[i].set_ylabel("d")
        ax[i].set_title(prop)

    fig.savefig(save_folder + "/figure.png")
    plt.show()


if __name__ == "__main__":
    main()
