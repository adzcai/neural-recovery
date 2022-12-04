import numpy as np
import os
from time import time
from common import get_parser, get_save_folder
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

from ncvx_network_train import train_model
from solve_problem import solve_problem
from utils import check_irregular


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
        elif form == "convex" or form == "minnorm":
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
        if args.optw is None:
            args.optw = 0
        if args.neu is None:
            if args.form == "minnorm":
                args.neu = 1
            else:
                args.neu = 2
    elif args.model == "skip":
        if args.optw is None:
            args.optw = 1
    else:
        raise NotImplementedError("Invalid model type.")
    print(str(args))

    save_folder = get_save_folder(args)
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

    for nidx, n in tqdm(enumerate(nvec), position=0, total=len(nvec), leave=True):
        t0 = time()
        for didx, d in tqdm(enumerate(dvec), position=1, total=len(dvec), leave=False):
            if args.model == "normalize" and n < d:
                for prop in record_properties:
                    records[prop][nidx, didx, :] = None
                continue

            for i in range(args.sample):
                if args.form == "nonconvex":
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
        times[nidx] = t1 - t0

    print("done experiment. times = " + str(times.round(2)))

    xgrid, ygrid = np.meshgrid(nvec, dvec)
    fig, ax = plt.subplots(
        len(record_properties), 1, figsize=(5, 5 * len(record_properties))
    )
    for i, prop in enumerate(record_properties):
        save_path = save_folder + "/" + prop
        np.save(save_path, records[prop])
        print("Saved " + prop + " to " + save_path + ".npy")
        grid = np.mean(records[prop], axis=2).T

        # if plot phase transition for distance, use extend='max'
        cs = ax[i].contourf(
            xgrid, ygrid, grid, levels=np.linspace(0, 1), cmap="jet", extend="max"
        )

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
