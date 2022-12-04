from typing import Optional
import numpy as np
import os
from time import time
from common import get_parser
import pickle
import torch
from collections import defaultdict

from ReLU_skip.cvx_train_skip import solve_problem as solve_cvx_skip
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

    save_folder = args.save_folder
    seed = args.seed
    np.random.seed(seed)
    flag = args.save_details
    dvec = np.arange(10, args.d + 1, 10)
    nvec = np.arange(10, args.n + 1, 10)
    dlen = dvec.size
    nlen = nvec.size

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    record_properties = get_record_properties(args.model, args.form)
    records = defaultdict(lambda: np.zeros((nlen, dlen, args.sample)))

    for nidx, n in enumerate(nvec):
        print("n = " + str(n))
        t0 = time()
        for didx, d in enumerate(dvec):
            if args.model == "normalize" and n < d:
                for prop in record_properties:
                    records[prop][nidx, didx, :] = None
                continue

            for i in range(args.sample):
                if args.form == "convex":
                    if args.mode == "skip":
                        data = solve_cvx_skip(n, d, args)
                    elif args.mode == "normalize":
                        data = solve_cvx_normalize(n, d, args)
                elif args.form == "nonconvex":
                    data, model = train_model(n, d, args)
                elif args.form == "minnorm":
                    if args.mode == "normalize":
                        data = solve_minnorm_normalize(n, d, args)
                elif args.form == "irregular":
                    records["prob"][nidx, didx, :] = check_irregular(n, d, args)

                for prop in record_properties:
                    records[prop][nidx, didx, i] = data[prop]

                if flag:
                    fname = get_fname(args, n, d, i)
                    with open(save_folder + fname + ".pkl", "wb") as file:
                        pickle.dump(data, file)
                    if args.form == "nonconvex":
                        torch.save(
                            model.state_dict(), save_folder + model.name() + fname
                        )
        t1 = time()
        print("time = " + str(t1 - t0))

    fname = get_fname(args)
    for prop in record_properties:
        np.save(save_folder + prop + "_" + fname, records[prop])


if __name__ == "__main__":
    main()
