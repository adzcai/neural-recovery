import os
import pickle
from time import time
import torch.nn as nn

import cvxpy as cp
import numpy as np

from common import check_feasible, gen_data, get_parser


def solve_problem(args):
    data = {}  # empty dict
    n, d = args.n, args.d
    X, w = gen_data(n, d, args.optx, args.optw)
    y = X @ w

    data["X"] = X
    data["w"] = w
    data["y"] = y

    mh = max(n, 50)
    U1 = np.random.randn(d, mh)
    dmat = X @ U1 >= 0
    dmat, _ind = np.unique(dmat, axis=1, return_index=True)
    if check_feasible(X):
        dmat = np.concatenate([dmat, np.ones((n, 1))], axis=1)
        data["exist_all_one"] = True
    else:
        data["exist_all_one"] = False

    # CVXPY variables
    m1 = dmat.shape[1]
    W0 = cp.Variable((d,))
    W = cp.Variable((d, m1))
    obj = cp.norm(W0, 2) + cp.mixed_norm(W.T, 2, 1)  # we think this is the fro norm
    constraints = [cp.sum(cp.multiply(dmat, (X @ W)), axis=1) + X @ W0 == y]
    # other conditions from [6] seem to have been ignored

    # solve the problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    param_dict = {}
    prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False, mosek_params=param_dict)

    w0 = W0.value
    optw = W.value
    tol = 1e-4
    flag = np.allclose(w0, w, atol=tol) and np.allclose(optw, 0, atol=tol)

    data["dmat"] = dmat
    data["opt_w0"] = w0
    data["opt_w"] = optw
    data["rec"] = flag
    return data


def main():
    parser = get_parser(optw=1)
    args = parser.parse_args()
    print(str(args))

    save_folder = args.save_folder
    seed = args.seed
    np.random.seed(seed)
    sample = args.sample
    optw = args.optw
    optx = args.optx
    flag = args.save_details
    dvec = np.arange(10, args.d + 1, 10)
    nvec = np.arange(10, args.n + 1, 10)
    dlen = dvec.size
    nlen = nvec.size

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    rec = np.zeros((nlen, dlen, sample))

    for nidx, n in enumerate(nvec):
        print("n = " + str(n))
        t0 = time()
        for didx, d in enumerate(dvec):
            for i in range(sample):
                data = solve_problem(n, d)
                rec[nidx, didx, i] = data["rec"]
                if flag:
                    fname = "rec_rate_skip_n{}_d{}_w{}_X{}_sample{}".format(
                        n, d, optw, optx, i
                    )
                    file = open(save_folder + fname + ".pkl", "wb")
                    pickle.dump(data, file)
                    file.close()

        t1 = time()
        print("time = " + str(t1 - t0))

    fname = "rec_rate_skip_n{}_d{}_w{}_X{}_sample{}".format(
        args.n, args.d, optw, optx, sample
    )
    np.save(save_folder + fname, rec)
    print(np.mean(rec, axis=2))


if __name__ == "__main__":
    main()
