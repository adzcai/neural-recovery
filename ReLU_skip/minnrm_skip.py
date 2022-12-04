import argparse
import math
import os
import pickle
from time import time

import cvxpy as cp
import numpy as np

from common import check_feasible, gen_data, get_parser


def solve_problem(n, d, sigma):
    data = {}  # empty dict
    X, w = gen_data(n, d)
    z = np.random.randn(n) * sigma / math.sqrt(n)
    y = X @ w + z
    data["X"] = X
    data["w"] = w
    data["y"] = y

    mh = max(n, 50)
    U1 = np.random.randn(d, mh)
    dmat = X @ U1 >= 0
    dmat, ind = np.unique(dmat, axis=1, return_index=True)
    if check_feasible(X):
        dmat = np.concatenate([dmat, np.ones((n, 1))], axis=1)
        data["exist_all_one"] = True
    else:
        data["exist_all_one"] = False

    # CVXPY variables
    m1 = dmat.shape[1]
    W0 = cp.Variable((d,))
    W = cp.Variable((d, m1))
    obj = cp.norm(W0, 2) + cp.mixed_norm(W.T, 2, 1)
    constraints = [cp.sum(cp.multiply(dmat, (X @ W)), axis=1) + X @ W0 == y]
    # solve the problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    param_dict = {}
    prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False, mosek_params=param_dict)

    w0 = W0.value
    optw = W.value
    dis1 = np.linalg.norm(w - w0)
    X1, z = gen_data(n, d)
    y_predict = np.sum(np.maximum(0, X1 @ optw), axis=1) + X1 @ w0
    dis2 = np.linalg.norm(y_predict - X1 @ w)

    data["dmat"] = dmat
    data["opt_w0"] = w0
    data["opt_w"] = optw
    data["dis_abs"] = dis1
    data["dis_test"] = dis2
    return data


def main():
    parser = get_parser(optw=1)
    args = parser.parse_args()
    print(str(args))

    save_folder = args.save_folder
    seed = args.seed
    np.random.seed(seed)
    sigma = args.sigma
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

    dis_abs = np.zeros((nlen, dlen, sample))
    dis_test = np.zeros((nlen, dlen, sample))

    for nidx, n in enumerate(nvec):
        print("n = " + str(n))
        t0 = time()
        for didx, d in enumerate(dvec):
            for i in range(sample):
                data = solve_problem(n, d, sigma)
                dis_abs[nidx, didx, i] = data["dis_abs"]
                dis_test[nidx, didx, i] = data["dis_test"]
                if flag:
                    fname = "minnrm_skip_n{}_d{}_w{}_X{}_sig{}_sample{}".format(
                        n, d, optw, optx, sigma, i
                    )
                    file = open(save_folder + fname + ".pkl", "wb")
                    pickle.dump(data, file)
                    file.close()

        t1 = time()
        print("time = " + str(t1 - t0))

    fname = "minnrm_skip_n{}_d{}_w{}_X{}_sig{}_sample{}".format(
        args.n, args.d, optw, optx, sigma, sample
    )
    np.save(save_folder + "dis_abs_" + fname, dis_abs)
    np.save(save_folder + "dis_test_" + fname, dis_test)


if __name__ == "__main__":
    main()
