import argparse
import math
import os
import pickle
from time import time

import cvxpy as cp
import numpy as np
import scipy.optimize as sciopt

from common import check_feasible, gen_data, get_parser


def solve_problem(args):
    n, d, sigma, neu = args.n, args.d, args.sigma, args.neu
    data = {}  # empty dict
    while True:
        X, w = gen_data(args)
        nrmw = np.linalg.norm(w, axis=0)
        w = w / nrmw
        y = np.maximum(0, X @ w)
        nrmy = np.linalg.norm(y, axis=0)
        if np.all(nrmy >= 1e-10):
            break
    y = np.sum(y / nrmy, axis=1)
    z = np.random.randn(n) * sigma / math.sqrt(n)
    y = y + z
    data["X"] = X
    data["w"] = w
    data["y"] = y

    mh = max(n, 50)
    U1 = np.concatenate([w, np.random.randn(d, mh)], axis=1)
    dmat = X @ U1 >= 0
    dmat, ind = np.unique(dmat, axis=1, return_index=True)
    if check_feasible(X):
        dmat = np.concatenate([dmat, np.ones((n, 1))], axis=1)
        data["exist_all_one"] = True
    else:
        data["exist_all_one"] = False

    # CVXPY variables
    m1 = dmat.shape[1]
    W = cp.Variable((d, m1))
    expr = np.zeros(n)
    constraints = []
    for i in range(m1):
        Xi = dmat[:, i].reshape((n, 1)) * X
        Ui, S, Vh = np.linalg.svd(Xi, full_matrices=False)
        ri = np.linalg.matrix_rank(Xi)
        if ri == d:
            expr += Ui @ W[:, i]
        elif ri == 0:
            constraints += [W[:, i] == 0]
        else:
            expr += Ui[:, np.arange(ri)] @ W[np.arange(ri), i]
            constraints += [W[np.arange(ri, d), i] == 0]

    obj = cp.mixed_norm(W.T, 2, 1)
    constraints += [expr == y]
    # solve the problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    param_dict = {}
    prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False, mosek_params=param_dict)

    optw = W.value
    data["i_map"] = np.zeros(neu)
    sum_square = 0
    for j in range(neu):
        k = np.nonzero(ind == j)[0][0]
        data["i_map"][j] = k
        wj = w[:, j]
        dj = dmat[:, k]
        Xj = dj.reshape((n, 1)) * X
        Uj, Sj, Vjh = np.linalg.svd(Xj, full_matrices=False)
        wj = (Sj.reshape((d, 1)) * Vjh) @ wj
        wj = wj / np.linalg.norm(wj)
        sum_square += np.linalg.norm(optw[:, k] - wj) ** 2
    dis1 = math.sqrt(sum_square)

    data["dmat"] = dmat
    data["opt_w"] = optw
    data["dis_abs"] = dis1
    return data


def main():
    parser = get_parser(neu=1)
    args = parser.parse_args()
    print(str(args))

    save_folder = args.save_folder
    seed = args.seed
    np.random.seed(seed)
    sigma = args.sigma
    sample = args.sample
    optw = args.optw
    optx = args.optx
    neu = args.neu
    flag = args.save_details
    dvec = np.arange(10, args.d + 1, 10)
    nvec = np.arange(10, args.n + 1, 10)
    dlen = dvec.size
    nlen = nvec.size

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    dis_abs = np.zeros((nlen, dlen, sample))

    for nidx, n in enumerate(nvec):
        print("n = " + str(n))
        t0 = time()
        for didx, d in enumerate(dvec):
            if n < d:
                dis_abs[nidx, didx, :] = None
                continue
            for i in range(sample):
                data = solve_problem(n, d, sigma, neu)
                dis_abs[nidx, didx, i] = data["dis_abs"]
                if flag:
                    fname = "minnrm_normal_n{}_d{}_w{}_X{}_sig{}_sample{}".format(
                        n, d, optw, optx, sigma, i
                    )
                    file = open(save_folder + fname + ".pkl", "wb")
                    pickle.dump(data, file)
                    file.close()

        t1 = time()
        print("time = " + str(t1 - t0))

    fname = "minnrm_n{}_d{}_w{}_X{}_sig{}_sample{}".format(
        args.n, args.d, optw, optx, sigma, sample
    )
    np.save(save_folder + "dis_abs_" + fname, dis_abs)


if __name__ == "__main__":
    main()
