import os
import pickle
from time import time

import cvxpy as cp
import numpy as np

from common import get_arr_patterns, get_parser, validate_data


def solve_problem(args):
    n, d, neu = args.n, args.d, args.neu

    data = {}
    X, w, y = validate_data(args)
    data["X"] = X
    data["w"] = w
    data["y"] = y

    dmat, ind, data["exist_all_one"] = get_arr_patterns(X, n, d, w)

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
    data["dmat"] = dmat
    data["opt_w"] = optw
    data["i_map"] = np.zeros(neu)
    for j in range(neu):
        k = np.nonzero(ind == j)[0][0]
        data["i_map"][j] = k
        wj = w[:, j]
        dj = dmat[:, k]
        Xj = dj.reshape((n, 1)) * X
        Uj, Sj, Vjh = np.linalg.svd(Xj, full_matrices=False)
        wj = (Sj.reshape((d, 1)) * Vjh) @ wj
        wj = wj / np.linalg.norm(wj)
        optw[:, k] -= wj

    tol = 1e-4
    data["rec"] = np.allclose(optw, 0, atol=tol)
    return data


def main():
    parser = get_parser(neu=1)
    args = parser.parse_args()
    print(str(args))

    save_folder = args.save_folder
    seed = args.seed
    np.random.seed(seed)
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

    rec = np.zeros((nlen, dlen, sample))

    for nidx, n in enumerate(nvec):
        print("n = " + str(n))
        t0 = time()
        for didx, d in enumerate(dvec):
            if n < d:
                rec[nidx, didx, :] = False
                continue
            for i in range(sample):
                data = solve_problem(n, d, neu)
                rec[nidx, didx, i] = data["rec"]
                if flag:
                    fname = "rec_rate_normal_n{}_d{}_w{}_X{}_sample{}".format(
                        n, d, optw, optx, i
                    )
                    file = open(save_folder + fname + ".pkl", "wb")
                    pickle.dump(data, file)
                    file.close()

        t1 = time()
        print("time = " + str(t1 - t0))

    fname = "rec_rate_n{}_d{}_w{}_X{}_sample{}".format(
        args.n, args.d, optw, optx, sample
    )
    np.save(save_folder + fname, rec)


if __name__ == "__main__":
    main()
