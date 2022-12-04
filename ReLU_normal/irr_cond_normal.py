import os
from time import time

import numpy as np

from common import check_feasible, get_parser, validate_data


def check_irr(args):
    n, d, neu = args.n, args.d, args.neu
    eps = 1e-10
    X, w, _y = validate_data(args, eps=eps)

    mh = max(n * 2, 50)
    U1 = np.concatenate([w, np.random.randn(d, mh)], axis=1)
    dmat = X @ U1 >= 0
    dmat, ind = np.unique(dmat, axis=1, return_index=True)
    if check_feasible(X):
        dmat = np.concatenate([dmat, np.ones((n, 1))], axis=1)

    j_array = np.nonzero(ind <= neu - 1)[0]
    j_map = ind[j_array]

    U = np.zeros((n, 0))
    uu = []
    for jidx, j in enumerate(j_array):
        k = j_map[jidx]
        Xj = dmat[:, j].reshape((n, 1)) * X
        Uj, Sj, Vjh = np.linalg.svd(Xj, full_matrices=False)
        rj = np.linalg.matrix_rank(Xj)
        wj = (Sj.reshape((d, 1)) * Vjh) @ w[:, k]
        wj = wj / np.linalg.norm(wj)
        U = np.concatenate([U, Uj[:, np.arange(rj)]], axis=1)
        uu = np.concatenate([uu, wj[np.arange(rj)]])
    lam = U @ np.linalg.pinv(U.T @ U) @ uu

    m1 = dmat.shape[1]
    count = 0
    for j in range(m1):
        if j in j_array:
            continue
        dj = dmat[:, j]
        Xj = dj.reshape((n, 1)) * X
        Uj, Sj, Vjh = np.linalg.svd(Xj, full_matrices=False)
        if np.linalg.norm(Uj.T @ lam) >= 1 + eps:
            count += 1

    return count == 0


def main():
    parser = get_parser(neu=2, samples=50)
    args = parser.parse_args()
    print(str(args))

    save_folder = args.save_folder
    seed = args.seed
    np.random.seed(seed)
    sample = args.sample
    optw = args.optw
    optx = args.optx
    neu = args.neu
    dvec = np.arange(10, args.d + 1, 10)
    nvec = np.arange(10, args.n + 1, 10)
    dlen = dvec.size
    nlen = nvec.size

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    prob = np.zeros((nlen, dlen, sample))

    for nidx, n in enumerate(nvec):
        print("n = " + str(n))
        t0 = time()
        for didx, d in enumerate(dvec):
            if n < d:
                prob[nidx, didx, :] = False
                continue
            for i in range(sample):
                prob[nidx, didx, i] = check_irr(n, d, neu)

        t1 = time()
        print("time = " + str(t1 - t0))

    fname = "irr_cond_n{}_d{}_w{}_X{}_sample{}".format(
        args.n, args.d, optw, optx, sample
    )
    np.save(save_folder + fname, prob)
    print(np.mean(prob, axis=2))


if __name__ == "__main__":
    main()
