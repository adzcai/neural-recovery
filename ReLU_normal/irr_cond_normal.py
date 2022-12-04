import numpy as np

from common import check_degenerate_arr_pattern, validate_data


def check_irregular(n, d, args):
    neu = args.neu
    eps = 1e-10
    X, w, _y = validate_data(args, eps=eps)

    mh = max(n * 2, 50)
    U1 = np.concatenate([w, np.random.randn(d, mh)], axis=1)
    dmat = X @ U1 >= 0
    dmat, ind = np.unique(dmat, axis=1, return_index=True)
    if check_degenerate_arr_pattern(X):
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
