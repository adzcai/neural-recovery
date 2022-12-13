"""
Neural Isometry Conditions proposed in the paper.
"""

import argparse
import numpy as np
from tqdm import tqdm

from training.common import (
    generate_X,
    generate_w,
    get_arrangement_patterns,
    idx_of_planted_in_patterns,
    mult_diag,
    plot_results,
)


def check_irrepresentable(n, d, args):
    eps = 1e-10
    X = generate_X(n, d, args.optx)
    w = generate_w(X, args.k, args.optw)
    dmat, ind, _i_map = get_arrangement_patterns(X, w, p_hat=max(n * 2, 50))

    j_array = np.nonzero(ind <= args.k - 1)[0]
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


def RIP(X, w, dmat):
    """
    Restricted Isometry Property
    """
    pass


def NIC_linear(X, w, D_mat, _ind):
    """
    Planted: Linear model
    Learned: ReLU+skip (p neurons + 1 skip)

    Definition 2 in the paper (middle of p. 5).
    If this holds, then optimum of ReLU+skip is where the skip weights are set to the planted linear model,
    and all ReLU weights are set to zero.
    """
    rhs = np.linalg.solve(X.T @ X, w)
    conditions = np.einsum("nd, np, nf, fk -> pd", X, D_mat, X, rhs, optimize=True)
    conditions = np.linalg.norm(conditions, axis=1)
    return np.all(conditions < 1), conditions


def NIC_1(X, w, D_mat, ind):
    """
    Planted: plain ReLU (1 neuron)
    Learned: plain ReLU (p neurons), i.e. Equation 11.

    Definition 3 in the paper (bottom of p. 9).
    If this holds, then optimum of learned is a single neuron learns the optimal weights and all others are set to zero.
    """

    p = D_mat.shape[1]
    s = idx_of_planted_in_patterns(ind, mask=p)
    X_scaled = mult_diag(D_mat, X)
    rhs = np.linalg.solve(X.T @ X_scaled[s], w.T)
    conditions = np.einsum("pnd, knf, kf -> pd", X_scaled[~s], X_scaled[s], rhs, optimize=True)
    conditions = np.linalg.norm(conditions, axis=1)
    return np.all(conditions < 1), conditions


def NIC_normalized(X, w, D_mat, ind):
    """
    Planted: ReLU+norm (1 neuron)
    Learned: ReLU+norm (p neurons)
    Definition 4 in the paper (top of p 10).
    If this holds, then optimum of ReLU+norm is the planted neuron, and all other weights set to zero.
    """
    s = idx_of_planted_in_patterns(ind).item()
    U, S, Vh = np.linalg.svd(mult_diag(D_mat, X), full_matrices=False)
    U_other = np.delete(U, s, axis=0)
    w = (S[s, :, None] * Vh[s]) @ w
    w /= np.linalg.norm(w)
    conditions = np.einsum("pnd, nf, fk -> pd", U_other, U[s], w, optimize=True)
    conditions = np.linalg.norm(conditions, axis=1)
    return np.all(conditions < 1), conditions


def NIC_k(X, w, D_mat, ind):
    """
    Definition 5 in the paper (bottom of p 10).
    If this condition holds, then the unique solution to Equation 11,
    i.e. convex formulation of a plain ReLU NN,
    is to learn the planted neurons and set all others to zero.
    """

    n, d = X.shape
    p = D_mat.shape[1]
    k = w.shape[1]

    s = idx_of_planted_in_patterns(ind, k, mask=p)  # k
    X = mult_diag(D_mat, X)  # (p, n, d)

    pseudoinv = np.linalg.pinv(X[s])

    conditions = np.einsum("pnd, kfn, fk -> pd", X[~s], pseudoinv, w, optimize=True)
    conditions = np.linalg.norm(conditions, axis=0)
    return np.all(conditions < 1), conditions


def NIC_k_normalized(X, w, dmat):
    """
    Definition 6 in the paper.
    If this condition holds, then the unique solution to Equation 16,
    i.e. convex formulation of a normalized ReLU NN,
    is to learn the planted neurons and set all others to zero.

    w is a (d x k) matrix of the weights.
    dmat is a (n x p) matrix of the arrangement patterns.
    """

    p = dmat.shape[1]

    # S = []
    # for i,  in enumerate(dmat.T):

    for j in range(p):
        D = np.diag(dmat[:, j])
        U, S, Vh = np.linalg.svd(D)

        norm = np.linalg.norm(X.T @ D @ np.linalg.inv(X.T @ X) @ w)
        if norm > 1:
            return False


conditions = [NIC_linear, NIC_normalized, NIC_1, NIC_k]


def plot_condition():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cond",
        type=str,
        choices=[fn.__name__ for fn in conditions],
        help="Condition to plot",
    )
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--optx", type=int, default=0)
    parser.add_argument("--optw", type=int, default=0)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    nvec = np.arange(10, args.n + 1, 10)
    dvec = np.arange(10, args.d + 1, 10)

    fn = [x for x in conditions if x.__name__ == args.cond][0]

    def run_trial(n, d):
        X = generate_X(n, d, args.optx)
        w = generate_w(X, args.k, args.optw)
        D_mat, ind, _all_ones = get_arrangement_patterns(X)
        return fn(X, w, D_mat, ind)

    flags_ary = np.empty((len(nvec), len(dvec), args.samples))
    norms_ary = np.empty((len(nvec), len(dvec), args.samples, max(args.n, 50) + 1))
    for i, n in enumerate(tqdm(nvec, desc="n samples", position=0, leave=True)):
        for j, d in enumerate(tqdm(dvec, desc="d dim", position=1, leave=False)):
            for k in range(args.samples):
                flag, norms = run_trial(n, d)
                flags_ary[i, j, k] = flag
                norms_ary[i, j, k, : len(norms)] = norms

    save_folder = (
        f"./results/nic/{args.cond}__n{args.n}__d{args.d}__k{args.k}__X{args.optx}__w{args.optw}/"
    )
    plot_results({args.cond: flags_ary}, nvec, dvec, save_folder=save_folder)


if __name__ == "__main__":
    plot_condition()
