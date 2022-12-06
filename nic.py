"""
Neural Isometry Conditions proposed in the paper.
"""

import argparse
import numpy as np
from tqdm import tqdm

from common import generate_X, generate_w, get_arrangement_patterns, idx_of_planted_in_patterns, mult_diag, plot_results

# def irrepresentable(X, w, dmat):
#     """

#     """
#     pass


def NIC_linear(X, w, D_mat, _ind):
    """
    Planted: Linear model
    Learned: ReLU+skip (p neurons + 1 skip)
    Definition 2 in the paper (middle of p 5).
    If this holds, then optimum of ReLU+skip is where the skip weights are set to the planted linear model, and all ReLU weights are set to zero.
    """
    n, d = X.shape
    assert w.shape == (d, 1), f"Expected {w.shape=} to be {(d, 1)=}"
    DX = mult_diag(D_mat, X) @ np.linalg.inv(X.T @ X) @ w  # (p, n, d) @ (d, d) @ (d, 1) = (p, n, 1)
    conditions = np.einsum("p n i, n d -> d p", DX, X)
    conditions /= np.linalg.norm(conditions, axis=0)
    return np.all(conditions < 1), conditions


def NIC_normalized(X, w, D_mat, ind):
    """
    Planted: ReLU+norm (1 neuron)
    Learned: ReLU+norm (p neurons)
    Definition 4 in the paper (top of p 10).
    If this holds, then optimum of ReLU+norm is the planted neuron, and all other weights set to zero.
    """
    n, d = X.shape
    assert w.shape == (d, 1), f"Expected {w.shape=} to be {(d, 1)=}"
    s = idx_of_planted_in_patterns(ind).item()
    U_opt, S_opt, Vh_opt = np.linalg.svd(D_mat[s] @ X, full_matrices=False)
    w = np.diag(S_opt) @ Vh_opt @ w
    w /= np.linalg.norm(w)

    for D in D_mat.T:
        if np.allclose(D, np.diagonal(D_opt)):
            continue
        U, _S, _Vh = np.linalg.svd(np.diag(D) @ X, full_matrices=False)
        term = U.T @ U_opt @ w
        if np.linalg.norm(term) >= 1:
            return False
    return True


def RIP(X, w, dmat):
    """
    Restricted Isometry Property
    """
    pass


def NIC_1(X, w, dmat):
    """
    Definition 3 in the paper.
    Suppose the planted model is a single ReLU neuron,
    such that y = relu(X @ w).
    If this holds, then the unique solution to Equation 11,
    i.e. convex formulation of a plain ReLU NN,
    is the single planted ReLU neuron.
    That is, a single neuron learns the optimal weights and all others are set to zero.
    """

    D_opt = (X @ w) >= 0
    XD_opt = D_opt.reshape(-1, 1) * X
    for D in dmat.T:
        if np.allclose(D, D_opt):
            continue
        XD_j = D.reshape(-1, 1) * X
        norm = (XD_j.T @ XD_opt) @ np.linalg.inv(X.T @ XD_opt) @ w
        if np.linalg.norm(norm) >= 1:
            return False
    return True


def NIC_k(X, w, D_mat, ind):
    """
    Definition 5 in the paper (bottom of p 10).
    If this condition holds, then the unique solution to Equation 11,
    i.e. convex formulation of a plain ReLU NN,
    is to learn the planted neurons and set all others to zero.
    """

    p = D_mat.shape[1]
    k = w.shape[1]

    s = idx_of_planted_in_patterns(ind, k)  # k
    XD = mult_diag(D_mat, X)  # (p, n, d)
    middle = np.einsum("k n d, d k -> n", XD[s], w)
    other_idx = np.setdiff1d(np.arange(p), s)
    conditions = np.einsum("m n d, n -> d m", XD[other_idx], middle)  # (d, p - k). Each column is a condition.
    conditions = np.linalg.norm(conditions, axis=0)  # p - k values

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
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--optx", type=int, default=0)
    parser.add_argument("--optw", type=int, default=0)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--cond", type=str, choices=[fn.__name__ for fn in conditions], default=0)
    args = parser.parse_args()

    nvec = np.arange(10, args.n + 1, 10)
    dvec = np.arange(10, args.d + 1, 10)

    fn = [x for x in conditions if x.__name__ == args.cond][0]

    def run_trial(n, d):
        X = generate_X(n, d, args.optx)
        w = generate_w(X, args.k, args.optw)
        dmat = get_arrangement_patterns(X)[0]
        return fn(X, w, dmat)

    records = {}
    records[args.cond] = np.array(
        [
            [[run_trial(n, d) for _ in range(args.samples)] for d in tqdm(dvec, position=1, leave=False)]
            for n in tqdm(nvec, position=0, leave=True)
        ]
    )

    save_folder = "./results/nic/{}__n{}__d{}__k{}__X{}__w{}".format(
        args.cond, args.n, args.d, args.k, args.optx, args.optw
    )
    plot_results(save_folder, records, [args.cond], nvec, dvec)


if __name__ == "__main__":
    plot_condition()
