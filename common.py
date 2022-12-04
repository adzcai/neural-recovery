import argparse
import torch
import math
import numpy as np
import scipy.optimize as sciopt


def get_parser(n_planted=None, samples=10, optw=0):
    parser = argparse.ArgumentParser(description="phase transition")
    parser.add_argument(
        "--form",
        type=str,
        default="convex",
        choices=["nonconvex", "convex", "minnorm"],
        help="whether to formulate optimization as convex program, nonconvex (neural network training), or min norm (relaxed)",
    )
    parser.add_argument("--n", type=int, default=400, help="number of sample")
    parser.add_argument("--d", type=int, default=100, help="number of dimension")
    parser.add_argument("--seed", type=int, default=97006855, help="random seed")
    parser.add_argument("--sample", type=int, default=samples, help="number of trials")
    parser.add_argument(
        "--neu", type=int, default=n_planted, help="number of planted neurons"
    )
    parser.add_argument("--plot", action="store_true", help="draw plots")

    parser.add_argument("--optw", type=int, default=optw, help="choice of w")
    # 0: randomly generated (Gaussian)
    # 1: smallest right eigenvector of X
    # 2: randomly generated ReLU network

    parser.add_argument("--optx", type=int, default=0, help="choice of X")
    # 0: Gaussian
    # 1: cubic Gaussian
    # 2: 0 + whitened
    # 3: 1 + whitened

    parser.add_argument("--sigma", type=float, default=0, help="noise")
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="whether to print information while training",
    )
    parser.add_argument(
        "--save_details",
        type=bool,
        default=False,
        help="whether to save training results",
    )
    parser.add_argument(
        "--save_folder", type=str, default="./results/", help="path to save results"
    )

    # nonconvex training
    parser.add_argument(
        "--num_epoch", type=int, default=400, help="number of training epochs"
    )
    parser.add_argument(
        "--beta", type=float, default=1e-6, help="weight decay parameter"
    )
    parser.add_argument("--lr", type=float, default=2e-3, help="learning rate")

    subparsers = parser.add_subparsers(
        required=True, dest="model", description="learned model for recovery"
    )
    for model in ("basic", "skip", "normalize"):
        subparsers.add_parser(model)

    return parser


def validate_data(n, d, args, eps=1e-10):
    """
    We don't want any neuron to return all 0s across the dataset
    """
    while True:
        X, w = generate_data(n, d, args)
        w /= np.linalg.norm(w, axis=0)
        y = np.maximum(0, X @ w)
        norm_y = np.linalg.norm(y, axis=0)
        if np.all(norm_y >= eps):
            break
    y = np.sum(y / norm_y, axis=1)
    return X, w, y


def check_degenerate_arr_pattern(X):
    """
    Check if there exists some hyperplane passing through the origin
    such that all elements of X lie on on side of the hyperplane,
    in which case we need to add the identity matrix to the set of arrangement patterns.
    """
    n, d = X.shape
    nrm = lambda x: np.linalg.norm(x, ord=2)
    x0 = np.random.randn(d)
    x0 = x0 / np.linalg.norm(x0)
    lc = sciopt.LinearConstraint(X, 0, np.inf)
    nlc = sciopt.NonlinearConstraint(nrm, 0, 1)
    res = sciopt.minimize(lambda x: -nrm(x), x0, constraints=[lc, nlc])
    if -res.fun <= 1e-6:
        return False  # no all-one arrangement
    else:
        return True  # exist all-one arrangement


def get_arrangement_patterns(X, w=None):
    """
    Get all possible arrangement patterns of X.
    That is, consider the set of hyperplanes passing through the origin with normal vector h.
    An "arrangement pattern" assigns 1 to all samples on the positive side of the hyperplane
    and 0 to all samples on the negative side.
    :return:
    - a (n x p) matrix of the arrangement patterns
    - the indices in the original randomly generated h
    - see `check_feasible`
    """
    n, d = X.shape
    mh = max(n, 50)
    U1 = np.random.randn(d, mh)
    if w is not None:
        U1 = np.concatenate([w, U1], axis=1)
    arr_patterns = X @ U1 >= 0
    arr_patterns, ind = np.unique(arr_patterns, axis=1, return_index=True)
    exists_all_ones = check_degenerate_arr_pattern(X)
    if exists_all_ones:
        arr_patterns = np.concatenate([arr_patterns, np.ones((n, 1))], axis=1)
    return arr_patterns, ind, exists_all_ones


def generate_data(n: int, d: int, args):
    optw, optx = args.optw, args.optx
    X = np.random.randn(n, d) / math.sqrt(n)
    if optx in [1, 3]:
        X = X**3
    if optx in [2, 4]:
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        if n < d:
            X = Vh
        else:
            X = U

    if args.neu is not None:
        # involving multiple planted neurons
        if optw == 0:
            w = np.random.randn(d, args.neu)
        elif optw == 1:
            w = np.eye(d, args.neu)
        elif optw == 2:
            if args.neu == 2:
                w = np.random.randn(d, 1)
                w = np.concatenate([w, -w], axis=1)
            else:
                raise TypeError("Invalid choice of planted neurons.")
    else:
        if optw == 0:
            w = np.random.randn(d)
            w = w / np.linalg.norm(w)
        elif optw == 1:
            U, S, Vh = np.linalg.svd(X, full_matrices=False)
            w = Vh[-1, :].T
        elif optw == 2:
            assert args.m is not None, "must specify number of hidden neurons m"

    return X, w
