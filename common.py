import argparse
from collections import namedtuple
import os
from typing import Optional
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.optimize as sciopt


def get_parser():
    parser = argparse.ArgumentParser(description="phase transition")

    # the involved models and optimization form
    parser.add_argument(
        "--planted", type=str, default="linear", choices=["linear", "plain", "normalized"], help="planted model"
    )
    parser.add_argument(
        "--learned",
        dest="model",
        type=str,
        default="skip",
        choices=["plain", "skip", "normalized"],
        help="learned model for recovery",
    )
    parser.add_argument(
        "--form",
        type=str,
        default="approx",
        choices=["gd", "exact", "approx", "relaxed"],
        help="whether to formulate optimization as convex program, GD (neural network training), or min norm (relaxed)",
    )

    # experiment parameters
    parser.add_argument("--n", type=int, default=400, help="number of sample")
    parser.add_argument("--d", type=int, default=100, help="number of dimension")
    parser.add_argument("--k", type=int, default=None, help="number of planted neurons")
    parser.add_argument("--sigma", type=float, default=0, help="noise")
    parser.add_argument(
        "--optw",
        type=int,
        default=None,  # depends on the planted model for theoretical analysis
        choices=[0, 1],
        help="choice of w. 0=Gaussian, 1=smallest PC of X",
    )
    parser.add_argument(
        "--optx",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="choice of X. 0=Gaussian, 1=Gaussian cubed, 2=whitened Gaussian, 3=whitened Gaussian cubed",
    )

    parser.add_argument("--seed", type=int, default=97006855, help="random seed")
    parser.add_argument("--sample", type=int, default=5, help="number of trials")

    # plotting and meta level
    parser.add_argument("--no_plot", action="store_true", help="avoid drawing plots")
    parser.add_argument("--verbose", action="store_true", help="whether to print information while training")
    parser.add_argument("--save_details", action="store_true", help="whether to save training results")
    parser.add_argument("--save_folder", type=str, default="./results/", help="path to save results")

    # nonconvex training
    parser.add_argument("--num_epoch", type=int, default=400, help="number of training epochs")
    parser.add_argument("--beta", type=float, default=1e-6, help="weight decay parameter")
    parser.add_argument("--lr", type=float, default=2e-3, help="learning rate")
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "sigmoid", "tanh", "gelu"],
        help="activation function",
    )

    return parser


def get_save_folder(args, n: int = None, d: int = None, sample: int = None):
    if n is None:
        n = args.n
    if d is None:
        d = args.d
    if sample is None:
        sample = args.sample

    return f"{args.save_folder}learned_{args.model}/planted_{args.planted}/form_{args.form}/trial__n{n}__d{d}__w{args.optw}__X{args.optx}__stdev{args.sigma}__sample{sample}/"


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


def generate_data(n: int, d: int, args, data: Optional[dict] = None, eps: Optional[float] = 1e-10):
    """
    Generate X (n x d), w (d x m), and y (n),
    where m is the number of hidden neurons,
    such that:
    - if model == "normalized", y[i] = sum_j ( relu( X[i] @ w[:, j] ) ).
    - if model == "skip", y = X @ w.
    Enforce that no column of y is all 0s.

    :param data: if not None, store the generated X, w, y in data
    :param eps: for checking if a column of y is close to 0
    :param only_X: if True, only generate X (used for generating test data for neural network training)
    """

    X = generate_X(n, d, cubic=args.optx in [1, 3], whiten=args.optx in [2, 3])
    w = generate_w(X, args.k, args.optw)

    try:
        y = generate_y(X, w, sigma=args.sigma, eps=eps, model=args.planted)
    except ValueError:
        # if the generated y has a column of all 0s, generate again
        return generate_data(n, d, args, data=data, eps=eps)

    if data is not None:
        data["X"] = X
        data["w"] = w
        data["y"] = y

    return X, w, y


def generate_X(n, d, cubic=False, whiten=False):
    """
    Generate an (n, d) design matrix X.
    See `get_parser` for the options for optx.
    """
    X = np.random.randn(n, d) / math.sqrt(n)
    if cubic:
        X = X**3
    if whiten:
        U, _S, Vh = np.linalg.svd(X, full_matrices=False)
        if n < d:
            X = Vh
        else:
            X = U
    return X


def generate_w(X, k, optw):
    """
    X is the (n x d) data matrix.
    k is the (optional) number of planted neurons.
    optw is the option for generating w.

    :return: w (d x k), where k is the number of hidden neurons.
    """
    d = X.shape[1]
    if k is not None:
        # involving multiple planted neurons
        if optw == 0:
            w = np.random.randn(d, k)
        elif optw == 1:
            w = np.eye(d, k)
        elif optw == 2:
            if k == 2:
                w = np.random.randn(d, 1)
                w = np.concatenate([w, -w], axis=1)
            else:
                raise TypeError("Invalid choice of planted neurons.")

    else:
        assert optw in [0, 1], "Invalid choice of planted neurons."
        if optw == 0:
            w = np.random.randn(d)
        elif optw == 1:
            _U, _S, Vh = np.linalg.svd(X, full_matrices=False)
            w = Vh[-1, :].T

    # the weights for each neuron should have unit norm
    w /= np.linalg.norm(w, axis=0)

    return w


def generate_y(X, w, sigma, eps=1e-10, model="linear"):
    """
    Generate y from the data X and the weights w.
    :param model: the planted model
    """
    if model == "normalized":
        y = np.maximum(0, X @ w)  # relu
        norm_y = np.linalg.norm(y, axis=0)
        if np.any(norm_y < eps):
            # if any columns are zero, re-generate
            raise ValueError("Some columns of y are zero.")
        y = np.sum(y / norm_y, axis=1)
    elif model == "plain":
        # assert False, str(w)
        y = np.sum(np.maximum(0, X @ w), axis=1)  # relu
    elif model == "linear":
        y = X @ w

    if sigma > 0:  # add noise
        n = X.shape[0]
        z = np.random.randn(n) * sigma / math.sqrt(n)
        y += z

    return y


def get_arrangement_patterns(X, w=None, n_sampled: Optional[int] = None):
    """
    Get "all" possible arrangement patterns of X.
    That is, consider a hyperplane passing through the origin.
    The corresponding "arrangement pattern" assigns 1 to all samples (of X) on the positive side of the hyperplane
    and 0 to all samples on the negative side.
    We approximate this by sampling n_sampled hyperplanes via Gaussian normal vectors.

    :param w: When the planted weights w (n, p) are given,
              the first k columns of D_mat are the arrangement patterns generated by the planted neurons.
    :param n_sampled: the number of sampled hyperplanes.
              We then form their corresponding arrangement patterns and take the unique ones.

    :return:
    - D_mat, a (n, p) matrix of the arrangement patterns
    - the indices into the original randomly generated normal vectors
    - a boolean returned by `check_degenerate_arr_pattern`
    """
    n, d = X.shape
    if n_sampled is None:
        n_sampled = max(n, 50)
    U1 = np.random.randn(d, n_sampled)
    if w is not None:
        U1 = np.concatenate([w, U1], axis=1)
    arr_patterns = X @ U1 >= 0

    # remove duplicates (since H should be a set). define p to be the number of unique patterns,
    # so D_mat is a (n x p) matrix.
    D_mat, ind = np.unique(arr_patterns, axis=1, return_index=True)
    exists_all_ones = check_degenerate_arr_pattern(X)
    if exists_all_ones:
        D_mat = np.concatenate([D_mat, np.ones((n, 1))], axis=1)
    return D_mat, ind, exists_all_ones


def idx_of_planted_in_patterns(ind, k=1):
    """
    Get the indices in D_mat corresponding to the k planted neurons.
    See the definition of `get_arrangement_patterns`:
    When the planted w is given, the first k columns of the randomly sampled hyperplanes
    are the arrangements corresponding to w.

    :return: j such that D_mat[:, j] is the arrangement pattern of the ith planted neuron.
    """
    return np.nonzero(ind == np.arange(k)[:, None])[1]


def mult_diag(D: np.ndarray, X: np.ndarray):
    """
    Multiply a diagonal matrix whose diagonal is D (n) with X (n x d).
    This is O(d*n), while constructing a diagonal matrix and multiplying is O(d*n^2).

    If D is a matrix whose columns are diagonals (n x k), then
    this function returns a 3-tensor (k, n, d) whose k-th element is the matrix np.diag(D[:, k]) @ X.
    """
    if D.ndim == 2:
        return D.T[:, :, None] * X
    return D.reshape(-1, 1) * X


def save_results(save_folder: str, records: dict):
    if not os.path.exists(save_folder):
        print("Creating folder: {}".format(save_folder))
        os.makedirs(save_folder)

    for prop, value in records.item():
        save_path = save_folder + prop
        np.save(save_path, value)
        print("Saved " + prop + " to " + save_path + ".npy")


def plot_results(records: dict, nvec, dvec, save_folder: str = None):
    xgrid, ygrid = np.meshgrid(nvec, dvec)
    fig, (ax,) = plt.subplots(1, len(records), figsize=(5 * len(records), 5), squeeze=False)
    for i, (prop, values) in enumerate(records.items()):
        grid = np.mean(values, axis=2).T

        # if plot phase transition for distance, use extend='max'
        cs = ax[i].contourf(xgrid, ygrid, grid, levels=np.linspace(0, 1), cmap="jet", extend="max")

        # if plot phase transition for probability
        # cs = ax.contourf(X, Y, Z, levels=np.arange(0,1.1,0.1), cmap=cm.jet)
        # if plot the boundary of success with probability 1
        # cs2 = ax.contour(X, Y, Z, levels=[0.9, 1], colors=('k',), linewidths=(2,))

        fig.colorbar(cs, ax=ax[i])
        ax[i].set_xlabel("n")
        ax[i].set_ylabel("d")
        ax[i].set_title(prop)

    if save_folder is not None:
        fig.savefig(save_folder + "figure.png")
    plt.show()
