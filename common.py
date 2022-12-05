import argparse
from collections import namedtuple
from typing import Optional
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.optimize as sciopt


ProblemSetting = namedtuple("", [
    "planted_model",
    "learned_model",
    "formulation",
])

# problem settings for which recovery can be meaningfully defined.
valid_settings = (
    # ("linear", "plain"),
    ("linear", "skip"),
    # ("linear", "normalize"),

    ("relu_plain", "plain"),
    ("relu_plain", "skip"),
    # ("relu_plain", "normalize"),

    # ("relu_norm", "plain"),
    # ("relu_norm", "skip"),
    ("relu_norm", "normalize"),
)


def get_parser(k=None, samples=10, optw=0):
    parser = argparse.ArgumentParser(description="phase transition")
    parser.add_argument(
        "--form",
        type=str,
        default="convex",
        choices=["nonconvex", "convex", "relaxed"],
        help="whether to formulate optimization as convex program, nonconvex (neural network training), or min norm (relaxed)",
    )
    parser.add_argument("--n", type=int, default=400, help="number of sample")
    parser.add_argument("--d", type=int, default=100, help="number of dimension")
    parser.add_argument("--k", type=int, default=k, help="number of planted neurons")
    parser.add_argument("--seed", type=int, default=97006855, help="random seed")
    parser.add_argument("--sample", type=int, default=samples, help="number of trials")
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
    for model in ("plain", "skip", "normalize"):
        subparsers.add_parser(model)

    return parser


def get_record_properties(model: str, form: str):
    """
    What properties to record for each model / form combination.
    """
    if model == "plain":
        if form == "nonconvex":
            return ["dis_abs", "test_err"]
        elif form == "convex":
            return ["dis_abs", "test_err"]
        elif form == "relaxed":
            return ["dis_abs", "test_err"]

    if model == "skip":
        if form == "nonconvex":
            return ["dis_abs", "test_err"]
        elif form == "convex":
            return ["dis_abs", "test_err"]
        elif form == "relaxed":
            return ["dis_abs", "test_err", "recovery"]

    if model == "normalize":
        if form == "nonconvex":
            return ["test_err"]
        elif form == "convex" or form == "relaxed":
            return ["dis_abs", "recovery"]
        elif form == "irregular":
            return ["prob"]

    raise NotImplementedError("Invalid model and form combination.")


def get_fname(
    args, n: Optional[int] = None, d: Optional[int] = None, sample: Optional[int] = None
):
    if n is None:
        n = args.n
    if d is None:
        d = args.d
    if sample is None:
        sample = args.sample

    return "learned-{}/true-{}/form-{}/trial_n{}_d{}_w{}_X{}_sig{}_sample{}".format(
        args.model,
        args.form,
        n,
        d,
        args.optw,
        args.optx,
        args.sigma,
        sample,
    )


def get_save_folder(args):
    return args.save_folder + get_fname(args)


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


def get_arrangement_patterns(X, w=None, mh: Optional[int] = None):
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
    if mh is None:
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


def generate_data(
    n: int, d: int, args, data: Optional[dict] = None, eps: Optional[float] = 1e-10
):
    """
    Generate X (n x d), w (d x m), and y (n),
    where m is the number of hidden neurons,
    such that:
    - if model == "normalize", y[i] = sum_j ( relu( X[i] @ w[:, j] ) ).
    - if model == "skip", y = X @ w.
    Enforce that no column of y is all 0s.

    :param data: if not None, store the generated X, w, y in data
    :param eps: for checking if a column of y is close to 0
    :param only_X: if True, only generate X (used for generating test data for neural network training)
    """

    X = generate_X(n, d, args.optx)
    w = generate_w(X, args.k, args.optw)

    try:
        y = generate_y(
            X, w, sigma=args.sigma, eps=eps, model=default_planted_model(args)
        )
    except ValueError:
        # if the generated y has a column of all 0s, generate again
        return generate_data(n, d, args, data=data, eps=eps)

    if data is not None:
        data["X"] = X
        data["w"] = w
        data["y"] = y

    return X, w, y


def generate_X(n, d, optx):
    X = np.random.randn(n, d) / math.sqrt(n)
    if optx in [1, 3]:
        X = X**3
    if optx in [2, 4]:
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
    if model == "relu-norm":
        y = np.maximum(0, X @ w)  # relu
        norm_y = np.linalg.norm(y, axis=0)
        if np.any(norm_y < eps):
            # if any columns are zero, re-generate
            raise ValueError("Some columns of y are zero.")
        y = np.sum(y / norm_y, axis=1)
    elif model == "relu-no-norm":
        y = np.maximum(0, X @ w)  # relu
    elif model == "linear":
        y = X @ w

    if sigma > 0:  # add noise
        n = X.shape[0]
        z = np.random.randn(n) * sigma / math.sqrt(n)
        y += z

    return y


def default_planted_model(args):
    return "relu" if args.model == "normalize" else "linear"


def plot_and_save(save_folder, records, record_properties, nvec, dvec):
    xgrid, ygrid = np.meshgrid(nvec, dvec)
    fig, ax = plt.subplots(
        1, len(record_properties), figsize=(5 * len(record_properties), 5)
    )
    for i, prop in enumerate(record_properties):
        save_path = save_folder + "/" + prop
        np.save(save_path, records[prop])
        print("Saved " + prop + " to " + save_path + ".npy")
        grid = np.mean(records[prop], axis=2).T

        # if plot phase transition for distance, use extend='max'
        cs = ax[i].contourf(
            xgrid, ygrid, grid, levels=np.linspace(0, 1), cmap="jet", extend="max"
        )

        # if plot phase transition for probability
        # cs = ax.contourf(X, Y, Z, levels=np.arange(0,1.1,0.1), cmap=cm.jet)
        # if plot the boundary of success with probability 1
        # cs2 = ax.contour(X, Y, Z, levels=[0.9, 1], colors=('k',), linewidths=(2,))

        fig.colorbar(cs, ax=ax[i])
        ax[i].set_xlabel("n")
        ax[i].set_ylabel("d")
        ax[i].set_title(prop)

    fig.savefig(save_folder + "/figure.png")
    plt.show()
