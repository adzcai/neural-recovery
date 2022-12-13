from collections import namedtuple
from typing import Optional
import math
import numpy as np
import scipy.optimize as sciopt

from utils import Args

Variables = namedtuple("Variables", ["W_pos", "W_neg", "w_skip"], defaults=[None] * 3)


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


def generate_data(n: int, d: int, args: Args, data: Optional[dict] = None, eps: float = None):
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

    X = generate_X(n, d, cubic=args.cubic, whiten=args.whiten)
    W_true = generate_w(X, args.k, args.optw)

    try:
        y = generate_y(
            X,
            Variables(W_pos=W_true),
            sigma=args.sigma,
            eps=eps,
            relu=args.planted != "linear",
            normalize=args.planted == "normalized",
        )
    except ValueError:
        # if the generated y has a column of all 0s, generate again
        return generate_data(n, d, args, data=data, eps=eps)

    if data is not None:
        data["X"] = X
        data["W"] = W_true
        data["y"] = y

    return X, W_true, y


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
    X is the (n, d) data matrix.
    k is the (optional) number of planted neurons.
    optw is the option for generating w.

    :return: w (d, k), where k is the number of hidden neurons.
    """
    d = X.shape[1]
    # involving multiple planted neurons
    if optw == 0:
        w = np.random.randn(d, k)
    elif optw == 1:
        _U, _S, Vh = np.linalg.svd(X, full_matrices=False)
        w = Vh[-k:, :].T
    elif optw == 2:
        w = np.eye(d, k)
    elif optw == 3 and k == 2:
        w = np.random.randn(d, 1)
        w = np.hstack([w, -w])
    else:
        raise TypeError("Invalid choice of planted neurons.")

    # the weights for each neuron should have unit norm
    w /= np.linalg.norm(w, axis=0)

    return w


def generate_y(X, weights: Variables, relu=False, normalize=False, sigma=0, eps: float = None):
    """
    Generally uses the provided weights to calculate y from the given X and w.
    Also used to generate y from the data X and the weights w.

    :param sigma: the standard deviation of the noise.
    :param eps: for checking if a column of y is close to 0
    :return: y (n)
    """

    W_pos, W_neg, w_skip = weights

    assert W_pos.ndim == 2, f"{W_pos.shape=} should be a 2D array."

    if W_neg is not None:
        if relu:
            y = np.maximum(0, X @ W_pos) - np.maximum(0, X @ W_neg)
        else:
            y = X @ (
                W_pos - W_neg
            )  # this scenario should never occur but is implemented for completeness
    else:
        if relu:
            y = np.maximum(0, X @ W_pos)
        else:
            y = X @ W_pos

    if normalize:
        norm_y = np.linalg.norm(y, axis=0)
        if eps is not None and np.any(norm_y < eps):
            # if any columns are zero, re-generate
            raise ValueError("Some columns of y are zero.")
        y /= norm_y

    y = np.sum(y, axis=1)

    if w_skip is not None:
        y += X @ w_skip

    if sigma > 0:  # add noise
        n = X.shape[0]
        z = np.random.randn(n) * sigma / math.sqrt(n)
        y += z

    return y


def get_arrangement_patterns(X, w: np.ndarray = None, p_hat: int = None):
    """
    Get "all" possible arrangement patterns of X.
    That is, consider a hyperplane passing through the origin.
    The corresponding "arrangement pattern" assigns 1 to all samples (of X) on the positive side of the hyperplane
    and 0 to all samples on the negative side.
    We approximate this by sampling p_hat hyperplanes via Gaussian normal vectors and then taking the unique generated patterns.

    :param w: When the planted weights w (n, p) are given,
              the first k columns of D_mat are the arrangement patterns generated by the planted neurons.
    :param p_hat: the number of sampled hyperplanes.
                  We then form their corresponding arrangement patterns and take the unique ones.

    :return:
    - D_mat, a (n, p) matrix of the arrangement patterns
    - the indices into the original randomly generated normal vectors
    - a boolean returned by `check_degenerate_arr_pattern`, in which case a vector of 1s is appended to D_mat
    """
    n, d = X.shape
    p_hat = max(n, 50, p_hat) if p_hat is not None else max(n, 50)

    U1 = np.random.randn(d, p_hat)
    if w is not None:
        U1 = np.hstack([w, U1])
    arr_patterns = X @ U1 >= 0

    # remove duplicates (since H should be a set). define p to be the number of unique patterns,
    # so D_mat is a (n x p) matrix.
    D_mat, ind = np.unique(arr_patterns, axis=1, return_index=True)
    exists_all_ones = check_degenerate_arr_pattern(X)
    if exists_all_ones:
        D_mat = np.hstack([D_mat, np.ones((n, 1))])
    return D_mat, ind, exists_all_ones


def idx_of_planted_in_patterns(ind, k=1, mask: int = None):
    """
    Get the indices in D_mat (n, p) corresponding to the k planted neurons.
    See the definition of `get_arrangement_patterns`:
    When the planted w is given, the first k columns of the randomly sampled hyperplanes
    are the arrangements corresponding to w.

    ind (p,) is the injection from D_mat to the original random hyperplanes.

    :return: s (k,) such that D_mat[:, s] gives the arrangement patterns of the k planted neurons.

    if mask is an integer, also returns a mask of shape (mask,) with the elements corresponding to s set to true.
    """
    indices = np.nonzero(ind == np.arange(k)[:, None])[1]
    if mask is not None:
        return indices == np.arange(mask)
    else:
        return indices


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
