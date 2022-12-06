"""
This file contains the metrics used to evaluate the performance of the learned models.
In general, functions expect the following arguments in the given order:
- args: the program arguments
- X: (n, d) the data matrix
- D_mat: (n, p) the arrangement patterns
- ind: (p) the indices of the arrangement patterns in the original random set
- w: (d, k) the planted weights
- w_pos: (d, p) the learned (positive) weights
- w_neg: (d, p) the learned negative weights
- w_skip: (d) the learned skip weights
- atol: the tolerance to use for checking if a learned neuron matches the grounded truth.
"""

from typing import Optional
import numpy as np

from common import generate_X, generate_y, idx_of_planted_in_patterns, mult_diag
from cvx_problems import Variables


def get_metrics(args, X, w, D_mat, ind, variables):
    model = args.learned
    if model == "plain":
        return get_metrics_plain(X, w, variables)
    if model == "normalized":
        return get_metrics_normalized(X, w, D_mat, ind, variables)
    elif model == "skip":
        return get_metrics_skip(w, variables)
    else:
        raise NotImplementedError("Unknown model: %s" % model)


def get_metrics_normalized(
    X: np.ndarray,
    W_true: np.ndarray,
    D_mat: np.ndarray,
    ind: np.ndarray,
    variables: Variables,
    atol=1e-4,
):
    """
    Planted: ReLU
    Learned: ReLU+normalize
    Formulation: approx formulation:
    - either W_pos is (d, p) for the relaxed formulation,
    - or W_pos and W_neg are both (d, p) for the approximate formulation

    Get metrics for the normalized ReLU model.
    Intuitively, measures the distance to the planted neurons' weights.
    Mathematically, we express the jth planted neuron in terms of the singular values of Dj @ X,
    where Dj is the corresponding diagonal arrangement pattern,
    and then compare the learned neurons with this expression.

    :param w: (d, k) planted weights of the ReLU
    :param w_pos: (n, p) the learned (positive) weights
    :param w_pos: (n, p) the learned negative weights
    :param tol: the tolerance to use for checking if a learned neuron matches the grounded truth.
    """
    k = W_true.shape[1]

    W_pos = variables.W_pos
    W_neg = variables.W_neg if variables.W_neg is not None else np.zeros_like(W_pos)

    s = idx_of_planted_in_patterns(ind, k)  # element of [p]^k. the indices of the planted neurons in D_mat
    DX = mult_diag(D_mat[:, s], X)  # (k, n, d)
    _U, S, Vh = np.linalg.svd(DX, full_matrices=False)  # (k, n, d), (k, d), (k, d, d). assumes d < n
    PC = S[:, :, None] * Vh  # (k, d, d). Scale the right singular vectors by the singular values.
    w_rotated = np.einsum("kmd, dk -> mk", PC, W_true)  # (d, k). take elementwise product and transpose
    w_rotated /= np.linalg.norm(w_rotated, axis=0)
    diff = W_pos[:, s] - W_neg[:, s] - w_rotated  # all have shape (d, k)
    dis_abs = np.linalg.norm(diff, ord="fro")
    recovery = np.allclose(diff, 0, atol=atol)

    return {
        "dis_abs": dis_abs,
        "recovery": recovery,
    }


def get_metrics_skip(w_true, variables: Variables, atol=1e-4):
    """
    Planted: Linear
    Learned: ReLU+skip
    Formulation: approx formulation:
    - either W_pos is (d, p) for the relaxed formulation,
    - or W_pos and W_neg are both (d, p) for the approximate formulation

    For underlying linear weights w,
    we define "recovery" as the following:
    the learned weights w_skip are close to w,
    and all of the learned weights (for the relu) W are close to 0.
    """

    w_skip, W_pos, W_neg = variables
    dis_abs = np.linalg.norm(w_true - w_skip)  # recovery error of linear weights

    recovery = np.allclose(w_skip, w_true, atol=atol) and np.allclose(W_pos, 0, atol=atol)
    if W_neg is not None:
        recovery = recovery and np.allclose(W_neg, 0, atol=atol)

    return {
        "dis_abs": dis_abs,
        "recovery": recovery,
    }


def get_metrics_plain(X, W_true, variables: Variables):
    """
    Planted: ReLU_plain
    Learned: ReLU_plain

    W_true: (d, k)
    W_pos: (d, p)
    W_neg: (d, p)
    """

    _, W_pos, W_neg = variables

    dis_abs = 0  # np.linalg.norm(W_true - W_pos)  # recovery error of linear weights

    recovery = True
    for neuron in W_true.T:
        recovery = recovery and np.any([np.allclose(neuron / np.linalg.norm(neuron), w_pos) for w_pos in W_pos.T])
        # recovery = recovery and np.any([np.allclose(0, w_neg) for w_neg in W_neg.T])

    # print the highest 10 norms of columns of W_pos
    # print("w_true", W_true)
    # print("y", y_true)
    # print("W_pos norms", np.sort(np.linalg.norm(W_pos, axis=0))[-10:])

    return {
        "dis_abs": dis_abs,
        "recovery": recovery,
    }


def get_test_err(n, d, optx, planted, learned, W_true, variables: Variables):
    # generate test data and calculate total test accuracy (MSE)
    X_test = generate_X(n, d, cubic=optx in [1, 3], whiten=optx in [2, 3])
    y_true = generate_y(X_test, Variables(W_pos=W_true), relu=planted != "linear", normalize=planted == "normalized")
    y_hat = generate_y(X_test, variables, relu=True, normalize=learned == "normalized")
    test_err = np.linalg.norm(y_hat - y_true)
    return test_err
