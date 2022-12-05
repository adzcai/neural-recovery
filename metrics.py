from typing import Optional
import numpy as np

from common import generate_X, idx_of_planted_in_patterns, mult_diag


def get_metrics(program_args, *args):
    model, form, planted = program_args.model, program_args.form, program_args.planted
    if model == "plain":
        if planted == "plain":
            return get_metrics_plain_approx(program_args, *args)
        else:
            return get_metrics_skip(program_args, *args)
    if model == "normalized":
        return get_metrics_normalized(program_args, *args)
    elif model == "skip":
        return get_metrics_skip(program_args, *args)
    else:
        raise NotImplementedError("Unknown model: %s" % model)


def get_metrics_normalized(
    args,
    X: np.ndarray,
    D_mat: np.ndarray,
    ind: np.ndarray,
    w: np.ndarray,
    w_pos: np.ndarray,
    w_neg: Optional[np.ndarray] = None,
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
    k = w.shape[1]

    if w_neg is None:
        w_neg = np.zeros_like(w_pos)

    s = idx_of_planted_in_patterns(ind, k)  # element of [p]^k. the indices of the planted neurons in D_mat
    DX = mult_diag(D_mat[s], X)  # (k, n, d)
    _U, S, Vh = np.linalg.svd(DX, full_matrices=False)  # (k, n, m), (k, m), (k, m, d) where m = min(n, d)
    PC = S[:, :, None] * Vh  # (k, m, d). Scale the right singular vectors by the singular values.
    w_rotated = np.einsum("kmd,dk->mk", PC, w)  # take elementwise product and transpose
    w_rotated /= np.linalg.norm(w_rotated, axis=0)
    diff = (w_pos - w_neg)[s] - w_rotated
    dis_abs = np.linalg.norm(diff, ord="fro")
    recovery = np.allclose(diff, 0, atol=atol)

    return {
        "i_map": s,
        "dis_abs": dis_abs,
        "recovery": recovery,
    }


def get_metrics_skip(args, X, _dmat, _ind, w_true, w_skip, W_pos, W_neg=None, atol=1e-4):
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
    n, d = X.shape

    dis_abs = np.linalg.norm(w_true - w_skip)  # recovery error of linear weights

    # generate test data and calculate total test accuracy (MSE)
    X_test = generate_X(n, d, cubic=args.optx in [1, 3], whiten=args.optx in [2, 3])
    if W_neg is not None:
        pos = np.maximum(0, X_test @ W_pos)  # relu
        neg = np.maximum(0, X_test @ W_neg)  # relu
        outputs = pos - neg
    else:
        outputs = np.maximum(0, X_test @ W_pos)

    y_predict = np.sum(outputs, axis=1) + X_test @ w_skip
    test_err = np.linalg.norm(y_predict - X_test @ w_true)

    recovery = np.allclose(w_skip, w_true, atol=atol) and np.allclose(W_pos, 0, atol=atol)
    if W_neg is not None:
        recovery = recovery and np.allclose(W_neg, 0, atol=atol)

    return {
        "dis_abs": dis_abs,
        "test_err": test_err,
        "recovery": recovery,
    }


def get_metrics_plain_approx(args, X, _dmat, _ind, W_true, W_pos, W_neg=None):
    """
    Planted: ReLU_plain
    Learned: ReLU_plain
    Formulation: approx formulation

    W_true: (d, k)
    W_pos: (d, p)
    W_neg: (d, p)
    """

    n, d = X.shape

    dis_abs = 0  # np.linalg.norm(W_true - W_pos)  # recovery error of linear weights

    # generate test data and calculate total test accuracy (MSE)
    X_test = generate_X(n, d, args.optx)
    if W_neg is not None:
        pos = np.maximum(0, X_test @ W_pos)  # relu
        neg = np.maximum(0, X_test @ W_neg)  # relu
        outputs = pos - neg
    else:
        outputs = np.maximum(0, X_test @ W_pos)

    y_predict = np.sum(outputs, axis=1)
    y_true = np.sum(np.maximum(0, X_test @ W_true), axis=1)
    test_err = np.linalg.norm(y_predict - y_true)

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
        "test_err": test_err,
        "recovery": recovery,
    }
