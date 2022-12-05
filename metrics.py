from typing import Optional
import numpy as np

from common import generate_X, generate_data


def get_metrics(program_args, *args, **kwargs):
    model, form = program_args.model, program_args.form
    if model == "plain":
        return get_metrics_skip(program_args, *args, **kwargs)
    if model == "normalize":
        return get_metrics_normalize(program_args, *args, **kwargs)
    elif model == "skip":
        return get_metrics_skip(program_args, *args, **kwargs)
    else:
        raise NotImplementedError("Unknown model: %s" % model)


def get_metrics_normalize(
    args,
    X: np.ndarray,
    dmat: np.ndarray,
    ind: np.ndarray,
    w: np.ndarray,
    w_pos: np.ndarray,
    w_neg: Optional[np.ndarray] = None,
    atol=1e-4,
):
    """
    Get metrics for the normalized ReLU model.
    Intuitively, measures the distance to the planted neurons' weights.
    Mathematically, we express the jth planted neuron in terms of the singular values of Dj @ X,
    where Dj is the corresponding diagonal arrangement pattern,
    and then compare the learned neurons with this expression.

    :param w: the planted weights
    :param w1: the learned (positive) weights
    :param w2: the learned negative weights
    :param tol: the tolerance to use for checking if a learned neuron matches the grounded truth.
    """
    p = w.shape[1]
    i_map = np.zeros(p)

    n, d = X.shape

    distances = np.copy(w_pos)
    if w_neg is not None:
        distances -= w_neg

    for j in range(p):  # for each of the planted neurons
        # get the smallest index k such that ind[k] = j
        # and therefore dmat[:, k] gives the jth vector in the _original_ dmat (before picking unique columns)
        # TODO is this always included in the new dmat though? why does this need to be included?
        k = np.nonzero(ind == j)[0][0]
        i_map[j] = k
        wj = w[:, j]
        dj = dmat[:, k]
        Xj = dj.reshape((n, 1)) * X
        _Uj, Sj, Vjh = np.linalg.svd(Xj, full_matrices=False)
        # scale the right singular vectors according to their singular values,
        # then rephrase the jth planted neuron in this basis
        wj = (Sj.reshape((d, 1)) * Vjh) @ wj
        wj /= np.linalg.norm(wj)

        # get distance to this planted neuron
        distances[:, k] -= wj

    dis_abs = np.linalg.norm(distances, ord="fro")
    recovery = np.allclose(distances, 0, atol=atol)

    return {
        "i_map": i_map,
        "dis_abs": dis_abs,
        "recovery": recovery,
    }


def get_metrics_skip(
    args, X, _dmat, _ind, w_true, w_skip, W_pos, W_neg=None, atol=1e-4
):
    """
    For underlying linear weights w,
    we define "recovery" as the following:
    the learned weights w_skip are close to w,
    and all of the learned weights (for the relu) W are close to 0.

    This either accepts a single W_pos for the relaxed convex formulation,
    or a W_pos and W_neg for the exact or approximate formulation.
    """
    n, d = X.shape

    dis_abs = np.linalg.norm(w_true - w_skip)  # recovery error of linear weights

    # generate test data and calculate total test accuracy (MSE)
    X_test = generate_X(n, d, args.optx)
    if W_neg is not None:
        pos = np.maximum(0, X_test @ W_pos)  # relu
        neg = np.maximum(0, X_test @ W_neg)  # relu
        outputs = pos - neg
    else:
        outputs = np.maximum(0, X_test @ W_pos)

    y_predict = np.sum(outputs, axis=1) + X_test @ w_skip
    test_err = np.linalg.norm(y_predict - X_test @ w_true)

    recovery = np.allclose(w_skip, w_true, atol=atol) and np.allclose(
        W_pos, 0, atol=atol
    )
    if W_neg is not None:
        recovery = recovery and np.allclose(W_neg, 0, atol=atol)

    return {
        "dis_abs": dis_abs,
        "test_err": test_err,
        "recovery": recovery,
    }
