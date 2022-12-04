import math
import cvxpy as cp
import numpy as np

from common import (
    generate_data,
    get_arrangement_patterns,
)


def get_problem(X, y, dmat, beta):
    """
    Convex formulation of skip network. See Equation 6 of the paper.
    """
    n, d = X.shape
    p = dmat.shape[1]

    W0 = cp.Variable(d)
    W_pos = cp.Variable((d, p))  # positive side
    W_neg = cp.Variable((d, p))  # negative side

    # constraints
    y_pos = cp.sum(cp.multiply(dmat, (X @ W_pos)), axis=1)
    y_neg = cp.sum(cp.multiply(dmat, (X @ W_neg)), axis=1)
    constraints = [
        cp.multiply((2 * dmat - np.ones((n, p))), (X @ W_pos)) >= 0,
        cp.multiply((2 * dmat - np.ones((n, p))), (X @ W_neg)) >= 0,
    ]

    # objective
    loss = cp.norm(X @ W0 + y_pos - y_neg - y, 2) ** 2
    regw = cp.norm(W0, 2) + cp.mixed_norm(W_pos.T, 2, 1) + cp.mixed_norm(W_neg.T, 2, 1)
    obj = loss + beta * regw

    prob = cp.Problem(cp.Minimize(obj), constraints)
    return prob, {"W0": W0, "W1": W_pos, "W2": W_neg}


def solve_problem(n, d, args):
    """
    Equation 6 of the paper
    """
    sigma = args.sigma

    data = {}  # empty dict
    X, w = generate_data(n, d, args)
    z = np.random.randn(n) * sigma / math.sqrt(n)
    y = X @ w + z
    data["X"] = X
    data["w"] = w
    data["y"] = y

    dmat, _ind, data["exist_all_one"] = get_arrangement_patterns(X)

    prob, variables = get_problem(X, y, dmat, args.beta)

    # solve the problem
    param_dict = {}
    prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False, mosek_params=param_dict)

    w0 = variables["W0"].value
    W_pos = variables["W1"].value
    W_neg = variables["W2"].value

    dis_abs = np.linalg.norm(w - w0)  # recovery error of linear weights

    # generate test data and calculate total test accuracy (MSE)
    X_test, z = generate_data(args)
    pos = np.maximum(0, X_test @ W_pos)
    neg = np.maximum(0, X_test @ W_neg)
    y_predict = np.sum(pos - neg, axis=1) + X_test @ w0
    test_err = np.linalg.norm(y_predict - X_test @ w)

    data["dmat"] = dmat
    data["opt_w0"] = w0
    data["opt_w1"] = W_pos
    data["opt_w2"] = W_neg

    data["dis_abs"] = dis_abs
    data["test_err"] = test_err
    return data
