import math
import cvxpy as cp
import numpy as np

from ReLU_normal.common_normal import get_loss
from common import get_arrangement_patterns, validate_data


def get_problem(X, y, dmat, beta):
    """
    Implement the convex optimization problem in cvxpy.
    Equation 26 in the paper, minus the skip connection.
    """

    n, d = X.shape
    p = dmat.shape[1]

    W1 = cp.Variable((d, p))
    W2 = cp.Variable((d, p))

    expr = cp.Variable(n)  # dummy variable for constructing objective

    constraints = []
    for i in range(p):
        di = dmat[:, i].reshape((n, 1))
        Xi = di * X
        Ui, S, Vh = np.linalg.svd(Xi, full_matrices=False)
        ri = np.linalg.matrix_rank(Xi)  # rank of Xi
        if ri == 0:
            constraints += [W1[:, i] == 0, W2[:, i] == 0]
        else:
            expr += Ui[:, np.arange(ri)] @ (W1[np.arange(ri), i] - W2[np.arange(ri), i])

            X1 = X @ Vh[np.arange(ri), :].T @ np.diag(1 / S[np.arange(ri)])
            X2 = (2 * di - 1) * X1

            constraints += [
                X2 @ W1[np.arange(ri), i] >= 0,
                X2 @ W2[np.arange(ri), i] >= 0,
            ]
            if ri < d:
                constraints += [
                    W1[np.arange(ri, d), i] == 0,
                    W2[np.arange(ri, d), i] == 0,
                ]

    # objective
    loss = cp.norm(expr - y, 2) ** 2
    regw = cp.mixed_norm(W1.T, 2, 1) + cp.mixed_norm(W2.T, 2, 1)
    obj = loss + beta * regw

    prob = cp.Problem(cp.Minimize(obj), constraints)

    return prob, {"W1": W1, "W2": W2}


def solve_problem(n, d, args):
    sigma, n_planted = args.sigma, args.neu
    data = {}  # empty dict
    X, w, y = validate_data(n, d, args)
    z = np.random.randn(n) * sigma / math.sqrt(n)  # add noise
    y = y + z
    data["X"] = X
    data["w"] = w
    data["y"] = y

    dmat, ind, data["exist_all_one"] = get_arrangement_patterns(X, w)

    prob, variables = get_problem(X, y, dmat, args.beta)

    # solve the problem
    param_dict = {}
    prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False, mosek_params=param_dict)

    w1 = variables["W1"].value
    w2 = variables["W2"].value

    data["dmat"] = dmat
    data["opt_w1"] = w1
    data["opt_w2"] = w2

    data["i_map"], data["dis_abs"], data["recovery"] = get_loss(
        n_planted, X, dmat, ind, w, w1, w2
    )

    return data
