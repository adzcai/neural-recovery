import math
import cvxpy as cp
import numpy as np

from common import generate_data, get_arrangement_patterns


def get_problem(X, y, dmat):
    d = X.shape[1]
    p = dmat.shape[1]

    W0 = cp.Variable((d,))  # skip connections
    W = cp.Variable((d, p))  # relu connections (second layer is all ones)

    obj = cp.norm(W0, 2) + cp.mixed_norm(W.T, 2, 1)
    # \sum_j D_j X W_j + X w_0 == y
    constraints = [cp.sum(cp.multiply(dmat, (X @ W)), axis=1) + X @ W0 == y]

    return obj, constraints, {"W0": W0, "W": W}


def solve_problem(n, d, args):
    data = {}  # empty dict
    X, w = generate_data(n, d, args)
    z = np.random.randn(n) * args.sigma / math.sqrt(n)
    y = X @ w + z
    data["X"] = X
    data["w"] = w
    data["y"] = y

    dmat, _ind, data["exist_all_one"] = get_arrangement_patterns(X)

    obj, constraints, variables = get_problem(X, y, dmat)

    # solve the problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False, mosek_params={})

    w0 = variables["W0"].value
    W = variables["W"].value
    dis1 = np.linalg.norm(w - w0)

    X1, z = generate_data(n, d)
    y_predict = np.sum(np.maximum(0, X1 @ W), axis=1) + X1 @ w0
    dis2 = np.linalg.norm(y_predict - X1 @ w)

    data["dmat"] = dmat
    data["opt_w0"] = w0
    data["opt_w"] = W
    data["dis_abs"] = dis1  # distance between true w and learned w0
    data["test_err"] = dis2
    return data
