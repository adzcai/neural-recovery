import math
import cvxpy as cp
import numpy as np

from ReLU_normal.common_normal import get_loss
from common import get_arrangement_patterns, validate_data


def get_problem(X, y, dmat):
    """
    Implement Equation 17 in the paper.
    After dropping all inequality constraints in equation 16 (implemented in `cvx_train_normal.py`),
    we're left with the group l1 norm minimization problem implemented below.
    """
    n, d = X.shape
    p = dmat.shape[1]

    W = cp.Variable((d, p))
    expr = cp.Variable(n)

    constraints = []
    for i in range(p):
        Xi = dmat[:, i].reshape((n, 1)) * X
        Ui, S, Vh = np.linalg.svd(Xi, full_matrices=False)
        ri = np.linalg.matrix_rank(Xi)
        if ri == d:
            expr += Ui @ W[:, i]
        elif ri == 0:
            constraints += [W[:, i] == 0]
        else:
            expr += Ui[:, np.arange(ri)] @ W[np.arange(ri), i]
            # TODO where does the below constraint appear in the paper?
            constraints += [W[np.arange(ri, d), i] == 0]
    constraints += [expr == y]

    obj = cp.mixed_norm(W.T, 2, 1)  # TODO is the transpose necessary?

    # solve the problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    return prob, {"W": W}


def solve_problem(n, d, args):
    sigma, neu = args.sigma, args.neu
    data = {}
    X, w, y = validate_data(n, d, args)
    z = np.random.randn(n) * sigma / math.sqrt(n)
    y = y + z
    data["X"] = X
    data["w"] = w
    data["y"] = y

    dmat, ind, data["exist_all_one"] = get_arrangement_patterns(X, w)

    prob, variables = get_problem(X, y, dmat)
    prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False, mosek_params={})

    W = variables["W"].value

    data["dmat"] = dmat
    data["opt_w"] = W

    data["i_map"], data["dis_abs"], data["recovery"] = get_loss(neu, X, dmat, ind, w, W)

    return data
