from collections import OrderedDict
import cvxpy as cp
import numpy as np


def cvx_relu(X, y, dmat, beta, skip=False, exact=False):
    """
    Convex formulation of either plain relu network or a network with skip connection when skip is True.

    if not exact and skip, this is 211
    if not exact and not skip, this is modified 211
    if exact and skip, this is 6, with the w_0 norm added (we think this was a typo)
    if exact and not skip, this is 11
    """
    n, d = X.shape
    p = dmat.shape[1]

    if skip:
        W0 = cp.Variable(d)
    W_pos = cp.Variable((d, p))  # positive side
    W_neg = cp.Variable((d, p))  # negative side

    # constraints
    y_pos = cp.sum(cp.multiply(dmat, (X @ W_pos)), axis=1)
    y_neg = cp.sum(cp.multiply(dmat, (X @ W_neg)), axis=1)
    signed_patterns = 2 * dmat - np.ones((n, p))
    constraints = [
        cp.multiply(signed_patterns, (X @ W_pos)) >= 0,
        cp.multiply(signed_patterns, (X @ W_neg)) >= 0,
    ]

    y_hat = y_pos - y_neg
    norm = cp.mixed_norm(W_pos.T, 2, 1) + cp.mixed_norm(W_neg.T, 2, 1)
    if skip:
        y_hat += X @ W0
        norm += cp.norm(W0, 2)

    if exact:
        constraints += [y_hat == y]
        obj = norm
    else:
        obj = cp.norm(y_hat - y, 2) ** 2 + beta * norm

    prob = cp.Problem(cp.Minimize(obj), constraints)

    if skip:
        return prob, OrderedDict(W0=W0, W1=W_pos, W2=W_neg)
    else:
        return prob, OrderedDict(W1=W_pos, W2=W_neg)


def cvx_relu_relax(X, y, dmat, _beta, skip=False):
    """
    Equation 15 of the paper,
    a relaxation (by dropping inequality constraints) of Equation 6,
    which is implemented in `cvx_train_skip.py`.
    """
    d = X.shape[1]
    p = dmat.shape[1]

    W0 = cp.Variable((d,))  # skip connections
    W = cp.Variable((d, p))  # relu connections (second layer is all ones)

    obj = cp.norm(W0, 2) + cp.mixed_norm(W.T, 2, 1)
    # \sum_j D_j X W_j + X w_0 == y
    y_hat = cp.sum(cp.multiply(dmat, (X @ W)), axis=1)
    if skip:
        y_hat += X @ W0
    constraints = [y_hat == y]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    return prob, OrderedDict(W0=W0, W=W)


def cvx_relu_normalized(X, y, D_mat, beta):
    """
    Implement the convex optimization problem in cvxpy.
    Equation 27 in the paper, minus the skip connection.
    This is equivalent to the regularized training of a two-layer relu network with normalization.
    """

    n, d = X.shape
    p = D_mat.shape[1]

    W_pos = cp.Variable((d, p))
    W_neg = cp.Variable((d, p))

    y_hat = cp.Variable(n)  # dummy variable for constructing objective

    constraints = []
    for i in range(p):
        di = D_mat[:, i].reshape((n, 1))
        Xi = di * X
        Ui, S, Vh = np.linalg.svd(Xi, full_matrices=False)
        ri = np.linalg.matrix_rank(Xi)  # rank of Xi
        if ri == 0:
            constraints += [W_pos[:, i] == 0, W_neg[:, i] == 0]
        else:
            y_hat += Ui[:, np.arange(ri)] @ (W_pos[np.arange(ri), i] - W_neg[np.arange(ri), i])

            X1 = X @ Vh[np.arange(ri), :].T @ np.diag(1 / S[np.arange(ri)])
            signed_pattern = (2 * di - 1) * X1

            constraints += [
                signed_pattern @ W_pos[np.arange(ri), i] >= 0,
                signed_pattern @ W_neg[np.arange(ri), i] >= 0,
            ]
            if ri < d:
                constraints += [
                    W_pos[np.arange(ri, d), i] == 0,
                    W_neg[np.arange(ri, d), i] == 0,
                ]

    # objective
    loss = cp.norm(y_hat - y, p=2) ** 2
    norm = cp.mixed_norm(W_pos.T, 2, 1) + cp.mixed_norm(W_neg.T, 2, 1)
    obj = loss + beta * norm

    prob = cp.Problem(cp.Minimize(obj), constraints)
    return prob, OrderedDict(W_pos=W_pos, W_neg=W_neg)


def cvx_relu_normalized_relax(X, y, D_mat, _beta):
    """
    Implement Equation 17 in the paper.
    After dropping all inequality constraints in Equation 16 (the equivalent convex program for a normalized relu network),
    we're left with the group l1 norm minimization problem implemented below.
    """
    n, d = X.shape
    p = D_mat.shape[1]

    W = cp.Variable((d, p))
    expr = cp.Variable(n)

    # constraints
    constraints = []
    for i in range(p):
        Xi = D_mat[:, i].reshape((n, 1)) * X
        Ui, _Si, _Vhi = np.linalg.svd(Xi, full_matrices=False)
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
    return prob, OrderedDict(W=W)
