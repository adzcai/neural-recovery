import cvxpy as cp
import numpy as np

from common import generate_data, get_arrangement_patterns
from cvx_problems import (
    cvx_relu_normalize,
    cvx_relu_normalize_relax,
    cvx_relu,
    cvx_relu_skip_relax,
)
from metrics import get_metrics

def NIC_plain_plain(X, w, dmat):
    dmat = np.array([np.diag(d) for d in dmat.T])
    # print("shapes", X.shape, w.shape, y.shape, dmat.shape)
    Dsi = np.array([np.diag(neuron) for neuron in ((X @ w) >= 0).T]) # (k, n, n)
    assert dmat.shape[1] == Dsi.shape[1] and dmat.shape[2] == Dsi.shape[2]
    S = []
    for d in Dsi:
        for i, other_d in enumerate(dmat):
            if np.allclose(d, other_d):
                S.append(i)
                break

    middle_prod = np.einsum("d n, k n m -> k d m", X.T, Dsi)
    middle = np.concatenate(middle_prod, axis=0).T
    w_hat = w / np.linalg.norm(w, axis=0)
    right = np.concatenate(w_hat.T, axis=0)
    NIC_holds = True
    for i, d in enumerate(dmat):
        if i not in S:
            NIC_holds = NIC_holds and np.linalg.norm(X.T @ d @ middle @ right) < 1
    return NIC_holds


def NIC_linear(X, w, dmat):
    dmat = np.array([np.diag(d) for d in dmat.T])
    NIC_holds = True
    w_hat = w / np.linalg.norm(w, axis=0)
    for d in dmat:
        NIC_holds = NIC_holds and np.linalg.norm(X.T @ d @ X @ np.linalg.inv(X.T @ X) @ w_hat) < 1
    return NIC_holds

# def NIC_normalized(X, w, dmat):
#     w = w.squeeze()
#     di = (X @ w) >= 0
#     dj = [np.diag(d) for d in dmat.T if not np.allclose(d, di)]
#     di = np.diag(di)
#     ui, sigi, vi = np.linalg.svd(di @ X)
#     uj, _, _ = np.linalg.svd(dj @ X)
#     w_hat = (np.diag(sigi) @ vi @ w) / np.linalg.norm((np.diag(sigi) @ vi @ w), axis=0)
#     np.einsum("p n m, n d, ")


def solve_problem(n, d, args):
    data = {}
    X, w, y = generate_data(n, d, args, data=data)

    mh = max(50, 2 * n if args.form == "irregular" else n)
    dmat, ind, data["exist_all_one"] = get_arrangement_patterns(
        X, w if args.model == "normalize" else None, mh=mh
    )
    
    if args.model == "plain":
        data["NIC_holds"] = NIC(X, w, dmat)
        if args.form == "approx":
            prob, variables = cvx_relu(X, y, dmat, args.beta, skip=False)
        elif args.form == 'exact':
            prob, variables = cvx_relu(X, y, dmat, args.beta, skip=False, exact=True)
        # else:
        #     prob, variables = cvx_relu_skip_relax(X, y, dmat, args.beta)
    elif args.model == "skip":
        if args.form == "approx":
            prob, variables = cvx_relu(X, y, dmat, args.beta, skip=True)
        elif args.form == "exact":
            prob, variables = cvx_relu(X, y, dmat, args.beta, skip=True, exact=True)    
        elif args.form == "relaxed":
            prob, variables = cvx_relu_skip_relax(X, y, dmat, args.beta)
        else:
            assert False
    elif args.model == "normalize":
        if args.form == "approx":
            prob, variables = cvx_relu_normalize(X, y, dmat, args.beta)
        elif args.form == "relaxed":
            prob, variables = cvx_relu_normalize_relax(X, y, dmat, args.beta)

    # solve the problem
    prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False, mosek_params={})

    data["dmat"] = dmat
    for key in variables:
        data["opt_" + key] = variables[key].value

    values = [var.value for var in variables.values()]
    metrics = get_metrics(args, X, dmat, ind, w, *values)
    data |= metrics

    return data
