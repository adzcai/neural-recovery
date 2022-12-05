import cvxpy as cp

from common import generate_data, get_arrangement_patterns
from cvx_problems import (
    cvx_relu_normalized,
    cvx_relu_normalized_relax,
    cvx_relu,
    cvx_relu_relax,
)
from metrics import get_metrics


def solve_problem(n, d, args):
    data = {}
    X, w, y = generate_data(n, d, args, data=data)

    mh = max(50, 2 * n if args.form == "irregular" else n)
    dmat, ind, data["exist_all_one"] = get_arrangement_patterns(
        X, w=w if args.model == "normalized" else None, n_sampled=mh
    )

    if args.model == "plain":
        if args.form == "approx":
            prob, variables = cvx_relu(X, y, dmat, args.beta, skip=False)
        elif args.form == "exact":
            prob, variables = cvx_relu(X, y, dmat, args.beta, skip=False, exact=True)
        elif args.form == "relaxed":
            prob, variables = cvx_relu_relax(X, y, dmat, args.beta, skip=False)
        else:
            raise ValueError("Unknown form {}".format(args.form))
    elif args.model == "skip":
        if args.form == "approx":
            prob, variables = cvx_relu(X, y, dmat, args.beta, skip=True)
        elif args.form == "exact":
            prob, variables = cvx_relu(X, y, dmat, args.beta, skip=True, exact=True)
        elif args.form == "relaxed":
            prob, variables = cvx_relu_relax(X, y, dmat, args.beta, skip=True)
        else:
            raise ValueError("Unknown form {}".format(args.form))
    elif args.model == "normalized":
        if args.form == "approx":
            prob, variables = cvx_relu_normalized(X, y, dmat, args.beta)
        elif args.form == "relaxed":
            prob, variables = cvx_relu_normalized_relax(X, y, dmat, args.beta)
    else:
        raise ValueError("Unknown model {}".format(args.model))

    # solve the problem
    prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False, mosek_params={})

    data["dmat"] = dmat
    for key in variables:
        data["opt_" + key] = variables[key].value

    values = [var.value for var in variables.values()]
    metrics = get_metrics(args, X, dmat, ind, w, *values)
    data |= metrics

    return data
