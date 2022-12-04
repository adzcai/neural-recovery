import cvxpy as cp

from common import generate_data, get_arrangement_patterns
from cvx_problems import (
    cvx_relu_normalize,
    cvx_relu_normalize_relax,
    cvx_relu_skip,
    cvx_relu_skip_relax,
)
from metrics import get_metrics


def solve_problem(n, d, args):
    data = {}
    X, w, y = generate_data(n, d, args, data=data)

    mh = max(50, 2 * n if args.form != "irregular" else n)
    dmat, ind, data["exist_all_one"] = get_arrangement_patterns(X, w, mh)

    if args.model == "skip":
        if args.form == "convex":
            prob, variables = cvx_relu_skip(X, y, dmat, args.beta)
        elif args.form == "minnorm":
            prob, variables = cvx_relu_skip_relax(X, y, dmat, args.beta)
    if args.model == "normalize":
        if args.form == "convex":
            prob, variables = cvx_relu_normalize(X, y, dmat, args.beta)
        elif args.form == "minnorm":
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
