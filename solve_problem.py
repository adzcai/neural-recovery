from typing import Tuple
import cvxpy as cp

from common import generate_data, get_arrangement_patterns
from cvx_problems import ConvexReLU, ConvexReLUNormalized, ConvexReLUNormalizedRelaxed, ConvexReLURelaxed
from metrics import get_metrics, get_test_err


def solve_problem(n, d, args) -> Tuple[dict, dict]:
    """
    Takes in the arguments for the problem setting,
    and returns the optimal variable values and the metrics.
    """
    data, metrics = {}, {}
    X, w, y = generate_data(n, d, args, data=data)

    mh = max(50, 2 * n if args.form == "irregular" else n)
    D_mat, ind, metrics["exist_all_one"] = get_arrangement_patterns(
        X, w=w if args.learned == "normalized" else None, n_sampled=mh
    )

    if args.learned == "plain" or args.learned == "skip":
        skip = args.learned == "skip"
        if args.form == "exact":
            program = ConvexReLU(X, y, D_mat, skip=skip, exact=True)
        elif args.form == "approx":
            program = ConvexReLU(X, y, D_mat, beta=args.beta, skip=skip)
        elif args.form == "relaxed":
            program = ConvexReLURelaxed(X, y, D_mat, skip=skip)
        else:
            raise ValueError("Unknown form {}".format(args.form))

    elif args.learned == "normalized":
        if args.form == "exact":
            program = ConvexReLUNormalized(X, y, D_mat, exact=True)
        elif args.form == "approx":
            program = ConvexReLUNormalized(X, y, D_mat, args.beta)
        elif args.form == "relaxed":
            program = ConvexReLUNormalizedRelaxed(X, y, D_mat)
        else:
            raise ValueError("Unknown form {}".format(args.form))

    else:
        raise ValueError("Unknown model {}".format(args.learned))

    program.solve()
    variables = program.get_variables()

    data["dmat"] = D_mat
    for key, value in variables._asdict().items():
        if value is not None:
            data["opt_" + key] = value

    metrics = get_metrics(args, X, w, D_mat, ind, variables)
    # not supported by Mosek solver
    # metrics["num_iters"] = prob.solver_stats.num_iters
    metrics["test_err"] = get_test_err(
        n,
        d,
        args.optx,
        args.planted,
        args.learned,
        variables,
    )

    return data, metrics
