from typing import Tuple

from training.common import Args, generate_data, get_arrangement_patterns
from training.skip import ConvexReLU
from training.normalized import ConvexReLUNormalized


def solve_problem(n: int, d: int, args: Args) -> Tuple[dict, dict]:
    """
    Takes in the arguments for the problem setting,
    and returns the optimal variable values and the metrics.
    """
    data, metrics = {}, {}
    X, W, y = generate_data(n, d, args, data=data, eps=1e-10)

    mh = max(50, 2 * n if args.form == "irregular" else n)
    D_mat, ind, metrics["exist_all_one"] = get_arrangement_patterns(
        X, w=W if args.learned == "normalized" else None, p_hat=mh
    )

    skip = args.learned == "skip"
    if args.learned == "plain" or skip:
        program = ConvexReLU(args.form, X, y, D_mat, beta=args.beta, skip=skip)
    elif args.learned == "normalized":
        program = ConvexReLUNormalized(args.form, X, y, D_mat, beta=args.beta)

    _problem, metrics["obj"] = program.solve()
    variables = program.get_variables()

    data["dmat"] = D_mat
    for key, value in variables._asdict().items():
        if value is not None:
            data["opt_" + key] = value

    metrics |= program.get_metrics(X, W, D_mat, ind, args.tol)
    # not supported by Mosek solver
    # metrics["num_iters"] = prob.solver_stats.num_iters
    metrics["test_err"] = program.get_test_err(
        n,
        d,
        cubic=args.cubic,
        whiten=args.whiten,
        planted=args.planted,
        W_true=W,
    )

    return data, metrics
