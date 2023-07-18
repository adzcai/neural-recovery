from typing import Tuple

from training.common import Args, generate_data, get_arrangement_patterns
from training.cvx_skip import ConvexReLU
from training.cvx_normalized import ConvexReLUNormalized


def solve_problem(n: int, d: int, args: Args) -> Tuple[dict, dict]:
    """
    Takes in the arguments for the problem setting,
    and returns the optimal variable values and the metrics.
    """
    data, metrics = {}, {}
    X, W, y = generate_data(n, d, args, data=data, eps=1e-10)

    D_mat, ind, metrics["exist_all_one"] = get_arrangement_patterns(
        X, w=W if args.learned == "normalized" else None, p_hat=max(50, n)
    )

    skip = args.learned == "skip"
    if args.learned == "plain" or skip:
        program = ConvexReLU(args.form, X, y, D_mat, beta=args.beta, skip=skip)
    elif args.learned == "normalized":
        program = ConvexReLUNormalized(args.form, X, y, D_mat, beta=args.beta)
    else:
        raise ValueError(f"Unknown learned model: {args.learned}")

    program.solve()  # SOLVE THE PROBLEM

    data["dmat"] = D_mat
    for key, value in program.get_variables()._asdict().items():
        if value is not None:
            data["opt_" + key] = value

    metrics |= program.get_metrics(X, W, D_mat, ind, args.tol)
    # not supported by Mosek solver
    # metrics["num_iters"] = prob.solver_stats.num_iters

    metrics["test_err"], metrics["test_dis"] = (
        program.get_test_err(
            n,
            d,
            cubic=args.cubic,
            whiten=args.whiten,
            planted=args.planted,
            W_true=W,
        )
        if n >= d
        else (None, None)
    )

    return data, metrics
