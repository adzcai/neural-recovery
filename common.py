import argparse
import torch
import math
import numpy as np
import scipy.optimize as sciopt


def get_parser(neu=None, default_samples=10, optw=0):
    parser = argparse.ArgumentParser(description="phase transition")
    parser.add_argument("--n", type=int, default=400, help="number of sample")
    parser.add_argument("--d", type=int, default=100, help="number of dimension")
    parser.add_argument("--seed", type=int, default=97006855, help="random seed")
    parser.add_argument(
        "--sample", type=int, default=default_samples, help="number of trials"
    )
    parser.add_argument(
        "--model",
        type=int,
        choices=[0, 1],
        default=0,
        help="underlying model. 0=linear, 1=relu network",
    )
    parser.add_argument(
        "--neu", type=int, default=neu, help="number of planted neurons"
    )

    parser.add_argument("--optw", type=int, default=optw, help="choice of w")
    # 0: randomly generated (Gaussian)
    # 1: smallest right eigenvector of X
    # 2: randomly generated ReLU network

    parser.add_argument("--optx", type=int, default=0, help="choice of X")
    # 0: Gaussian
    # 1: cubic Gaussian
    # 2: 0 + whitened
    # 3: 1 + whitened

    parser.add_argument("--sigma", type=float, default=0, help="noise")
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="whether to print information while training",
    )
    parser.add_argument(
        "--save_details",
        type=bool,
        default=True,
        help="whether to save training results",
    )
    parser.add_argument(
        "--save_folder", type=str, default="./results/", help="path to save results"
    )

    # nonconvex training
    parser.add_argument(
        "--num_epoch", type=int, default=400, help="number of training epochs"
    )
    parser.add_argument(
        "--beta", type=float, default=1e-6, help="weight decay parameter"
    )
    parser.add_argument("--lr", type=float, default=2e-3, help="learning rate")

    return parser


def validate_data(args, eps=1e-10):
    """
    We don't want any neuron to return all 0s across the dataset
    """
    while True:
        X, w = gen_data(args)
        w /= np.linalg.norm(w, axis=0)
        y = np.maximum(0, X @ w)
        norm_y = np.linalg.norm(y, axis=0)
        if np.all(norm_y >= eps):
            break
    y = np.sum(y / norm_y, axis=1)
    return X, w, y


def check_feasible(X):
    """
    Check if there exists some hyperplane passing through the origin
    such that all elements of X lie on on side of the hyperplane,
    in which case we need to add the identity matrix to the set of arrangement patterns.
    """
    n, d = X.shape
    nrm = lambda x: np.linalg.norm(x, ord=2)
    x0 = np.random.randn(d)
    x0 = x0 / np.linalg.norm(x0)
    lc = sciopt.LinearConstraint(X, 0, np.inf)
    nlc = sciopt.NonlinearConstraint(nrm, 0, 1)
    res = sciopt.minimize(lambda x: -nrm(x), x0, constraints=[lc, nlc])
    if -res.fun <= 1e-6:
        return False  # no all-one arrangement
    else:
        return True  # exist all-one arrangement


def get_arr_patterns(X, n, d, w=None):
    mh = max(n, 50)
    U1 = np.random.randn(d, mh)
    if w is not None:
        U1 = np.concatenate([w, U1], axis=1)
    arr_patterns = X @ U1 >= 0
    arr_patterns, ind = np.unique(arr_patterns, axis=1, return_index=True)
    feasible = check_feasible(X)
    if feasible:
        arr_patterns = np.concatenate([arr_patterns, np.ones((n, 1))], axis=1)
    return arr_patterns, ind, feasible


def gen_data(args):
    n, d = args.n, args.d
    X = np.random.randn(n, d) / math.sqrt(n)
    if args.optx in [1, 3]:
        X = X**3
    if args.optx in [2, 4]:
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        if n < d:
            X = Vh
        else:
            X = U

    if args.neu is not None:
        # involving a planted neuron
        if args.optw == 0:
            w = np.random.randn(d, args.neu)
        elif args.optw == 1:
            w = np.eye(d, args.neu)
        elif args.optw == 2:
            if args.neu == 2:
                w = np.random.randn(d, 1)
                w = np.concatenate([w, -w], axis=1)
            else:
                raise TypeError("Invalid choice of planted neurons.")
    else:
        if args.optw == 0:
            w = np.random.randn(d)
            w = w / np.linalg.norm(w)
        elif args.optw == 1:
            U, S, Vh = np.linalg.svd(X, full_matrices=False)
            w = Vh[-1, :].T
        elif args.optw == 2:
            assert args.m is not None, "must specify number of hidden neurons m"

    return X, w


def train_network(model, Xtrain, ytrain, Xtest, ytest, args):
    if args.verbose:
        print("---------------------------training---------------------------")

    # get initialization statistics
    y_predict = model(Xtrain)
    loss = torch.linalg.norm(y_predict - ytrain) ** 2
    train_err_init = loss.item()
    with torch.no_grad():
        test_err_init = torch.linalg.norm(model(Xtest) - ytest) ** 2
        test_err_init = test_err_init.item()

    loss_train = np.zeros(args.num_epoch)
    loss_test = np.zeros(args.num_epoch)

    if args.verbose:
        print(
            "Epoch [{}/{}], Train error: {}, Test error: {}".format(
                0, args.num_epoch, train_err_init, test_err_init
            )
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.beta
    )

    for epoch in range(args.num_epoch):
        optimizer.zero_grad()
        y_predict = model(Xtrain)
        loss = torch.linalg.norm(y_predict - ytrain) ** 2
        loss.backward()
        optimizer.step()

        loss_train[epoch] = loss.item()
        with torch.no_grad():
            test_err = torch.linalg.norm(model(Xtest) - ytest) ** 2
            loss_test[epoch] = test_err.item()

        if args.verbose:
            print(
                "Epoch [{}/{}], Train error: {}, Test error: {}".format(
                    epoch + 1, args.num_epoch, loss_train[epoch], loss_test[epoch]
                )
            )

    loss_train = np.concatenate([np.array([train_err_init]), loss_train])
    loss_test = np.concatenate([np.array([test_err_init]), loss_test])

    return loss_train, loss_test
