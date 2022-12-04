import math
import os
import pickle
from time import time

import numpy as np
import torch

from networks import ReLUnormal, ReLUskip
from common import gen_data, get_parser, train_network, validate_data


def train_model(args, mode="normal"):
    n, d, sigma = args.n, args.d, args.sigma

    data = {}  # empty dict

    # training data
    if mode == "normal":
        X, w, y = validate_data(args)
    else:
        X, w = gen_data(args)
        y = X @ w
    z = np.random.randn(n) * sigma / math.sqrt(n)
    y += z
    data["X"] = X
    data["w"] = w
    data["y"] = y

    # test data
    Xtest, z = gen_data(args)
    if mode == "normal":
        ytest = np.maximum(0, Xtest @ w)
        norm_y = np.linalg.norm(ytest, axis=0)
        ytest = np.sum(
            ytest / norm_y, axis=1
        )  # equivalent to a "second layer" of just 1s
    else:
        ytest = Xtest @ w
    data["X_test"] = Xtest
    data["y_test"] = ytest

    Xtrain, ytrain, Xtest, ytest = [
        torch.from_numpy(t).float() for t in (X, y, Xtest, ytest)
    ]

    m = n + 1
    if mode == "normal":
        model = ReLUnormal(m=m, n=n, d=d)
    else:
        model = ReLUskip(m=m, n=n, d=d)
    loss_train, loss_test = train_network(model, Xtrain, ytrain, Xtest, ytest, args)

    data["loss_train"] = loss_train
    data["loss_test"] = loss_test
    if mode == "skip":
        # compare the (merged) skip connection weights with the true weights
        w0 = model.w0.weight.detach().numpy()
        alpha0 = model.alpha0.weight.item()
        data["dis_abs"] = np.linalg.norm(alpha0 * w0.T - w, ord=2)
    data["dis_test"] = math.sqrt(loss_test[-1])
    return data, model


def main():
    parser = get_parser(neu=None, optw=None)
    args = parser.parse_args()
    if args.model == "ReLU_normal":
        mode = "normal"
        title = "ncvx_train_normal"
        if args.neu is None:
            args.neu = 2
    elif args.model == "ReLU_skip":
        parser = get_parser(optw=1)
        mode = "skip"
        title = "ncvx_train_skip"
        if args.optw is None:
            args.optw = 1
    else:
        raise NotImplementedError("Invalid model type.")
    print(str(args))

    save_folder = args.save_folder
    seed = args.seed
    np.random.seed(seed)
    sigma = args.sigma
    sample = args.sample
    optw = args.optw
    optx = args.optx
    flag = args.save_details
    dvec = np.arange(10, args.d + 1, 10)
    nvec = np.arange(10, args.n + 1, 10)
    dlen = dvec.size
    nlen = nvec.size

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if mode == "skip":
        dis_abs = np.zeros((nlen, dlen, sample))
    dis_test = np.zeros((nlen, dlen, sample))

    for nidx, n in enumerate(nvec):
        print("n = " + str(n))
        t0 = time()
        for didx, d in enumerate(dvec):
            for i in range(sample):
                data, model = train_model(args)
                if mode == "skip":
                    dis_abs[nidx, didx, i] = data["dis_abs"]
                dis_test[nidx, didx, i] = data["dis_test"]
                if flag:
                    fname = "_n{}_d{}_w{}_X{}_sig{}_sample{}".format(
                        n, d, optw, optx, sigma, i
                    )
                    file = open(save_folder + title + fname + ".pkl", "wb")
                    pickle.dump(data, file)
                    file.close()
                    torch.save(model.state_dict(), save_folder + model.name() + fname)

        t1 = time()
        print("time = " + str(t1 - t0))

    fname = "_n{}_d{}_w{}_X{}_sig{}_sample{}".format(
        args.n, args.d, optw, optx, sigma, sample
    )
    if mode == "skip":
        np.save(save_folder + "dis_abs_" + title + fname, dis_abs)
    np.save(save_folder + "dis_test_" + title + fname, dis_test)


if __name__ == "__main__":
    main()
