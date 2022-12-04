import argparse
import math
import os
import pickle
from time import time

import numpy as np
import torch
import torch.optim as optim

from common import gen_data, get_parser, train_network
from networks import ReLUskip


def train_model(args):
    n, d, sigma = args.n, args.d, args.sigma
    data = {}  # empty dict
    X, w = gen_data(n, d)
    w = w.reshape((d, 1))
    z = np.random.randn(n, 1) * sigma / math.sqrt(n)
    y = X @ w + z
    data["X"] = X
    data["w"] = w
    data["y"] = y

    Xtest, z = gen_data(n, d)
    ytest = Xtest @ w
    data["X_test"] = Xtest
    data["y_test"] = ytest

    Xtrain = torch.from_numpy(X).float()
    ytrain = torch.from_numpy(y).float()
    Xtest = torch.from_numpy(Xtest).float()
    ytest = torch.from_numpy(ytest).float()

    m = n + 1
    model = ReLUskip(m=m, n=n, d=d)
    loss_train, loss_test = train_network(model, Xtrain, ytrain, Xtest, ytest, args)

    data["loss_train"] = loss_train
    data["loss_test"] = loss_test
    w0 = model.w0.weight.detach().numpy()
    alpha0 = model.alpha0.weight.item()
    data["dis_abs"] = np.linalg.norm(alpha0 * w0.T - w, 2)
    data["dis_test"] = math.sqrt(loss_test[-1])
    return data, model


def main():
    parser = get_parser(optw=1)
    args = parser.parse_args()
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

    dis_abs = np.zeros((nlen, dlen, sample))
    dis_test = np.zeros((nlen, dlen, sample))

    for nidx, n in enumerate(nvec):
        print("n = " + str(n))
        t0 = time()
        for didx, d in enumerate(dvec):
            for i in range(sample):
                data, model = train_model(n, d, sigma, verbose=args.verbose)
                dis_abs[nidx, didx, i] = data["dis_abs"]
                dis_test[nidx, didx, i] = data["dis_test"]
                if flag:
                    fname = "_n{}_d{}_w{}_X{}_sig{}_sample{}".format(
                        n, d, optw, optx, sigma, i
                    )
                    file = open(save_folder + "ncvx_train_skip" + fname + ".pkl", "wb")
                    pickle.dump(data, file)
                    file.close()
                    torch.save(model.state_dict(), save_folder + model.name() + fname)

        t1 = time()
        print("time = " + str(t1 - t0))

    fname = "_n{}_d{}_w{}_X{}_sig{}_sample{}".format(
        args.n, args.d, optw, optx, sigma, sample
    )
    np.save(save_folder + "dis_abs_ncvx_train_skip" + fname, dis_abs)
    np.save(save_folder + "dis_test_ncvx_train_skip" + fname, dis_test)


if __name__ == "__main__":
    main()
