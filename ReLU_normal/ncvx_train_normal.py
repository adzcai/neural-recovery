import math
import os
import pickle
from time import time

import numpy as np
import torch

from networks import ReLUnormal
from common import gen_data, get_parser, train_network


def train_model(args):
    n, d, sigma, neu = args.n, args.d, args.sigma, args.neu

    data = {}  # empty dict
    while True:
        X, w = gen_data(args)
        nrmw = np.linalg.norm(w, axis=0)
        w = w / nrmw
        y = np.maximum(0, X @ w)
        nrmy = np.linalg.norm(y, axis=0)
        if np.all(nrmy >= 1e-10):
            break

    y = np.sum(y / nrmy, axis=1)
    z = np.random.randn(n) * sigma / math.sqrt(n)
    y = y + z
    data["X"] = X
    data["w"] = w
    data["y"] = y
    Xtest, z = gen_data(args)
    ytest = np.maximum(0, Xtest @ w)
    nrmy = np.linalg.norm(ytest, axis=0)
    ytest = np.sum(ytest / nrmy, axis=1)
    data["X_test"] = Xtest
    data["y_test"] = ytest
    y = y.reshape((n, 1))
    ytest = ytest.reshape((n, 1))

    Xtrain = torch.from_numpy(X).float()
    ytrain = torch.from_numpy(y).float()
    Xtest = torch.from_numpy(Xtest).float()
    ytest = torch.from_numpy(ytest).float()

    m = n + 1
    model = ReLUnormal(m=m, n=n, d=d)

    loss_train, loss_test = train_network(model, Xtrain, ytrain, Xtest, ytest, args)

    data["loss_train"] = loss_train
    data["loss_test"] = loss_test
    data["dis_test"] = math.sqrt(loss_test[-1])
    return data, model


def main():
    parser = get_parser(neu=2)
    args = parser.parse_args()
    print(str(args))

    save_folder = args.save_folder
    seed = args.seed
    np.random.seed(seed)
    sigma = args.sigma
    sample = args.sample
    optw = args.optw
    optx = args.optx
    neu = args.neu
    flag = args.save_details
    dvec = np.arange(10, args.d + 1, 10)
    nvec = np.arange(10, args.n + 1, 10)
    dlen = dvec.size
    nlen = nvec.size

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    dis_test = np.zeros((nlen, dlen, sample))

    for nidx, n in enumerate(nvec):
        print("n = " + str(n))
        t0 = time()
        for didx, d in enumerate(dvec):
            for i in range(sample):
                data, model = train_model(n, d, sigma, neu, verbose=args.verbose)

                dis_test[nidx, didx, i] = data["dis_test"]
                if flag:
                    fname = "_n{}_d{}_w{}_X{}_sig{}_sample{}".format(
                        n, d, optw, optx, sigma, i
                    )
                    file = open(
                        save_folder + "ncvx_train_normal" + fname + ".pkl", "wb"
                    )
                    pickle.dump(data, file)
                    file.close()
                    torch.save(model.state_dict(), save_folder + model.name() + fname)

        t1 = time()
        print("time = " + str(t1 - t0))

    fname = "_n{}_d{}_w{}_X{}_sig{}_sample{}".format(
        args.n, args.d, optw, optx, sigma, sample
    )
    np.save(save_folder + "dis_test_ncvx_train_normal" + fname, dis_test)


if __name__ == "__main__":
    main()
