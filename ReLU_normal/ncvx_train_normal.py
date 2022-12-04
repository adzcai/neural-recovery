import math
import os
import pickle
from time import time

import numpy as np
import torch

from networks import ReLUnormal
from common import gen_data, get_parser, train_network, validate_data


def train_model(args):
    n, d, sigma = args.n, args.d, args.sigma

    data = {}  # empty dict

    # training data
    X, w, y = validate_data(args)
    z = np.random.randn(n) * sigma / math.sqrt(n)
    y = y + z
    data["X"] = X
    data["w"] = w
    data["y"] = y

    # test data
    Xtest, z = gen_data(args)
    ytest = np.maximum(0, Xtest @ w)
    norm_y = np.linalg.norm(ytest, axis=0)
    ytest = np.sum(ytest / norm_y, axis=1)  # equivalent to a "second layer" of just 1s
    data["X_test"] = Xtest
    data["y_test"] = ytest
    y = y.reshape((n, 1))
    ytest = ytest.reshape((n, 1))

    Xtrain, ytrain, Xtest, ytest = [torch.from_numpy(t).float() for t in (X, y, Xtest, ytest)]

    m = n + 1
    model = ReLUnormal(m=m, n=n, d=d)

    loss_train, loss_test = train_network(model, Xtrain, ytrain, Xtest, ytest, args)

    data["loss_train"] = loss_train
    data["loss_test"] = loss_test
    data["dis_test"] = math.sqrt(loss_test[-1])
    return data, model


def main(title="ncvx_train_normal"):
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
                    file = open(save_folder + title + fname + ".pkl", "wb")
                    pickle.dump(data, file)
                    file.close()
                    torch.save(model.state_dict(), save_folder + model.name() + fname)

        t1 = time()
        print("time = " + str(t1 - t0))

    fname = "_n{}_d{}_w{}_X{}_sig{}_sample{}".format(
        args.n, args.d, optw, optx, sigma, sample
    )
    np.save(save_folder + "dis_test_" + title + fname, dis_test)


if __name__ == "__main__":
    main()
