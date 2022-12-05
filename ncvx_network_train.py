import math

import numpy as np
import torch

from networks import ReLUnormal, ReLUskip
from common import (
    generate_X,
    generate_data,
    generate_y,
    get_save_folder,
)


def train_model(n, d, sample, args):
    sigma = args.sigma
    data = {}  # empty dict

    # training data
    X, w, y = generate_data(n, d, args)

    # test data
    Xtest = generate_X(n, d, args)
    ytest = generate_y(args, X, w, sigma=sigma, eps=0, model=args.planted)

    data["X_test"] = Xtest
    data["y_test"] = ytest

    Xtrain, ytrain, Xtest, ytest = [torch.from_numpy(t).float() for t in (X, y, Xtest, ytest)]

    m = n + 1
    if args.model == "normal":
        model = ReLUnormal(m=m, n=n, d=d, act=args.act)
    else:
        model = ReLUskip(m=m, n=n, d=d, act=args.act)
    loss_train, loss_test = train_network(model, Xtrain, ytrain, Xtest, ytest, args)

    data["loss_train"] = loss_train
    data["loss_test"] = loss_test
    data["test_err"] = math.sqrt(loss_test[-1])

    if args.model == "skip":
        # compare the (merged) skip connection weights with the true weights
        w0 = model.w0.weight.detach().numpy()
        alpha0 = model.alpha0.weight.item()

        data["dis_abs"] = np.linalg.norm(alpha0 * w0.T - w, ord=2)
        # draft below, but removed, since the additional second layer might cause some interference
        # data["recovery"] = np.allclose(alpha0 * w0.T, w, atol=1e-4) and np.allclose((alpha @ W1).reshape(-1), 0, atol=1e-4)

    if args.save_details:
        torch.save(
            model.state_dict(),
            get_save_folder(args) + f"weights__n{n}__d{d}__sample{sample}.pth",
        )

    return data


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
        print("Epoch [{}/{}], Train error: {}, Test error: {}".format(0, args.num_epoch, train_err_init, test_err_init))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.beta)

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
