import math
import numpy as np
import torch
from torch.nn import Module
from torch import Tensor

from training.networks import ReLUnormal, ReLUskip
from training.common import (
    generate_X,
    generate_data,
    generate_y,
)


def train_model(n, d, sample, args) -> dict:
    sigma = args.sigma
    data, metrics = {}, {}

    # training data
    X, w, y = generate_data(n, d, args, eps=1e-10)

    # test data
    Xtest = generate_X(n, d, args)
    ytest = generate_y(
        X,
        Variables(W_pos=w),
        sigma=sigma,
        eps=0,
        relu=args.planted != "linear",
        normalize=args.planted == "normalized",
    )

    data["X_test"] = Xtest
    data["y_test"] = ytest

    Xtrain, ytrain, Xtest, ytest = [torch.from_numpy(t) for t in (X, y, Xtest, ytest)]

    m = n + 1
    if args.learned == "normalized":
        model = ReLUnormal(m=m, n=n, d=d, act=args.act)
    else:
        model = ReLUskip(m=m, n=n, d=d, act=args.act)
    loss_train, loss_test = train_network(
        model,
        Xtrain,
        ytrain,
        Xtest,
        ytest,
        num_epochs=args.num_epoch,
        verbose=not args.quiet,
    )

    data["loss_train"] = loss_train
    data["loss_test"] = loss_test

    metrics["test_err"] = math.sqrt(loss_test[-1])

    if args.learned == "skip":
        # compare the (merged) skip connection weights with the true weights
        w0 = model.w0.weight.detach().numpy()
        alpha0 = model.alpha0.weight.item()

        metrics["dis_abs"] = np.linalg.norm(alpha0 * w0.T - w, ord=2)
        # draft below, but removed, since the additional second layer might cause some interference
        # data["recovery"] = np.allclose(alpha0 * w0.T, w, atol=1e-4) and np.allclose((alpha @ W1).reshape(-1), 0, atol=1e-4)

    if args.save_details:
        torch.save(
            model.state_dict(),
            args.get_save_folder() + f"weights__n{n}__d{d}__sample{sample}.pth",
        )

    return data, metrics


def train_network(
    model: Module,
    Xtrain: Tensor,
    ytrain: Tensor,
    Xtest: Tensor,
    ytest: Tensor,
    num_epochs,
    verbose=False,
    lr=None,
    beta=None,
):
    if verbose:
        print("---------------------------training---------------------------")

    # get initialization statistics
    y_predict = model(Xtrain)
    loss = torch.linalg.norm(y_predict - ytrain) ** 2
    train_err_init = loss.item()
    with torch.no_grad():
        test_err_init = torch.linalg.norm(model(Xtest) - ytest) ** 2
        test_err_init = test_err_init.item()

    if verbose:
        print(
            "Epoch [{}/{}], Train error: {}, Test error: {}".format(
                0, num_epochs, train_err_init, test_err_init
            )
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=beta)

    loss_train, loss_test = np.zeros(2, num_epochs)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_predict = model(Xtrain)
        loss = torch.linalg.norm(y_predict - ytrain) ** 2
        loss.backward()
        optimizer.step()

        loss_train[epoch] = loss.item()
        with torch.no_grad():
            test_err = torch.linalg.norm(model(Xtest) - ytest) ** 2
            loss_test[epoch] = test_err.item()

        if verbose:
            print(
                "Epoch [{}/{}], Train error: {}, Test error: {}".format(
                    epoch + 1, num_epochs, loss_train[epoch], loss_test[epoch]
                )
            )

    loss_train = np.concatenate([np.array([train_err_init]), loss_train])
    loss_test = np.concatenate([np.array([test_err_init]), loss_test])

    return loss_train, loss_test
