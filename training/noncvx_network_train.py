import math
import numpy as np
import torch
from torch.nn import Module
from torch import Tensor
from tqdm import tqdm

from training.networks import ReLUplain, ReLUnormal, ReLUskip, activations
from training.common import (
    generate_X,
    generate_data,
    generate_y,
)
from utils import Args


def train_model(n: int, d: int, sample: int, args: Args) -> dict:
    data, metrics = {}, {}

    # training data
    Xtrain, W_true, ytrain = generate_data(n, d, args, data=data, eps=1e-10)

    # test data
    Xtest = generate_X(n, d, args.cubic, args.whiten)
    ytest = generate_y(
        Xtest,
        W_true,
        sigma=args.sigma,
        eps=0,
        relu=args.planted != "linear",
        normalize=args.planted == "normalized",
    )

    data["X_test"] = Xtest
    data["y_test"] = ytest

    Xtrain, ytrain, Xtest, ytest = [
        torch.from_numpy(t).float() for t in (Xtrain, ytrain, Xtest, ytest)
    ]

    m = args.m if args.m is not None else n + 1
    act = activations[args.activation]
    if args.learned == "normalized":
        model = ReLUnormal(m, n, d, act)
    elif args.learned == "skip":
        model = ReLUskip(m, n, d, act)
    elif args.learned == "plain":
        model = ReLUplain(m, n, d, act)
    else:
        raise NotImplementedError

    data["loss_train"], data["loss_test"] = train_network(
        model,
        Xtrain,
        ytrain,
        Xtest,
        ytest,
        num_epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        verbose=not args.quiet,
    )

    metrics["test_err"] = math.sqrt(data["loss_test"][-1])

    if args.learned == "skip":
        # compare the (merged) skip connection weights with the true weights
        w_skip = model.w_skip.weight.detach().numpy()
        w_skip = model.alpha_skip.weight.item() * w_skip.T
        metrics["dis_abs"] = np.linalg.norm(w_skip - W_true)
        metrics["cos_sim"] = (w_skip.flatten() @ W_true.flatten()) / (
            np.linalg.norm(w_skip) * np.linalg.norm(W_true)
        )

        # draft below, but removed, since the additional second layer might cause some interference
        # metrics["recovery"] = np.allclose(w_skip, W_true, atol=args.tol) # and np.allclose((alpha @ W1).reshape(-1), 0, atol=args.tol)
    elif args.learned == "plain":
        w = model.W_relu.weight.detach().numpy()
        alpha = model.alpha_relu.weight.item()

    W_relu = model.W_relu.weight.detach().numpy()  # (m, d)
    alpha_relu = model.alpha_relu.weight.detach().numpy()  # (1, m)
    metrics["other_norm"] = np.linalg.norm(W_relu * alpha_relu.T, ord="fro")

    if args.save_details:
        torch.save(
            model.state_dict(),
            args.get_save_folder()
            + f"weights__n{n}__d{d}__sample{sample}__act{args.activation}.pth",
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
    """
    Full-batch gradient descent.
    """

    # reshape y so that subtraction works properly and doesn't get broadcasted
    if ytrain.ndim == 1:
        ytrain = ytrain[:, None]
    if ytest.ndim == 1:
        ytest = ytest[:, None]

    loss_train, loss_test = np.zeros((2, 1 + num_epochs))

    # get initialization statistics
    with torch.no_grad():
        loss_init = ((model(Xtrain) - ytrain) ** 2).sum()
        test_err_init = ((model(Xtest) - ytest) ** 2).sum()

        loss_train[0] = loss_init.item()
        loss_test[0] = test_err_init.item()

    epoch_iter = range(1, num_epochs + 1)
    if verbose:
        epoch_iter = tqdm(epoch_iter, total=num_epochs, leave=None)

    for epoch in epoch_iter:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=beta)
        optimizer.zero_grad()
        loss = ((model(Xtrain) - ytrain) ** 2).sum()
        loss.backward()
        optimizer.step()
        loss_train[epoch] = loss.item()

        with torch.no_grad():
            test_err = ((model(Xtest) - ytest) ** 2).sum()
            loss_test[epoch] = test_err.item()

        if verbose:
            epoch_iter.set_description(
                "Train error: {:.3f}, Test error: {:.3f}".format(
                    loss_train[epoch], loss_test[epoch]
                )
            )

    return loss_train, loss_test
