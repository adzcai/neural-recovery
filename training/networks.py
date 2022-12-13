import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(act: str):
    if act == "relu":
        return nn.ReLU()
    if act == "leaky_relu":
        return nn.LeakyReLU()
    if act == "tanh":
        return nn.Tanh()
    if act == "sigmoid":
        return nn.Sigmoid()
    raise NotImplementedError


class ReLUnormal(nn.Module):
    """
    ReLU with normalization layer

    In order:
    - Linear layer (d, m)
    - ReLU
    - L2 Normalization layer (with learned scaling param alpha)
    - Linear layer (m, 1)

    m is number of neurons in the hidden layer
    n is unused
    d is dimension of the input
    """

    def __init__(self, m, n, d, act):
        super(ReLUnormal, self).__init__()
        self.m, self.n, self.d = m, n, d

        self.W1 = nn.Linear(self.d, self.m, bias=False)
        self.w2 = nn.Linear(self.m, 1, bias=False)
        self.alpha = nn.Parameter(torch.ones(self.m))

        self.act = get_activation(act)

    def forward(self, X):
        Xu = self.act(self.W1(X))
        y1 = self.alpha @ F.normalize(Xu, dim=0)
        y = self.w2(y1)
        return y

    def name(self):
        return "ReLU_network_with_normalization_layer"


class ReLUskip(nn.Module):
    """
    ReLU with skip connection and no normalization

    In order:
    - Concat of:
        - Relu path: Linear layer (d, m), then ReLU
        - Skip path: Linear layer (d, 1)
    - Linear (alpha) layer (m + 1, 1)

    m is number of neurons in the hidden layer
    n is unused
    d is dimension of the input
    """

    def __init__(self, m, n, d, act):
        super(ReLUskip, self).__init__()
        self.m, self.n, self.d = m, n, d

        self.W1 = nn.Linear(self.d, self.m, bias=False)
        self.alpha = nn.Linear(self.m, 1, bias=False)

        # skip connections
        self.w0 = nn.Linear(self.d, 1, bias=False)
        self.alpha0 = nn.Linear(1, 1, bias=False)

        self.act = get_activation(act)

    def forward(self, X):
        Xu = self.act(self.W1(X))
        y = self.alpha(Xu) + self.alpha0(self.w0(X))
        return y

    def name(self):
        return "ReLU_network_with_skip_connection"
