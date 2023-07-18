import torch
import torch.nn as nn
import torch.nn.functional as F


activations = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "gelu": nn.GELU(),
}


class ReLUplain(nn.Module):
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
        super().__init__()

        self.W_relu = nn.Linear(d, m, bias=False)
        self.alpha_relu = nn.Linear(m, 1, bias=False)

        self.act = act

    def forward(self, X):
        y = self.alpha_relu(self.act(self.W_relu(X)))
        return y

    def name(self):
        return "ReLU_network_plain"


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

        self.act = act

    def forward(self, X):
        Xu = self.act(self.W1(X))
        out = self.alpha * F.normalize(Xu, dim=0)
        y = self.w2(out)
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
        super().__init__()

        self.W_relu = nn.Linear(d, m, bias=False)
        self.alpha_relu = nn.Linear(m, 1, bias=False)

        # skip connections
        self.w_skip = nn.Linear(d, 1, bias=False)
        self.alpha_skip = nn.Linear(1, 1, bias=False)

        self.act = act

    def forward(self, X):
        out_relu = self.alpha_relu(self.act(self.W_relu(X)))
        out_skip = self.alpha_skip(self.w_skip(X))
        y = out_relu + out_skip
        return y

    def name(self):
        return "ReLU_network_with_skip_connection"
