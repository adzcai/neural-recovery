"""
Neural Isometry Conditions proposed in the paper.
"""

import numpy as np

def irrepresentable(X, w, dmat):
    """
    
    """
    pass

def RIP(X, w, dmat):
    """
    Restricted Isometry Property
    """
    pass

def NIC_linear(X, w, dmat):
    """
    Definition 2 in the paper.

    dmat is (n, p) matrix of the arrangement patterns.
    """

    p = dmat.shape[1]
    for j in range(p):
        norm = X.T @ (dmat * X) @ np.linalg.inv(X.T @ X) @ w

def NIC_1(X, w, dmat):
    """
    Definition 3 in the paper.
    Suppose the planted model is a single ReLU neuron,
    such that y = relu(X @ w).
    If this holds, then the unique solution to Equation 11,
    i.e. convex formulation of a plain ReLU NN,
    is the single planted ReLU neuron.
    That is, a single neuron learns the optimal weights and all others are set to zero.
    """

    D_opt = (X @ w) >= 0
    XD_opt = D_opt.reshape(-1, 1) * X
    p = dmat.shape[1]
    for j in range(p):
        D = dmat[:, j]
        if np.allclose(D, D_opt):
            continue
        XD_j = D.reshape(-1, 1) * X
        norm = (XD_j.T @ XD_opt) @ np.linalg.inv(X.T @ XD_opt) @ w
        if np.linalg.norm(norm) > 1:
            return False
    return True

def NIC_k(X, w, dmat):
    """
    Definition 5 in the paper.
    If this condition holds, then the unique solution to Equation 11,
    i.e. convex formulation of a plain ReLU NN,
    is to learn the planted neurons and set all others to zero.
    """

    dmat = np.array([np.diag(d) for d in dmat.T])
    # print("shapes", X.shape, w.shape, y.shape, dmat.shape)
    Dsi = np.array([np.diag(neuron) for neuron in ((X @ w) >= 0).T]) # (k, n, n)
    assert dmat.shape[1] == Dsi.shape[1] and dmat.shape[2] == Dsi.shape[2]
    S = []
    for d in Dsi:
        for i, other_d in enumerate(dmat):
            if np.allclose(d, other_d):
                S.append(i)
                break

    middle_prod = np.einsum("d n, k n m -> k d m", X.T, Dsi)
    middle = np.concatenate(middle_prod, axis=0).T
    w_hat = w / np.linalg.norm(w, axis=0)
    right = np.concatenate(w_hat.T, axis=0)
    NIC_holds = True
    for i, d in enumerate(dmat):
        if i not in S:
            NIC_holds = NIC_holds and np.linalg.norm(X.T @ d @ middle @ right) < 1
    return NIC_holds

def NIC_k_normalized(X, w, dmat):
    """
    Definition 6 in the paper.
    If this condition holds, then the unique solution to Equation 16,
    i.e. convex formulation of a normalized ReLU NN,
    is to learn the planted neurons and set all others to zero.

    w is a (d x k) matrix of the weights.
    dmat is a (n x p) matrix of the arrangement patterns.
    """

    p = dmat.shape[1]

    # S = []
    # for i,  in enumerate(dmat.T):

    for j in range(p):
        D = np.diag(dmat[:, j])
        U, S, Vh = np.linalg.svd(D)

        norm = np.linalg.norm(X.T @ D @ np.linalg.inv(X.T @ X) @ w)
        if norm > 1:
            return False
