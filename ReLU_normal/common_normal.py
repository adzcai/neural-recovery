import numpy as np


def get_loss(n_planted, X, dmat, ind, w, w1, w2=None, tol=1e-4):
    """
    Intuitively, measures the distance to the planted neurons' weights.
    :param strict: if not None, additionally returns a flag tracking if each neuron matches its planted weight.
    """
    i_map = np.zeros(n_planted)

    sum_square = 0
    n, d = X.shape

    if w2 is None:
        w2 = np.zeros_like(w1)

    recovered = True

    for j in range(n_planted):  # for each of the planted neurons
        # get the smallest index k such that ind[k] = j
        # and therefore dmat[:, k] gives the jth vector in the _original_ dmat (before picking unique columns)
        # TODO is this always included in the new dmat though? why does this need to be included?
        k = np.nonzero(ind == j)[0][0]
        i_map[j] = k
        wj = w[:, j]
        dj = dmat[:, k]
        Xj = dj.reshape((n, 1)) * X
        _Uj, Sj, Vjh = np.linalg.svd(Xj, full_matrices=False)
        # scale the right singular vectors according to their singular values,
        # then rephrase the jth planted neuron in this basis
        wj = (Sj.reshape((d, 1)) * Vjh) @ wj
        wj /= np.linalg.norm(wj)

        dist = np.sum(
            (w1[:, k] - w2[:, k] - wj) ** 2
        )  # distance to this planted neuron
        sum_square += dist
        recovered = recovered and dist <= tol

    dis_abs = np.sqrt(sum_square)

    return i_map, dis_abs, recovered
