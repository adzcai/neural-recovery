import numpy as np
import matplotlib.pyplot as plt

from common import generate_X, generate_w, get_arrangement_patterns


def get_prob():
    seed = 18116275
    # seed = np.random.randint(1e8)
    print("seed = " + str(seed))
    np.random.seed(seed)
    n = 100
    n1 = n // 2
    n2 = n - n1
    sample = 5000
    dvec = np.arange(20, 100, 20)
    svec = np.arange(1, 9, 0.5)
    dlen = dvec.size
    slen = svec.size
    prob = np.zeros((slen, dlen))
    fig1, ax1 = plt.subplots()
    opt = 3  # type of \mu_1 and \mu_2

    for didx, d in enumerate(dvec):
        print("d = " + str(d))
        for sidx, sigma in enumerate(svec):
            count = 0
            if opt == 1:
                mu1 = np.ones((1, d))
                mu2 = -np.ones((1, d))
            elif opt == 2:
                mu1 = np.random.randn(1, d)
                mu2 = np.random.randn(1, d)
            elif opt == 3:
                mu1 = np.random.randn(1, d)
                mu1 = mu1 / np.linalg.norm(mu1)
                mu2 = np.random.randn(1, d)
                mu2 = mu2 / np.linalg.norm(mu2)

            for i in range(sample):
                z1 = np.random.randn(n1, d) / sigma
                z2 = np.random.randn(n2, d) / sigma
                X = np.concatenate([mu1 + z1, mu2 + z2])
                u = mu1 / np.linalg.norm(mu1) - mu2 / np.linalg.norm(mu2)
                di = X @ u.T >= 0
                if np.all(di[0:n1]) and not np.any(di[n1:n]):
                    count += 1

            prob[sidx, didx] = count / sample

    np.save("ReLU_prob" + str(opt), prob)

    for didx, d in enumerate(dvec):
        ax1.plot(svec, prob[:, didx], "o-", markersize=4, label="d = " + str(d))

    ax1.set_xlabel("$\sigma$")
    ax1.set_ylabel("Probability")
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend()
    plt.show()
    fig1.savefig("ReLU_prob" + str(opt) + ".png")


def check_irregular(n, d, args):
    eps = 1e-10
    X = generate_X(n, d, args.optx)
    w = generate_w(X, args.k, args.optw)
    dmat, ind, _i_map = get_arrangement_patterns(X, w, mh=max(n * 2, 50))

    j_array = np.nonzero(ind <= args.k - 1)[0]
    j_map = ind[j_array]

    U = np.zeros((n, 0))
    uu = []
    for jidx, j in enumerate(j_array):
        k = j_map[jidx]
        Xj = dmat[:, j].reshape((n, 1)) * X
        Uj, Sj, Vjh = np.linalg.svd(Xj, full_matrices=False)
        rj = np.linalg.matrix_rank(Xj)
        wj = (Sj.reshape((d, 1)) * Vjh) @ w[:, k]
        wj = wj / np.linalg.norm(wj)
        U = np.concatenate([U, Uj[:, np.arange(rj)]], axis=1)
        uu = np.concatenate([uu, wj[np.arange(rj)]])
    lam = U @ np.linalg.pinv(U.T @ U) @ uu

    m1 = dmat.shape[1]
    count = 0
    for j in range(m1):
        if j in j_array:
            continue
        dj = dmat[:, j]
        Xj = dj.reshape((n, 1)) * X
        Uj, Sj, Vjh = np.linalg.svd(Xj, full_matrices=False)
        if np.linalg.norm(Uj.T @ lam) >= 1 + eps:
            count += 1

    return count == 0
