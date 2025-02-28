import numpy as np
import cvxpy as cp

from training.common import idx_of_planted_in_patterns, mult_diag
from training.cvx_base import ConvexProgram
from utils import Args


class ConvexReLUNormalized(ConvexProgram):
    """
    Implement the convex optimization problem in cvxpy.
    form    | equation
    --------+---------
    exact   | 16 (top of page 9)
    approx  | 212 (bottom of page 9)
    relaxed | 17 (middle of page 9)

    This is equivalent to the regularized training of a two-layer relu network with normalization.
    """

    def __init__(self, form, X, y, D_mat, beta):
        super().__init__(form, X, y, D_mat, beta)

        n, d = X.shape
        assert d <= n, "d must be less than or equal to n"

        self.U_masked, self.S_inv, self.Vh, self.mask = self.get_svd()
        self.residual = self.predict_training() - y

    def get_svd(self, tol=1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the positive left singular vectors across all arrangement patterns.
        """
        # (p, n, d) | (p, d) | (p, d, d)
        U, S, Vh = np.linalg.svd(mult_diag(self.D_mat, self.X), full_matrices=False)
        mask = S > tol
        U_masked = np.transpose(U, (1, 0, 2))[:, mask]

        S[mask] = 1 / S[mask]  # take inverse of positive singular values
        return U_masked, S, Vh, mask

    def get_objective(self):
        norm = cp.mixed_norm(self.W_pos.T, 2, 1)
        if self.W_neg is not None:
            norm += cp.mixed_norm(self.W_neg.T, 2, 1)
        if self.form == "approx":
            return cp.sum_squares(self.residual) + self.beta * norm
        else:
            return norm

    def get_constraints(self):
        constraints = []

        # not included to the paper, but in their code.
        # set all the weights to zero for the singular vectors that are not used
        if not np.all(self.mask):
            constraints += [self.W_pos.T[~self.mask] == 0]
            if self.W_neg is not None:
                constraints += [self.W_neg.T[~self.mask] == 0]

        if self.form in ("exact", "relaxed"):
            constraints += [self.residual == 0]

        if self.form == "relaxed":
            return constraints

        # p indexes the arrangement patterns
        # n indexes the data points
        # d indexes the features
        # m = d indexes the singular vectors
        C = np.einsum(
            "np, nd, pmd, pm -> npm",
            2 * self.D_mat - 1,
            self.X,
            self.Vh,
            self.S_inv,
            optimize=True,
        )
        C = C[:, self.mask]

        constraints = [C @ self.W_pos.T[self.mask] >= 0]
        if self.W_neg is not None:
            constraints += [C @ self.W_neg.T[self.mask] >= 0]

        return constraints

    def predict_training(self):
        W = self.W_pos.T[self.mask]
        if self.W_neg is not None:
            W -= self.W_neg.T[self.mask]
        return self.U_masked @ W

    def predict(self, X: np.ndarray) -> np.ndarray:
        W = self.W_pos.value
        if self.W_neg is not None:
            W -= self.W_neg.value
        y_hat = X @ W  # (n, p)
        y_hat /= np.linalg.norm(y_hat, axis=0)
        return y_hat.sum(axis=1)

    def get_metrics(
        self,
        X: np.ndarray,
        W_true: np.ndarray,
        D_mat: np.ndarray,
        ind: np.ndarray,
        tol=Args.tol,
    ):
        """
        Measures the distance to the planted neurons' weights.
        Mathematically, we express the jth planted neuron in terms of
        the singular values of the subset of X given by Dj @ X,
        where Dj is the corresponding diagonal arrangement pattern,
        and then compare the learned neurons with this expression.
        """
        metrics = super().get_metrics(X, W_true, D_mat, ind, tol)
        k = W_true.shape[1]

        # element of [p]^k. the indices of the planted neurons in D_mat
        s = idx_of_planted_in_patterns(ind, k)
        X_scaled = mult_diag(D_mat[:, s], X)  # (k, n, d)

        # (k, n, d), (k, d), (k, d, d). assumes d < n
        _U, S, Vh = np.linalg.svd(X_scaled, full_matrices=False)

        # (k, d, d). Scale the right singular vectors by the singular values.
        Vh_scaled = mult_diag(S.T, Vh)

        W_rotated = np.einsum("kdf, fk -> dk", Vh_scaled, W_true, optimize=True)
        W_rotated /= np.linalg.norm(W_rotated, axis=0)

        diff = self.W_pos.value[:, s] - W_rotated  # all have shape (d, k)
        if self.W_neg is not None:
            diff -= self.W_neg.value[:, s]

        dis_abs = np.linalg.norm(diff, ord="fro")
        recovery = np.allclose(diff, 0, atol=tol)

        return metrics | {
            "dis_abs": dis_abs,
            "recovery": recovery,
        }
