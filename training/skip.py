import numpy as np
import cvxpy as cp

from training.convex_program import ConvexProgram
from utils import Args


class ConvexReLU(ConvexProgram):
    """
    Convex formulation of either plain relu network or a network with skip connection when skip is True.

    form    | skip |  equation and page number in the paper
    --------+------+---------------------------------------
    approx  |  y   |  211 (bottom of p. 57)
    approx  |  n   |  211 (skip connection removed)
    exact   |  y   |  6 (top of p. 5), with the w_0 norm added (we think this was a typo)
    exact   |  n   |  11 (top of p. 8)
    relaxed |  y   |  15 (bottom of p. 8).
    """

    def __init__(
        self,
        form: str,
        X: np.ndarray,
        y: np.ndarray,
        D_mat: np.ndarray,
        beta: float,
        skip: bool,
    ):
        super().__init__(form, X, y, D_mat, beta)

        d = X.shape[1]

        self.w_skip = cp.Variable(d, "w_skip") if skip else None

        self.residual = self.training_predict() - self.y

    def get_objective(self):
        norm = cp.mixed_norm(self.W_pos.T, 2, 1)
        if self.W_neg is not None:
            norm += cp.mixed_norm(self.W_neg.T, 2, 1)
        if self.w_skip is not None:
            norm += cp.norm(self.w_skip, 2)

        if self.form == "approx":
            return cp.sum_squares(self.residual) + self.beta * norm
        else:
            return norm

    def get_constraints(self):
        if self.form == "relaxed":
            return [self.residual == 0]

        signed_patterns = 2 * self.D_mat - 1  # (n, p)
        constraints = [cp.multiply(signed_patterns, self.X @ self.W_pos) >= 0]
        if self.W_neg is not None:
            constraints += [cp.multiply(signed_patterns, self.X @ self.W_neg) >= 0]
        if self.form == "exact":
            constraints += [self.residual == 0]
        return constraints

    def training_predict(self):
        """
        Constrain this to be equal to y, or close to it (in the approximate case).
        """
        y_hat = cp.sum(cp.multiply(self.D_mat, (self.X @ self.W_pos)), axis=1)
        if self.W_neg is not None:
            y_hat -= cp.sum(cp.multiply(self.D_mat, (self.X @ self.W_neg)), axis=1)
        if self.w_skip is not None:
            y_hat += self.X @ self.w_skip
        return y_hat

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_hat = np.maximum(0, X @ self.W_pos.value)
        if self.W_neg is not None:
            y_hat -= np.maximum(0, X @ self.W_neg.value)
        y_hat = y_hat.sum(axis=1)
        if self.w_skip is not None:
            y_hat += X @ self.w_skip.value
        return y_hat

    def get_metrics_plain(self, W_true: np.ndarray):
        dis_abs = 0  # np.linalg.norm(W_true - W_pos)  # recovery error of linear weights

        recovery = True
        for neuron in W_true.T:
            recovery = recovery and np.any(
                [
                    np.allclose(neuron / np.linalg.norm(neuron), w_pos)
                    for w_pos in self.W_pos.value.T
                ]
            )
            # recovery = recovery and np.any([np.allclose(0, w_neg) for w_neg in W_neg.T])

        # print the highest 10 norms of columns of W_pos
        # print("w_true", W_true)
        # print("y", y_true)
        # print("W_pos norms", np.sort(np.linalg.norm(W_pos, axis=0))[-10:])

        return {
            "dis_abs": dis_abs,
            "recovery": recovery,
        }

    def get_metrics(
        self, X: np.ndarray, W_true: np.ndarray, D_mat: np.ndarray, ind: np.ndarray, tol=Args.tol
    ) -> dict[str, float]:
        """
        We define "recovery" as the following:
        the learned weights w_skip are close to w,
        and all of the learned weights (for the relu) are close to 0.
        """

        w_true = W_true.flatten()
        dis_abs = np.linalg.norm(w_true - self.w_skip.value)  # recovery error of linear weights

        recovery = np.allclose(self.w_skip.value, w_true, atol=tol) and np.allclose(
            self.W_pos.value, 0, atol=tol
        )
        if self.W_neg is not None:
            recovery = recovery and np.allclose(self.W_neg.value, 0, atol=tol)

        return {
            "dis_abs": dis_abs,
            "recovery": recovery,
        }
