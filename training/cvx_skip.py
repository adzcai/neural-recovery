import numpy as np
import cvxpy as cp

from training.cvx_base import ConvexProgram
from utils import Args


class ConvexReLU(ConvexProgram):
    """
    Convex formulation of either plain relu network or a network with skip connection when skip is True.
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
        W = self.W_pos
        if self.W_neg is not None:
            W -= self.W_neg
        y_hat = cp.sum(cp.multiply(self.D_mat, (self.X @ W)), axis=1)
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


    def get_metrics(
        self, X: np.ndarray, W_true: np.ndarray, D_mat: np.ndarray, ind: np.ndarray, tol=Args.tol
    ) -> dict[str, float]:
        """
        We define "recovery" as the following:
        the learned weights w_skip are close to w,
        and all of the learned weights (for the relu) are close to 0.
        """
        metrics = super().get_metrics(X, W_true, D_mat, ind, tol)

        if self.w_skip is None:
            assert W_true.shape[1] == 1, "Only supported for 1 planted neuron"
            diff = W_true.flatten() - self.W_pos.value.sum(axis=1)

            return {
                "dis_abs": np.linalg.norm(diff),
                "recovery": diff < tol,
            }

        w_true = W_true.flatten()
        dis_abs = np.linalg.norm(w_true - self.w_skip.value)  # recovery error of linear weights

        recovery = np.allclose(self.w_skip.value, w_true, atol=tol) and np.allclose(
            self.W_pos.value, 0, atol=tol
        )
        if self.W_neg is not None:
            recovery = recovery and np.allclose(self.W_neg.value, 0, atol=tol)

        cos_sim = (w_true @ self.w_skip.value) / (
            np.linalg.norm(w_true) * np.linalg.norm(self.w_skip.value)
        )

        return metrics | {
            "dis_abs": dis_abs,
            "recovery": recovery,
            "cos_sim": cos_sim,
        }
