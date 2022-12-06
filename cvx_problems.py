from abc import ABC
from collections import namedtuple
import cvxpy as cp
import numpy as np

from common import mult_diag


Variables = namedtuple("Variables", ["W_pos", "W_neg", "w_skip"], defaults=[None] * 3)


class ConvexProgram(ABC):
    def get_objective(self) -> cp.Variable:
        pass

    def get_constraints(self) -> list[cp.Variable]:
        pass

    def get_variables(self) -> Variables:
        """
        Get a namedtuple of the variables in this problem.
        """
        variables = Variables()
        if getattr(self, "W", None) is not None:
            variables = variables._replace(W_pos=self.W.value)
        if getattr(self, "W_pos", None) is not None:
            variables = variables._replace(W_pos=self.W_pos.value)
        if getattr(self, "W_neg", None) is not None:
            variables = variables._replace(W_neg=self.W_neg.value)
        if getattr(self, "w_skip", None) is not None:
            variables = variables._replace(w_skip=self.w_skip.value)
        return variables

    def solve(self) -> tuple[cp.Problem, float, Variables]:
        """
        Compiles the objective and constraints and solves this convex program using Mosek.
        """
        prob = cp.Problem(cp.Minimize(self.get_objective()), self.get_constraints())
        opt = prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False, mosek_params={})
        return prob, opt, self.get_variables()


class ConvexReLU(ConvexProgram):
    """
    Convex formulation of either plain relu network or a network with skip connection when skip is True.

    arguments                |  equation and page number in the paper
    -------------------------+-------------------------------------------------
    not exact and skip,      |  211 (bottom of p. 57)
    not exact and not skip   |  211 (skip connection removed)
    exact and skip           |  6 (top of p. 5), with the w_0 norm added (we think this was a typo)
    exact and not skip       |  11 (top of p. 8)
    """

    def __init__(self, X, y, D_mat, beta, skip=False, exact=False):
        super().__init__()
        self.X = X
        self.y = y
        self.D_mat = D_mat
        self.beta = beta
        self.exact = exact

        d = X.shape[1]
        p = D_mat.shape[1]

        self.W_pos = cp.Variable((d, p), "W_pos")
        self.W_neg = cp.Variable((d, p), "W_neg")
        if skip:
            self.w_skip = cp.Variable(d, "W_skip")
        else:
            self.w_skip = None

        y_pos = cp.sum(self.D_mat * (self.X @ self.W_pos), axis=1)
        y_neg = cp.sum(self.D_mat * (self.X @ self.W_neg), axis=1)
        self.residual = y_pos - y_neg - y
        if skip:
            self.residual += self.X @ self.w_skip

    def get_objective(self):
        norm = cp.mixed_norm(self.W_pos.T, 2, 1) + cp.mixed_norm(self.W_neg.T, 2, 1)
        if self.w_skip is not None:
            norm += cp.norm(self.w_skip, 2)
        if self.exact:
            return norm
        else:
            return cp.sum_squares(self.residual) + self.beta * norm

    def get_constraints(self):
        signed_patterns = 2 * self.D_mat - 1
        constraints = [
            signed_patterns * (self.X @ self.W_pos) >= 0,
            signed_patterns * (self.X @ self.W_neg) >= 0,
        ]
        if self.exact:
            constraints += [self.residual == 0]
        return constraints


class ConvexReLURelaxed(ConvexProgram):
    """
    Equation 15 of the paper (bottom of p. 8).
    This is a relaxed version of Equation 6 (same as equation 14),
    which is implemented above.
    The inequality constraints are dropped.
    """

    def __init__(self, X, y, D_mat, skip=False) -> tuple[cp.Problem, Variables]:
        super().__init__()
        self.X = X
        self.y = y
        self.D_mat = D_mat

        d = X.shape[1]
        p = D_mat.shape[1]

        self.W = cp.Variable((d, p), "W")  # relu connections (second layer is all ones)
        if skip:
            self.w_skip = cp.Variable(d, "W_skip")  # skip connections
        else:
            self.w_skip = None

    def get_objective(self):
        obj = cp.mixed_norm(self.W.T, 2, 1)
        if self.w_skip is not None:
            obj += cp.norm(self.w_skip, 2)
        return obj

    def get_constraints(self):
        y_hat = cp.sum(self.D_mat * (self.X @ self.W), axis=1)
        if self.w_skip is not None:
            y_hat += self.X @ self.w_skip
        return [y_hat == self.y]


class ConvexReLUNormalized(ConvexProgram):
    """
    Implement the convex optimization problem in cvxpy.
    If "exact" is True, this is an implementation of Equation 16 (top of page 9).
    Otherwise, this implements the approximate form in Equation 212.
    This is equivalent to the regularized training of a two-layer relu network with normalization.
    """

    def __init__(self, X, y, D_mat, beta, exact=False):
        super().__init__()
        n, d = X.shape
        assert d <= n, "d must be less than or equal to n"
        p = D_mat.shape[1]

        self.beta = beta
        self.exact = exact
        self.D_mat = D_mat

        self.W_pos = cp.Variable((d, p), "W_pos")
        self.W_neg = cp.Variable((d, p), "W_neg")

        U, S, _Vh = np.linalg.svd(mult_diag(D_mat, X), full_matrices=False)
        self.mask = S > 1e-12
        S[self.mask] = 1 / S[self.mask]  # take inverse of positive singular values
        self.S_inv = S
        self.residual = (
            np.transpose(U, (1, 0, 2))[:, self.mask] @ (self.W_pos.T[self.mask] - self.W_neg.T[self.mask]) - y
        )

    def get_objective(self):
        norm = cp.mixed_norm(self.W_pos.T, 2, 1) + cp.mixed_norm(self.W_neg.T, 2, 1)
        if self.exact:
            return norm
        else:
            return cp.sum_squares(self.residual) + self.beta * norm

    def get_constraints(self):
        C = np.einsum("np, nd, pmd, pm -> npm", 2 * self.D_mat - 1, self.X, self.Vh, self.S_inv, optimize=True)
        C = C[:, self.mask]
        constraints = [C @ self.W_pos.T[self.mask] >= 0, C @ self.W_neg.T[self.mask] >= 0]
        if not np.all(self.mask):
            constraints += [self.W_pos.T[~self.mask] == 0, self.W_neg.T[~self.mask] == 0]
        if self.exact:
            constraints += [self.residual == 0]
        return constraints


class ConvexReLUNormalizedRelaxed(ConvexProgram):
    """
    Implement Equation 17 in the paper (middle of page 9).
    After dropping all inequality constraints in Equation 16 (above),
    and we're left with the simple group l1 norm minimization problem implemented below.
    """

    def __init__(self, X, y, D_mat):
        self.X = X
        self.y = y
        self.D_mat = D_mat

        d = X.shape[1]
        p = D_mat.shape[1]

        self.W = cp.Variable((d, p), "W")

    def get_objective(self):
        return cp.mixed_norm(self.W.T, 2, 1)

    def get_constraints(self):
        U, S, _Vh = np.linalg.svd(mult_diag(self.D_mat, self.X), full_matrices=False)
        mask = S > 1e-12
        y_hat = np.transpose(U, (1, 0, 2))[:, mask] @ self.W.T[mask]
        constraints = [y_hat == self.y]
        if not np.all(mask):
            constraints += [self.W.T[~mask] == 0]
        return constraints
