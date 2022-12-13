from abc import ABC
import cvxpy as cp
import numpy as np

from training.common import Variables, generate_X, generate_y
from utils import Args


class ConvexProgram(ABC):
    def __init__(self, form: str, X: np.ndarray, y: np.ndarray, D_mat: np.ndarray, beta: float):
        super().__init__()

        self.X = X
        self.y = y
        self.D_mat = D_mat
        self.beta = beta
        self.form = form

        d = X.shape[1]
        p = D_mat.shape[1]

        self.W_pos = cp.Variable((d, p), "W_pos")
        self.W_neg = cp.Variable((d, p), "W_neg") if self.form != "relaxed" else None

    def get_objective(self) -> cp.Variable:
        raise NotImplementedError

    def get_constraints(self) -> list[cp.Variable]:
        raise NotImplementedError

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict y (n,) from X (n, d) using the trained weights.
        """
        raise NotImplementedError

    def solve(self) -> tuple[cp.Problem, float, Variables]:
        """
        Compiles the objective and constraints and solves this convex program using Mosek.
        """
        obj, constraints = self.get_objective(), self.get_constraints()
        problem = cp.Problem(cp.Minimize(obj), constraints)
        opt = problem.solve(solver=cp.MOSEK, warm_start=True, verbose=False, mosek_params={})
        return problem, opt

    def get_metrics(
        self, X: np.ndarray, W_true: np.ndarray, D_mat: np.ndarray, ind: np.ndarray, tol=Args.tol
    ) -> dict[str, float]:
        """
        :param X: (n, d) design matrix
        :param W_true: (d, k) planted weights of the ReLU
        :param D_mat: (d, p) matrix of the diagonal arrangement patterns
        :param ind: (p,) indices of the arrangement patterns in the initial random order
        :param tol: the tolerance to use for checking if a learned neuron matches the grounded truth
        """
        raise NotImplementedError

    def get_test_err(
        self,
        n: int,
        d: int,
        cubic: bool,
        whiten: bool,
        planted: str,
        W_true,
    ):
        """
        Get the test error of the trained model on a new design matrix of size (n, d) and the given distribution.
        """
        X_test = generate_X(n, d, cubic=cubic, whiten=whiten)
        y_true = generate_y(
            X_test,
            Variables(W_pos=W_true),
            relu=planted != "linear",
            normalize=planted == "normalized",
        )
        y_hat = self.predict(X_test)
        test_err = np.linalg.norm(y_hat - y_true)
        return test_err
