from abc import ABC
import cvxpy as cp
import numpy as np

from training.common import Variables, generate_X, generate_y
from utils import ALL_FORMS, Args


class ConvexProgram(ABC):
    def __init__(self, form: str, X: np.ndarray, y: np.ndarray, D_mat: np.ndarray, beta: float):
        super().__init__()

        assert (
            form in ALL_FORMS and form != "gd"
        ), "form must be one of 'exact', 'approx', or 'relaxed'."

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
        variables = {}
        for possible_var in ("W", "W_pos", "W_neg", "w_skip"):
            if getattr(self, possible_var, None) is not None:
                variables[possible_var] = getattr(self, possible_var).value
        return Variables(variables)

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

        W = self.W_pos.value
        if self.W_neg is not None:
            W -= self.W_neg.value
        return {
            "other_norm": np.linalg.norm(W, ord="fro"),
        }

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
            W_true,
            relu=planted != "linear",
            normalize=planted == "normalized",
        )
        y_hat = self.predict(X_test)
        squared_err = (y_hat - y_true) ** 2
        test_err = np.mean(squared_err)
        test_dis = np.sqrt(squared_err.sum())
        return test_err, test_dis
