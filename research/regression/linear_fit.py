import numpy as np
from sklearn import linear_model

from .tree import MSEDecisionTree


class Fitter:
    def __init__(self, xs: np.ndarray, ys: np.ndarray) -> None:
        self.xs = xs
        self.ys = ys

    def decision_tree(self) -> MSEDecisionTree:
        """
        Returns the structure of a decision tree trained to classify areas of
        good and poor model performance.
        """
        return MSEDecisionTree.from_points(
            self.xs, self.ys, self.predict(self.xs)
        )

    def predict(self, xs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def evaluate(self, xs: np.ndarray, ys: np.ndarray) -> float:
        preds = self.predict(xs)
        return np.mean((preds - ys) ** 2)


class LinearFitSKL(Fitter):
    def __init__(self, xs: np.ndarray, ys: np.ndarray) -> None:
        super().__init__(xs, ys)
        self.reg = linear_model.LinearRegression().fit(xs, ys)

    def predict(self, xs: np.ndarray) -> np.ndarray:
        return self.reg.predict(xs)
