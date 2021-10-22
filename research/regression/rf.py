import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .linear_fit import Fitter


class RandomForestFit(Fitter):
    def __init__(self, xs: np.ndarray, ys: np.ndarray) -> None:
        super().__init__(xs, ys)
        self.reg = RandomForestRegressor()
        self.reg.fit(xs, ys)

    def predict(self, xs: np.ndarray) -> np.ndarray:
        return self.reg.predict(xs)
