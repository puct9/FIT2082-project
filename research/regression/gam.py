import numpy as np
from pygam import LinearGAM
from pygam import s as gam_s
from pygam.terms import TermList

from .linear_fit import Fitter


class GAMFit(Fitter):
    @staticmethod
    def _splines(n: int) -> TermList:
        r = gam_s(0)
        for x in range(1, n):
            r += gam_s(x)
        return r

    def __init__(self, xs: np.ndarray, ys: np.ndarray) -> None:
        super().__init__(xs, ys)
        self.reg = LinearGAM(GAMFit._splines(xs.shape[1])).fit(xs, ys)

    def predict(self, xs: np.ndarray) -> np.ndarray:
        return self.reg.predict(xs)
