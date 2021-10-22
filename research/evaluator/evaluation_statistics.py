from typing import List

import numpy as np
import pandas as pd


class EvalStats:
    """Container for evaluation statistics"""

    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        where: np.ndarray,
        eval_base: np.ndarray,
        eval_filtered: np.ndarray,
        expected: float,
        target_coverage: float,
        expected_coverage: float,
        comparison_loss: float,
    ) -> None:
        self.xs = xs
        self.ys = ys
        self.where = where
        self.eval_base = eval_base
        self.eval_filtered = eval_filtered
        self.mse_expected = expected
        self.target_coverage = target_coverage
        self.expected_coverage = expected_coverage
        self.comparison_loss = comparison_loss

        # Statistics
        self.se_base = np.square(ys - eval_base)
        self.se_filtered = np.square(ys[where] - eval_filtered)
        self.mse_base = np.mean(self.se_base)
        self.mse_filtered = np.mean(self.se_filtered)
        self.mse_prop_base = self.mse_filtered / self.mse_base
        self.coverage = where.shape[0] / xs.shape[0]


def stats_to_df(
    stats: List[EvalStats], columns: List[str], dataset: str = None
) -> pd.DataFrame:
    """
    Convert list of `EvalStats` to `DataFrame`. Useful for further analysis.

    Args:
        stats: List of `EvalStats` (typically from
            `research.evaluator.evaluate_dataset.evaluate`)
        columns: List of names of columns
        dataset: Optional dataset name

    Returns:
        `DataFrame` of results
    """
    dct = {}
    if dataset is not None:
        dct["dataset"] = dataset
    for col in columns:
        dct[col] = [getattr(s, col) for s in stats]
    return pd.DataFrame(data=dct)
