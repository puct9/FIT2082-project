from typing import List, Type

import numpy as np

from ..datasets.bases import Dataset
from ..regression.linear_fit import Fitter
from ..regression.rf import RandomForestFit
from ..regression.tree import MSEDecisionTree
from .evaluation_statistics import EvalStats


def evaluate(
    dataset: Dataset,
    fitter_cls: Type[Fitter],
    strong_fitter_cls: Type[Fitter] = RandomForestFit,
    retrain_on_filtered: bool = False,
) -> List[EvalStats]:
    """Evaluate a dataset

    Args:
        dataset: Dataset
        fitter_cls: Weak fitter class
        strong_fitter_cls: Strong fitter class to use as performance baseline
        retrain_on_filtered: Set to True if weak fitter class should be
            retrained on filtered data

    Returns:
        List of `EvalStats`. Use `stats_to_df` to convert list to `DataFrame`.
    """
    # Get unbiased estimates for points in training set
    x_inputs = []
    y_estimates = []
    y_targets = []
    for xs, ys, xs_val, ys_val in dataset.generate_folds(10):
        fitter = fitter_cls(xs, ys)
        preds = fitter.predict(xs_val)
        x_inputs.append(xs_val)
        y_estimates.append(preds)
        y_targets.append(ys_val)
    x_inputs = np.concatenate(x_inputs)
    y_estimates = np.concatenate(y_estimates)
    y_targets = np.concatenate(y_targets)

    # Baseline score against stronger model
    strong_fit = strong_fitter_cls(dataset.x_train, dataset.y_train)
    strong_loss = np.square(
        dataset.y_test - strong_fit.predict(dataset.x_test)
    ).mean()

    # Create decision tree and evaluate on testing (holdout) set
    tree = MSEDecisionTree.from_points(x_inputs, y_targets, y_estimates)

    # Retrain on entire training set
    fitter = fitter_cls(dataset.x_train, dataset.y_train)
    eval_base = fitter.predict(dataset.x_test)

    eval_stats: List[EvalStats] = []

    increment = 0.1
    for coverage in np.arange(increment, 1 + increment, increment):
        search_info = tree.suggest_subsets(min_coverage=coverage, num=None)
        rules = search_info.rule
        where_train = rules.mask(dataset.x_train)
        if retrain_on_filtered:
            x_train_filtered = dataset.x_train[where_train]
            y_train_filtered = dataset.y_train[where_train]
            fitter = fitter_cls(x_train_filtered, y_train_filtered)
            eval_base = fitter.predict(dataset.x_test)
        where = rules.where(dataset.x_test)
        xs_filtered = dataset.x_test[where]
        eval_filtered = fitter.predict(xs_filtered)
        eval_stats.append(
            EvalStats(
                dataset.x_test,
                dataset.y_test,
                where,
                eval_base,
                eval_filtered,
                search_info.metric,
                coverage,  # target coverage
                where_train.mean(),  # expected coverage from search
                strong_loss,
            )
        )

    return eval_stats
