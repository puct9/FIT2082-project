# Utilities for converting scikit-learn decision trees into a more interpreable
# format
from typing import List, Optional

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from ..bases import DecisionTreeNodeBase, SearchStatistics
from ..rules import RuleSet
from .search import search


class MSEDecisionTree:
    @classmethod
    def from_points(
        cls, xs: np.ndarray, ys: np.ndarray, preds: np.ndarray
    ) -> "MSEDecisionTree":
        """
        Returns the structure of a decision tree trained to classify areas of
        good and poor model performance.

        Args:
            xs: Predictors of shape (N, P) where P is the number of predictors
            ys: Targets of shape (N,)
            preds: Predictions of shape (N,)
        """
        losses = (preds - ys) ** 2
        # Decision tree classifier
        clf = DecisionTreeRegressor(
            max_depth=xs.shape[1],
            min_samples_split=max(1, xs.shape[0] // 10),
        ).fit(xs, losses)

        tree = clf.tree_
        children_left = tree.children_left
        children_right = tree.children_right
        # Feature and threshold for condition
        features = tree.feature
        thresholds = tree.threshold
        # Comes in shape (N, 1, 1) for some reason
        expected_values = tree.value.squeeze(-1).squeeze(-1)
        samples = tree.n_node_samples
        nodes: List[Optional["MSEDecisionTreeNode"]] = [None] * tree.node_count

        # Add nodes to the list and set their children and parents
        stack = [0]
        nodes[0] = MSEDecisionTreeNode()
        nodes[0].position = 1
        nodes[0].depth = 0
        while stack:
            idx = stack.pop()
            node = nodes[idx]
            # Update node information
            node.condition = (features[idx], thresholds[idx])
            node.metric = expected_values[idx]
            node.samples = samples[idx]
            node.coverage = samples[idx] / nodes[0].samples
            # Inspect children
            left_idx = children_left[idx]
            right_idx = children_right[idx]
            # This node has no children if these values are identical
            if left_idx == right_idx:
                continue
            stack.append(left_idx)
            stack.append(right_idx)
            # Update tree
            left = MSEDecisionTreeNode()
            right = MSEDecisionTreeNode()
            nodes[left_idx] = left
            nodes[right_idx] = right
            node.set_children(left, right)

        return cls(clf, nodes)

    def __init__(self, clf, nodes: List["MSEDecisionTreeNode"]):
        self.clf = clf
        self.nodes = nodes
        self.tree = nodes[0]

    def suggest_subsets(self, min_coverage=0.1, num=1) -> SearchStatistics:
        return search(self, min_coverage, num)

    def all_leaves_subsets(self):
        nodes = filter(
            lambda x: x.left is None and x.splits[0] < x.splits[1], self.nodes
        )
        conjunctions = [node.conditions() for node in nodes]
        return RuleSet(conjunctions, "|")


class MSEDecisionTreeNode(DecisionTreeNodeBase):
    __slots__ = DecisionTreeNodeBase.__slots__ + ("avgloss",)

    def __init__(self) -> None:
        super().__init__()
        self.avgloss: float = None

    @property
    def metric(self) -> float:
        return self.avgloss

    @metric.setter
    def metric(self, value: float) -> None:
        self.avgloss = value
