from typing import Sequence

import numpy as np


class RuleBase:
    def mask(self, arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def where(self, arr: np.ndarray) -> np.ndarray:
        return np.where(self.mask(arr))[0]

    def filter(self, arr: np.ndarray) -> np.ndarray:
        return arr[self.where(arr)]

    def coverage(self, arr: np.ndarray) -> float:
        return self.mask(arr).mean()


class Rule(RuleBase):
    MAPPING = {
        "<": "__lt__",
        ">": "__gt__",
        "<=": "__le__",
        ">=": "__ge__",
    }
    REVERSE = {v: k for k, v in MAPPING.items()}

    def __init__(self, feature: int, comparator: str, value: float) -> None:
        self.feature = feature
        self.comparator = Rule.MAPPING[comparator]
        self.value = value

    def __repr__(self) -> str:
        return (
            f"Rule({self.feature}, '{Rule.REVERSE[self.comparator]}', "
            f"{self.value})"
        )

    def mask(self, arr: np.ndarray) -> np.ndarray:
        # arr is ndarray of shape (N, features)
        return getattr(arr[:, self.feature], self.comparator)(self.value)


class RuleSet(RuleBase):
    MAPPING = {"&": "__and__", "|": "__or__"}
    REVERSE = {v: k for k, v in MAPPING.items()}

    def __init__(self, rules: Sequence[RuleBase], reduction: str) -> None:
        # `reduction` must be one of '&' or '|'
        self.rules = rules
        self.reduction = RuleSet.MAPPING[reduction]

    def __repr__(self) -> str:
        return f"RuleSet({self.rules}, '{RuleSet.REVERSE[self.reduction]}')"

    def mask(self, arr: np.ndarray) -> np.ndarray:
        if self.reduction == "__and__":
            mask = np.ones(arr.shape[0], dtype=bool)
        else:
            mask = np.zeros(arr.shape[0], dtype=bool)
        for rule in self.rules:
            mask = getattr(mask, self.reduction)(rule.mask(arr))
        return mask
