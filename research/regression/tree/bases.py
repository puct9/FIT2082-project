from typing import Generator, Optional, Tuple

from .rules import Rule, RuleSet


class DecisionTreeNodeBase:
    __slots__ = (
        "left",
        "right",
        "parent",
        "condition",
        "samples",
        "coverage",
        "position",
        "depth",
    )

    def __init__(self) -> None:
        self.left: Optional["DecisionTreeNodeBase"] = None
        self.right: Optional["DecisionTreeNodeBase"] = None
        self.parent: Optional["DecisionTreeNodeBase"] = None
        # Condition is always <=
        # Format (dimension, threshold)
        self.condition: Tuple[int, float] = None
        self.samples: int = None
        self.coverage: float = None
        # Integer representing position of node in tree (using bits)
        self.position: int = None
        self.depth: int = None

    @property
    def metric(self) -> float:
        raise NotImplementedError

    @metric.setter
    def metric(self, value: float) -> None:
        raise NotImplementedError

    def set_children(
        self, left: "DecisionTreeNodeBase", right: "DecisionTreeNodeBase"
    ):
        left.parent = self
        right.parent = self
        left.position = (self.position << 1) | 1
        right.position = self.position << 1
        left.depth = self.depth + 1
        right.depth = self.depth + 1
        self.left = left
        self.right = right

    def generate_nodes(self) -> Generator["DecisionTreeNodeBase", None, None]:
        # Returns a list of child nodes including itself
        yield self
        if self.left is not None:
            yield from self.left.generate_nodes()
            yield from self.right.generate_nodes()

    def generate_leaves(self) -> Generator["DecisionTreeNodeBase", None, None]:
        # Gets all leaves below including itself if applicable
        if self.left is None:
            yield self
        else:
            yield from self.left.generate_leaves()
            yield from self.right.generate_leaves()

    def conditions(self) -> RuleSet:
        # Get all conditions required to reach this node
        res = []
        node = self
        while node.parent is not None:
            is_left_child = node.parent.left is node
            res.append(node.parent.condition + (is_left_child,))
            node = node.parent
        # Logic should be commutative but reverse it for clarity
        rules = [
            Rule(rule[0], "<=" if rule[2] else ">", rule[1])
            for rule in res[::-1]
        ]
        return RuleSet(rules, "&")

    def is_child_of(self, other: "DecisionTreeNodeBase") -> bool:
        shift = self.depth - other.depth
        if shift <= 0:
            return False
        shifted = self.position >> shift
        return shifted == other.position

    def is_ancestor_of(self, other: "DecisionTreeNodeBase") -> bool:
        return other.is_child_of(self)


class SearchStatistics:
    def __init__(self, metric_name: str, metric: float, rule: RuleSet) -> None:
        self.metric_name = metric_name
        self.metric = metric
        self.rule = rule

    def __repr__(self) -> str:
        return (
            f"SearchStatistics('{self.metric_name}', {self.metric}, "
            f"{self.rule})"
        )
