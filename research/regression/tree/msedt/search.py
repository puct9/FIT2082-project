# Search algorithm for suggesting best combination of rules
from typing import TYPE_CHECKING, List, Optional, Sequence

from ..bases import SearchStatistics
from ..rules import RuleSet

if TYPE_CHECKING:
    from .tree import MSEDecisionTree, MSEDecisionTreeNode


class SearchNode:
    def __init__(
        self,
        dt_node: "MSEDecisionTreeNode",
        parent: "SearchNode",
        all_dt_nodes: Sequence["MSEDecisionTreeNode"] = None,
    ):
        # Corresponding decision tree node
        self.dt_node = dt_node
        self.parent = parent
        if parent is not None:
            self.samples: int = parent.samples + dt_node.samples
            self.avgloss: float = (
                parent.avgloss * parent.samples
                + dt_node.avgloss * dt_node.samples
            ) / self.samples
            self.depth = parent.depth + 1
            self._children_cache = self.filter_valid_children(
                parent._children_cache
            )
        else:
            # Root node condition
            self.samples = 0
            self.avgloss: float = 0
            self.depth = 0
            if all_dt_nodes is None:
                raise ValueError(
                    "all_dt_nodes cannot be None if no parent is provided"
                )
            self._children_cache = all_dt_nodes
        self.children: Optional[Sequence["SearchNode"]] = None
        self.terminal = False

    def validate_terminal(self, target_samples: int, max_depth: int) -> None:
        # Cannot be empty
        self.terminal |= not self._children_cache
        # Must be less than target
        self.terminal |= self.samples > target_samples
        # Must be less than depth
        self.terminal |= self.depth >= max_depth

    def expand(self) -> None:
        if self.children is not None or self.terminal:
            raise ValueError("Node already expanded or is terminal")
        self.children = []
        for dt_node in self._children_cache:
            self.children.append(SearchNode(dt_node, self))

    def filter_valid_children(
        self, dt_nodes: Sequence["MSEDecisionTreeNode"]
    ) -> Sequence["MSEDecisionTreeNode"]:
        if self.dt_node is None:
            return dt_nodes
        res = []
        for node in dt_nodes:
            # Pruning tactics
            if (
                node.position <= self.dt_node.position
                or node.is_ancestor_of(self.dt_node)
                or node.is_child_of(self.dt_node)
            ):
                continue
            else:
                res.append(node)
        return res

    def walk_to_leaf(self) -> Optional["SearchNode"]:
        if self.children is None:
            return self
        for child in self.children:
            if not child.terminal:
                return child.walk_to_leaf()
        # Non children are non-terminal; update our state
        self.terminal = True


def search(
    tree: "MSEDecisionTree", min_coverage: float = 0.1, num: int = None
) -> SearchStatistics:
    """Generate optimal rule for target coverage

    Args:
        tree: MSE decision tree
        min_coverage: Minimum expected coverage allowed. Defaults to 0.1.
        num: Maximum number of nodes. Set to `None` (recommended) for no limit.

    Returns:
        Search statistics
    """
    if num is None or num == float("inf"):
        return search_all(tree, min_coverage)
    # `tree` is a GoodBadDecisionTree
    min_samples = tree.tree.samples * min_coverage
    top_node = SearchNode(None, None, tree.nodes)
    best_node = None
    top_node.expand()
    while not top_node.terminal:
        leaf = top_node.walk_to_leaf()
        if leaf is None:
            continue
        # Check if leaf is better than current best
        if best_node is None or (
            leaf.samples >= min_samples and leaf.avgloss < best_node.avgloss
        ):
            best_node = leaf
        leaf.validate_terminal(min_samples, num)
        if not leaf.terminal:
            leaf.expand()

    # Extract conditions from best node
    rules = []
    node = best_node
    while node.dt_node is not None:
        rules.append(node.dt_node)
        node = node.parent
    conjunctions = [rule.conditions() for rule in rules]
    return SearchStatistics(
        "Expected loss", best_node.avgloss, RuleSet(conjunctions, "|")
    )


def search_all(
    tree: "MSEDecisionTree", min_coverage: float = 0.1
) -> SearchStatistics:
    ordered_leaves = sorted(
        tree.tree.generate_leaves(), key=lambda x: x.metric, reverse=True
    )
    used_leaves: List["MSEDecisionTreeNode"] = []
    cumulative_coverage = 0
    while cumulative_coverage < min_coverage - 1e-6:  # XXX: Use an epsilon
        leaf = ordered_leaves.pop()  # This list is in reversed order
        cumulative_coverage += leaf.coverage
        used_leaves.append(leaf)
    # Calculate weighted average loss
    avgloss = (
        sum(node.avgloss * node.coverage for node in used_leaves)
        / cumulative_coverage
    )
    conjunctions = [node.conditions() for node in used_leaves]
    return SearchStatistics(
        "Expected loss", avgloss, RuleSet(conjunctions, "|")
    )
