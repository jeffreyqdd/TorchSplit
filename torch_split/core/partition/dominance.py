"""Dominance analysis for TorchGraph augmented with ENTRY and EXIT nodes."""

from pathlib import Path
import uuid
from collections import deque
from dataclasses import dataclass

import graphviz  # type: ignore
from frozendict import frozendict

from torch_split.core.ir import TorchGraph, ConcreteNode
import torch_split.core.log as logging

logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class VirtualNode(uuid.UUID):
    """Nodes which have a dominance relationship. Are a subset of ConcreteNode plus ENTRY and EXIT."""

    name: str
    value: uuid.UUID

    @staticmethod
    def new() -> "VirtualNode":
        return VirtualNode("None", uuid.uuid4())

    def to_concrete(self) -> ConcreteNode:
        assert not self.is_entry() and not self.is_exit(), "Cannot convert ENTRY/EXIT to ConcreteNode"
        return ConcreteNode(self.value, self.name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)

    def is_entry(self) -> bool:
        return self.name == "VIRTUAL_ENTRY"

    def is_exit(self) -> bool:
        return self.name == "VIRTUAL_EXIT"


class DominanceInformation:
    def __init__(self, tg: TorchGraph):
        self.torch_graph = tg
        self.entry_node: VirtualNode = VirtualNode("VIRTUAL_ENTRY", uuid.uuid4())
        self.exit_node: VirtualNode = VirtualNode("VIRTUAL_EXIT", uuid.uuid4())

        logger.info("computing dominance information for graph with %d nodes", len(tg.execution_order))

        d_successors = self.build_successors(tg, self.entry_node, self.exit_node)
        d_successors = prune_unreachable(d_successors, self.entry_node, self.exit_node)
        d_predecessors = self.build_predecessors(d_successors)

        d_all_nodes = set(d_successors.keys()).union(*d_successors.values())

        d_dom = self.compute_dominance_sets(self.entry_node, d_all_nodes, d_predecessors)
        d_pdom = self.compute_dominance_sets(self.exit_node, d_all_nodes, d_successors)
        d_dom_tree = self.build_dominance_tree(d_dom)
        d_pdom_tree = self.build_dominance_tree(d_pdom)
        d_reverse_dom_tree = self.reverse_tree(d_dom_tree)
        d_reverse_pdom_tree = self.reverse_tree(d_pdom_tree)

        self.all_nodes = frozenset(d_all_nodes)
        self.successors = frozendict({n: frozenset(succs) for n, succs in d_successors.items()})
        self.predecessors = frozendict({n: frozenset(preds) for n, preds in d_predecessors.items()})
        self.dom = frozendict({n: frozenset(ds) for n, ds in d_dom.items()})
        self.pdom = frozendict({n: frozenset(ds) for n, ds in d_pdom.items()})
        self.dom_tree = frozendict({n: frozenset(cs) for n, cs in d_dom_tree.items()})
        self.pdom_tree = frozendict({n: frozenset(cs) for n, cs in d_pdom_tree.items()})
        self.reverse_dom_tree = frozendict({n: frozenset(cs) for n, cs in d_reverse_dom_tree.items()})
        self.reverse_pdom_tree = frozendict({n: frozenset(cs) for n, cs in d_reverse_pdom_tree.items()})

    def from_concrete(self, c_node: ConcreteNode) -> VirtualNode:
        """Convert a ConcreteNode to a VirtualNode."""
        v_node = DominanceInformation._unsafe_from_concrete(c_node)
        if v_node not in self.all_nodes:
            raise ValueError(f"ConcreteNode {c_node} not in dominance graph")
        return v_node

    def has_virtual_repr(self, c_node: ConcreteNode) -> bool:
        """Returns true if the concrete node has a representation in this dominance graph."""
        return DominanceInformation._unsafe_from_concrete(c_node) in self.all_nodes

    @staticmethod
    def _unsafe_from_concrete(c_node: ConcreteNode) -> VirtualNode:
        """Convert a ConcreteNode to a VirtualNode. FOR INTERNAL USE ONLY"""
        return VirtualNode(c_node.name, c_node.uuid)

    def ordered_nodes(self):
        """Return nodes in TorchGraph execution order, with ENTRY first and EXIT last."""
        yield self.entry_node
        for c in self.torch_graph.execution_order:
            v = DominanceInformation._unsafe_from_concrete(c)
            if v in self.all_nodes:
                yield v
        yield self.exit_node

    def get_successors(self, node: VirtualNode) -> frozenset[VirtualNode]:
        """Return the set of successor nodes of `node`."""
        return self.successors.get(node, frozenset())

    def get_predecessors(self, node: VirtualNode) -> frozenset[VirtualNode]:
        """Return the set of predecessor nodes of `node`."""
        return self.predecessors.get(node, frozenset())

    def dom_root(self) -> VirtualNode:
        """Return the entry node of the dominance graph."""
        return self.entry_node

    def pdom_root(self) -> VirtualNode:
        """Return the exit node of the post-dominance graph."""
        return self.exit_node

    def dominators(self, node: VirtualNode) -> frozenset[VirtualNode]:
        """Return the set of nodes that dominate `node`."""
        return self.dom.get(node, frozenset())

    def post_dominators(self, node: VirtualNode) -> frozenset[VirtualNode]:
        """Return the set of nodes that post-dominate `node`."""
        return self.pdom.get(node, frozenset())

    def dom_tree_children(self, node: VirtualNode) -> frozenset[VirtualNode]:
        """Return children that this node **immediately dominates**"""
        return self.dom_tree.get(node, frozenset())

    def nodes(self) -> frozenset[VirtualNode]:
        """Return the set of all nodes in the augmented graph."""
        return frozenset(self.all_nodes)

    def is_entry(self, node: VirtualNode) -> bool:
        """Check if the node is the entry node."""
        return node == self.entry_node

    def is_exit(self, node: VirtualNode) -> bool:
        """Check if the node is the exit node."""
        return node == self.exit_node

    @staticmethod
    def build_successors(tg: TorchGraph, entry: VirtualNode, exit: VirtualNode) -> dict[VirtualNode, set[VirtualNode]]:
        core = {
            DominanceInformation._unsafe_from_concrete(c): set(
                map(DominanceInformation._unsafe_from_concrete, tg.get_successors(c))
            )
            for c in tg.execution_order
        }
        outputs = {DominanceInformation._unsafe_from_concrete(c): {exit} for c in tg.outputs}
        successors = {
            entry: set(map(DominanceInformation._unsafe_from_concrete, tg.inputs)),
            **core,
            **outputs,
        }

        return successors

    @staticmethod
    def build_predecessors(
        successors: dict[VirtualNode, set[VirtualNode]],
    ) -> dict[VirtualNode, set[VirtualNode]]:
        """Inverse of successor adjacency."""
        return DominanceInformation.reverse_tree(successors)

    @staticmethod
    def reverse_tree(
        tree: dict[VirtualNode, set[VirtualNode]],
    ) -> dict[VirtualNode, set[VirtualNode]]:
        """
        Reverse adjacency (u -> v) becomes (v -> u).
        Ensures nodes that only appear as targets are present as keys.
        """
        if not tree:
            return {}
        all_nodes = set(tree.keys()).union(*tree.values())
        rev: dict[VirtualNode, set[VirtualNode]] = {n: set() for n in all_nodes}
        for u, succs in tree.items():
            for v in succs:
                rev[v].add(u)
            # Ensure isolated nodes remain present
            rev.setdefault(u, rev.get(u, set()))
        return rev

    @staticmethod
    def compute_dominance_sets(
        root: VirtualNode,
        all_nodes: set[VirtualNode],
        rel: dict[VirtualNode, set[VirtualNode]],
    ) -> dict[VirtualNode, set[VirtualNode]]:
        """
        Iterative dataflow: Dom(n) = {n} ∪ ⋂_{p ∈ rel(n)} Dom(p), with Dom(root) = {root}.
        `rel` is usually predecessors (for dominance) or successors (for post-dominance).
        """
        if not all_nodes:
            return {}

        # Initialize dominance sets: root dominates only itself, others start with all nodes
        dom = {n: ({n} if (n == root or not rel.get(n)) else set(all_nodes)) for n in all_nodes}

        changed = True
        iteration = 0
        max_iterations = len(all_nodes) * 2  # conservative cap

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for n in all_nodes:
                if n == root:
                    continue

                prev = dom[n]
                # Compute intersection of dominators over all predecessors/successors
                preds = rel.get(n, set())
                inter = set(all_nodes) if preds else set()
                for p in preds:
                    inter &= dom[p] if inter else dom[p].copy()

                new = inter | {n}
                if new != prev:
                    dom[n] = new
                    changed = True

        total = sum(len(s) for s in dom.values())
        avg = total / len(dom) if dom else 0.0
        logger.debug("[dim]dominance: avg %.1f dominators per node[/]", avg)

        if iteration >= max_iterations:
            logger.warning("[red]dominance computation hit iteration limit (%d)[/]", max_iterations)

        return dom

    @staticmethod
    def build_dominance_tree(
        dsets: dict[VirtualNode, set[VirtualNode]],
    ) -> dict[VirtualNode, set[VirtualNode]]:
        """
        Immediate dominator tree from full dominance sets:
        idom(n) = argmax_{d ∈ Dom(n) \\ {n}} |Dom(d)|   (break ties arbitrarily).
        """
        if not dsets:
            return {}

        strict = {n: (ds - {n}) for n, ds in dsets.items()}
        # Select the strict dominator with the largest dom-set (immediate dominator)
        idom = {
            n: max(D, key=lambda d: len(dsets[d]))  # type: ignore[arg-type]
            for n, D in strict.items()
            if D
        }

        tree: dict[VirtualNode, set[VirtualNode]] = {}
        for n, parent in idom.items():
            tree.setdefault(parent, set()).add(n)
        # Ensure all nodes appear as keys in the tree
        for n in dsets.keys():
            tree.setdefault(n, tree.get(n, set()))
        return tree

    def _render_graph(
        self,
        tree: dict[VirtualNode, set[VirtualNode]] | frozendict[VirtualNode, frozenset[VirtualNode]],
        graph: graphviz.Digraph,
        flatten: bool = False,
    ) -> None:
        """
        Render a (post-)dominance tree into a graphviz.Digraph.
        If `flatten` is True, lay edges left-to-right and disambiguate node names with a local suffix.
        """

        def label(n: VirtualNode) -> str:
            return n.name

        def add(n: VirtualNode, suffix: str = "") -> str:
            name = f"{n.value}{suffix}"
            graph.node(name, label=label(n), shape="box")
            return name

        if flatten:
            graph.attr(rankdir="LR")
            for i, (u, vs) in enumerate(tree.items()):
                u_name = add(u, str(i))
                for v in vs:
                    graph.edge(u_name, add(v, str(i)))
        else:
            for u, vs in tree.items():
                u_name = add(u)
                for v in vs:
                    graph.edge(u_name, add(v))

    def visualize_dominance(self, export_path: Path):
        """exports four graphs: dominance tree, post-dominance tree, reverse dominance tree, reverse post-dominance tree"""

        export_path.absolute().mkdir(parents=True, exist_ok=True)

        dom_gv = graphviz.Digraph(name="Dominance Set")
        dom_gv.attr(nodesep="0.1", ranksep="0.3")
        self._render_graph(self.dom, dom_gv, flatten=True)
        dom_gv.render(export_path / "dominance_set", format="pdf")

        dom_tree_gv = graphviz.Digraph(name="Dominance Tree")
        dom_tree_gv.attr(nodesep="0.1", ranksep="0.3")
        self._render_graph(self.dom_tree, dom_tree_gv)
        dom_tree_gv.render(export_path / "dominance_tree", format="pdf")

        pdom_tree_gv = graphviz.Digraph(name="Post Dominance Tree")
        pdom_tree_gv.attr(nodesep="0.1", ranksep="0.3")
        self._render_graph(self.pdom_tree, pdom_tree_gv)
        pdom_tree_gv.render(export_path / "post_dominance_tree", format="pdf")

        rev_dom_tree_gv = graphviz.Digraph(name="Reverse Dominance Tree")
        rev_dom_tree_gv.attr(nodesep="0.1", ranksep="0.3")
        self._render_graph(self.reverse_dom_tree, rev_dom_tree_gv)
        rev_dom_tree_gv.render(export_path / "rev_dominance_tree", format="pdf")

        rev_pdom_tree_gv = graphviz.Digraph(name="Reverse Post Dominance Tree")
        rev_pdom_tree_gv.attr(nodesep="0.1", ranksep="0.3")
        self._render_graph(self.reverse_pdom_tree, rev_pdom_tree_gv)
        rev_pdom_tree_gv.render(export_path / "rev_post_dominance_tree", format="pdf")


def prune_unreachable(
    successors: dict[VirtualNode, set[VirtualNode]],
    entry: VirtualNode,
    exit: VirtualNode,
) -> dict[VirtualNode, set[VirtualNode]]:
    """Remove nodes not reachable from ENTRY or that cannot reach EXIT."""

    # Forward reachability: collect all nodes reachable from entry
    reachable_from_entry = set()
    queue = deque([entry])

    while queue:
        node = queue.popleft()
        if node not in reachable_from_entry:
            reachable_from_entry.add(node)
            queue.extend(successors.get(node, []))

    # Reverse reachability: collect all nodes that can reach exit
    reverse: dict[VirtualNode, set[VirtualNode]] = {n: set() for n in successors}
    for u, outs in successors.items():
        for v in outs:
            reverse.setdefault(v, set()).add(u)

    reachable_to_exit = set()
    queue = deque([exit])

    while queue:
        node = queue.popleft()
        if node not in reachable_to_exit:
            reachable_to_exit.add(node)
            queue.extend(reverse.get(node, []))

    # Keep only nodes valid in both directions
    valid_nodes = reachable_from_entry & reachable_to_exit

    # Rebuild adjacency with only valid nodes
    pruned = {u: {v for v in outs if v in valid_nodes} for u, outs in successors.items() if u in valid_nodes}

    return pruned
