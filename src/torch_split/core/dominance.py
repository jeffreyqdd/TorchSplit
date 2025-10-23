import uuid
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, Set

import graphviz  # type: ignore
from frozendict import frozendict

from torch_split.core.ir import TorchGraph
from torch_split.logging import get_logger

logger = get_logger(__name__)

# DominanceUuid = uuid.UUID


@dataclass(frozen=True)
class VirtualNodeId(uuid.UUID):
    value: uuid.UUID

    @staticmethod
    def new() -> "VirtualNodeId":
        return VirtualNodeId(uuid.uuid4())


class DominanceInformation:
    def __init__(self, tg: TorchGraph):
        self.torch_graph = tg
        self.entry_node: VirtualNodeId = VirtualNodeId(uuid.uuid4())
        self.exit_node: VirtualNodeId = VirtualNodeId(uuid.uuid4())

        # Core relationships
        successors = self.build_successors(tg, self.entry_node, self.exit_node)
        successors = prune_unreachable(successors, self.entry_node, self.exit_node)
        predecessors = self.build_predecessors(successors)

        # All nodes = union(keys ∪ values) — guarantees inclusion of entry/exit/leafs
        all_nodes = set(successors.keys()).union(*successors.values())

        dom = self.compute_dominance_sets(self.entry_node, all_nodes, predecessors)
        pdom = self.compute_dominance_sets(self.exit_node, all_nodes, successors)
        dom_tree = self.build_dominance_tree(dom)
        pdom_tree = self.build_dominance_tree(pdom)
        reverse_dom_tree = self.reverse_tree(dom_tree)
        reverse_pdom_tree = self.reverse_tree(pdom_tree)

        self.all_nodes = frozenset(all_nodes)
        self.successors = frozendict({n: frozenset(succs) for n, succs in successors.items()})
        self.predecessors = frozendict({n: frozenset(preds) for n, preds in predecessors.items()})
        self.dom = frozendict({n: frozenset(ds) for n, ds in dom.items()})
        self.pdom = frozendict({n: frozenset(ds) for n, ds in pdom.items()})
        self.dom_tree = frozendict({n: frozenset(cs) for n, cs in dom_tree.items()})
        self.pdom_tree = frozendict({n: frozenset(cs) for n, cs in pdom_tree.items()})
        self.reverse_dom_tree = frozendict({n: frozenset(cs) for n, cs in reverse_dom_tree.items()})
        self.reverse_pdom_tree = frozendict({n: frozenset(cs) for n, cs in reverse_pdom_tree.items()})

    # ---------- Public API ----------

    def dominators(self, node: VirtualNodeId) -> frozenset[VirtualNodeId]:
        """Return the set of nodes that dominate `node`."""
        return self.dom.get(node, frozenset())

    def post_dominators(self, node: VirtualNodeId) -> frozenset[VirtualNodeId]:
        """Return the set of nodes that post-dominate `node`."""
        return self.pdom.get(node, frozenset())

    def nodes(self) -> frozenset[VirtualNodeId]:
        """Return the set of all nodes in the augmented graph."""
        return frozenset(self.all_nodes)

    def successors_of(self, node: VirtualNodeId) -> frozenset[VirtualNodeId]:
        """Return the set of successor nodes of `node`."""
        return self.successors.get(node, frozenset())

    def predecessors_of(self, node: VirtualNodeId) -> frozenset[VirtualNodeId]:
        """Return the set of predecessor nodes of `node`."""
        return self.predecessors.get(node, frozenset())

    def name_of(self, node: VirtualNodeId) -> str:
        """Return the name of the node from the underlying TorchGraph, or special names for entry/exit."""
        if node == self.entry_node:
            return "ENTRY"
        if node == self.exit_node:
            return "EXIT"
        return TorchGraph.name_from_node(self.torch_graph.nodes[node.value])

    def safe_unwrap(self, node: VirtualNodeId) -> uuid.UUID:
        """Unwrap a DominanceUuid to a uuid.UUID; errors on entry/exit nodes."""
        if node == self.entry_node or node == self.exit_node:
            raise ValueError("Cannot unwrap entry/exit nodes to TorchGraph Uuids")
        return node.value

    def safe_unwrap_iterable(self, nodes: Iterable[VirtualNodeId]) -> Iterable[uuid.UUID]:
        """Unwrap a set of DominanceUuids to a set of uuid.UUIDs; errors on entry/exit nodes."""
        return (self.safe_unwrap(n) for n in nodes)

    def is_entry(self, node: VirtualNodeId) -> bool:
        """Check if the node is the entry node."""
        return node == self.entry_node

    def is_exit(self, node: VirtualNodeId) -> bool:
        """Check if the node is the exit node."""
        return node == self.exit_node

    # ---------- Graph construction ----------

    @staticmethod
    def build_successors(
        tg: TorchGraph, entry: VirtualNodeId, exit: VirtualNodeId
    ) -> Dict[VirtualNodeId, Set[VirtualNodeId]]:
        core = {VirtualNodeId(uid): set(map(VirtualNodeId, tg.node_dataflow.get(uid, ()))) for uid in tg.nodes}

        successors = {
            entry: set(map(VirtualNodeId, tg.inputs)),
            **core,
            VirtualNodeId(tg.return_node): {exit},
        }

        return successors

    @staticmethod
    def build_predecessors(
        successors: Dict[VirtualNodeId, Set[VirtualNodeId]],
    ) -> Dict[VirtualNodeId, Set[VirtualNodeId]]:
        """Inverse of successor adjacency."""
        return DominanceInformation.reverse_tree(successors)

    @staticmethod
    def reverse_tree(tree: Dict[VirtualNodeId, Set[VirtualNodeId]]) -> Dict[VirtualNodeId, Set[VirtualNodeId]]:
        """
        Reverse adjacency (u -> v) becomes (v -> u).
        Ensures nodes that only appear as targets are present as keys.
        """
        if not tree:
            return {}
        all_nodes = set(tree.keys()).union(*tree.values())
        rev: Dict[VirtualNodeId, Set[VirtualNodeId]] = {n: set() for n in all_nodes}
        for u, succs in tree.items():
            for v in succs:
                rev[v].add(u)
            # Ensure isolated nodes remain present
            rev.setdefault(u, rev.get(u, set()))
        return rev

    # ---------- Dominance ----------

    @staticmethod
    def compute_dominance_sets(
        root: VirtualNodeId,
        all_nodes: Set[VirtualNodeId],
        rel: Dict[VirtualNodeId, Set[VirtualNodeId]],
    ) -> Dict[VirtualNodeId, Set[VirtualNodeId]]:
        """
        Iterative dataflow: Dom(n) = {n} ∪ ⋂_{p ∈ rel(n)} Dom(p), with Dom(root) = {root}.
        `rel` is usually predecessors (for dominance) or successors (for post-dominance).
        """
        if not all_nodes:
            return {}

        # Initialize: root or nodes with no predecessors/successors start at {n}; others start at all_nodes.
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
                # Intersection over predecessors/successors; empty family => empty set
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
        else:
            logger.info("[bold blue]dominance computation complete[/] [dim](%d iterations)[/]", iteration)

        return dom

    @staticmethod
    def build_dominance_tree(dsets: Dict[VirtualNodeId, Set[VirtualNodeId]]) -> Dict[VirtualNodeId, Set[VirtualNodeId]]:
        """
        Immediate dominator tree from full dominance sets:
        idom(n) = argmax_{d ∈ Dom(n) \ {n}} |Dom(d)|   (break ties arbitrarily).
        """
        if not dsets:
            return {}

        strict = {n: (ds - {n}) for n, ds in dsets.items()}
        # Choose the strict dominator with the largest dom-set (closest ancestor)
        idom = {
            n: max(D, key=lambda d: len(dsets[d]))  # type: ignore[arg-type]
            for n, D in strict.items()
            if D
        }

        tree: Dict[VirtualNodeId, Set[VirtualNodeId]] = {}
        for n, parent in idom.items():
            tree.setdefault(parent, set()).add(n)
        # ensure isolated nodes appear as keys (optional; keeps rendering simpler)
        for n in dsets.keys():
            tree.setdefault(n, tree.get(n, set()))
        return tree

    # ---------- Rendering ----------

    def render_graph(
        self,
        tree: Dict[VirtualNodeId, Set[VirtualNodeId]] | frozendict[VirtualNodeId, frozenset[VirtualNodeId]],
        graph: graphviz.Digraph,
        flatten: bool = False,
    ) -> None:
        """
        Render a (post-)dominance tree into a graphviz.Digraph.
        If `flatten` is True, lay edges left-to-right and disambiguate node names with a local suffix.
        """

        def label(n: VirtualNodeId) -> str:
            if n == self.entry_node:
                return "ENTRY"
            if n == self.exit_node:
                return "EXIT"
            return TorchGraph.name_from_node(self.torch_graph.nodes[n.value])

        def add(n: VirtualNodeId, suffix: str = "") -> str:
            name = f"{n.value}{suffix}"
            graph.node(name, label=label(n))
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


def prune_unreachable(
    successors: Dict[VirtualNodeId, Set[VirtualNodeId]], entry: VirtualNodeId, exit: VirtualNodeId
) -> Dict[VirtualNodeId, Set[VirtualNodeId]]:
    """
    Remove any nodes not reachable from ENTRY (forward) or
    that cannot reach EXIT (reverse).
    """

    # --------- 1️⃣ Forward Reachability: From ENTRY ----------
    reachable_from_entry = set()
    queue = deque([entry])

    while queue:
        node = queue.popleft()
        if node not in reachable_from_entry:
            reachable_from_entry.add(node)
            queue.extend(successors.get(node, []))

    # --------- 2️⃣ Reverse Reachability: To EXIT ----------
    # Build reverse edges
    reverse: dict[VirtualNodeId, set[VirtualNodeId]] = {n: set() for n in successors}
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

    # --------- 3️⃣ Intersection: Nodes valid in both directions ----------
    valid_nodes = reachable_from_entry & reachable_to_exit

    # --------- 4️⃣ Rebuild pruned adjacency ----------
    pruned = {u: {v for v in outs if v in valid_nodes} for u, outs in successors.items() if u in valid_nodes}

    return pruned
