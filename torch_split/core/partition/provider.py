import itertools
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeVar

import graphviz  # type: ignore
import torch.fx as fx

import torch_split.core.assertions as assertions
import torch_split.log as logging
from torch_split.core.switchboard import (
    Switchboard,
    ComponentMetadata,
    ComponentName,
    Entrypoint,
    DownstreamNode,
    Layout,
)
import torch_split.core.utils as utils
from torch_split.core.ir import ConcreteNode, TorchGraph
from torch_split.core.partition.dominance import DominanceInformation, VirtualNode

logger = logging.get_logger(__name__)

T = TypeVar("T", bound=VirtualNode | ConcreteNode)
VirtualNodeSet = frozenset[VirtualNode]
ConcreteNodeSet = frozenset[ConcreteNode]


@dataclass(frozen=True)
class Subgraph:
    inputs: tuple[ConcreteNode, ...]
    """Subset of a cut's split set. Non-paremeter nodes. Should be input into the enclosed region"""

    outputs: tuple[ConcreteNode, ...]
    """Subset of a cut's join set. Non-parameter nodes. Should be output from the enclosed region (out-degree of 0)"""

    enclosed_region: ConcreteNodeSet
    """all node ids in this subcomponent including inputs and outputs"""


@dataclass(frozen=True)
class Cut:
    """A preliminary cut candidate defined by source and sink nodes. May not be entry or exit nodes since they cannot be inputs into torch operations."""

    split: VirtualNodeSet
    """data in this set are broadcasted to all subgraphs"""

    join: VirtualNodeSet
    """data from the subgraphs are aggregated here"""


@dataclass(frozen=True)
class Partition:
    """Contains a cut that segmenets a SESE and the disjoint subgraphs formed by the cut"""

    cut: Cut
    """the cut that formed this partition"""

    subgraphs: frozenset[Subgraph]
    """the subgraphs formed by this cut"""


class PartitionProvider:
    """Provides partitioning of a TorchGraph based on dominance relationships."""

    def __init__(self, torch_graph: TorchGraph) -> None:
        self.torch_graph = torch_graph
        self.dominance = DominanceInformation(torch_graph)

    def visualize_dataflow(self, export_path: Path, include_parameters: bool = False):
        """export the dataflow graph as a pdf"""
        export_path.absolute().mkdir(parents=True, exist_ok=True)
        dot = graphviz.Digraph(name="Dataflow Graph")
        dot.attr(nodesep="0.1", ranksep="0.3")
        self.torch_graph.render_graph(dot, include_parameters=include_parameters)
        dot.render(export_path / "dataflow_graph", format="pdf")

        self.dominance.visualize_dominance(export_path)

    def all_partitions(self) -> Iterable[Partition]:
        """Return a generator of all valid partitions in the graph."""

        # purpose of this section is to filter cut candidates down to cuts that yield immediate subgraphs.
        all_valid_cuts: set[Cut] = set()
        for cut in self._get_sese_region():
            assert len(cut.split) == 1, "only one item allowed"
            assert len(cut.join) == 1, "only one item allowed"

            split_node = set(cut.split).pop()
            children = self.dominance.dom_tree_children(set(cut.split).pop())

            visited_count: dict[VirtualNode, int] = defaultdict(lambda: 0)
            for child in children:
                # start from an immeditely dominated node, traverse updwards (predecessors) until we hit the cut.join
                # or another immediately dominated node
                source = frozenset([child])
                next = self.dominance.get_predecessors
                for node in self._flood_fill_helper(source, (cut.split | children) - source, next):
                    visited_count[node] += 1

            # after we remove "children" from interacted nodes count, we are left with refined "cut.split" nodes
            refined_split: VirtualNodeSet = frozenset(
                filter(
                    lambda n: n not in children and visited_count[n] >= max(len(children) - 1, 2),
                    visited_count.keys(),
                )
            )

            if len(refined_split) == 0:
                continue

            # similary, bubble up the split nodes so that the split + join form a minimal cut.
            join_node = set(cut.join).pop()
            while len(parent := self.dominance.get_predecessors(join_node)) == 1:
                parent_node = set(parent).pop()
                if parent_node in self.dominance.post_dominators(split_node):
                    join_node = parent_node

            # check that split_node and join_node actually have nodes in between
            # we subtract .split and .join to remove the cut nodes themselves
            candidate_cut = Cut(split=frozenset([split_node]), join=frozenset([join_node]))
            enclosed_region = (self._get_core_region(candidate_cut) - candidate_cut.split) - candidate_cut.join

            if len(enclosed_region) == 0:
                continue

            # we will skip this cut if the the split and join nodes are spearated by a single edge
            if join_node in self.dominance.get_successors(split_node):
                continue

            if split_node in self.dominance.get_predecessors(split_node):
                continue

            all_valid_cuts.add(candidate_cut)

        # purpose of this section is to create partion (find disjoint subgraphs) from each valid cut
        for cut in all_valid_cuts:
            # we expand split node to avoid virtual ENTRY node this is why we ensure that split and join are
            # not directly conncted above. **because we've expanded the entry node, there is no need to
            # subtract it from the final enclosed region !!!!

            concrete_split: set[ConcreteNode] = set()
            for succs in (self.dominance.get_successors(n) for n in cut.split):
                assertions.disjoint(succs, cut.join)
                concrete_split.update(map(VirtualNode.to_concrete, (i for i in succs)))

            concrete_join: ConcreteNodeSet = ConcreteNodeSet(map(VirtualNode.to_concrete, cut.join))

            # remove the join nodes so that our subgraph code actually will return disjoint subgraphs
            # print("[debug]: cut region: ", [n.name for n in cut.split], "→", [n.name for n in cut.join])
            # print("[debug]:     concrete split: ", [n.name for n in concrete_split])
            # print("[debug]:     concrete join: ", [n.name for n in concrete_join])
            # REMINDER: There is no need to substract concrete_split
            concrete_region = self._get_enclosed_region(frozenset(concrete_split), concrete_join) - concrete_join
            # print("[debug]:     er: ", concrete_region)
            subgraphs: list[ConcreteNodeSet] = list(self._get_subgraphs(frozenset(concrete_region), self._adjacent))

            if len(subgraphs) <= 1:
                continue

            yield Partition(
                cut=cut,
                subgraphs=frozenset(
                    Subgraph(
                        # remove concrete_join (inputs) if it is a virtual entry node
                        # (special case)
                        inputs=self._subgraph_inputs(sg),
                        outputs=self._subgraph_outputs(sg),
                        enclosed_region=sg,
                    )
                    for sg in subgraphs
                ),
            )

    def create_switchboard(
        self, selected_partitions: Iterable[Partition], roots: Optional["PartitionProvider"] = None
    ) -> Switchboard:
        if roots is not None:
            curr_tg = self.torch_graph
            root_tg = roots.torch_graph

            if len(curr_tg.execution_order) != len(root_tg.execution_order):
                raise ValueError("Number of nodes is not the same between current and root TorchGraph")

            for curr_node, root_node in zip(curr_tg.execution_order, root_tg.execution_order):
                if curr_node.name != root_node.name:
                    raise ValueError("Execution order does not match between current and root TorchGraph")

            curr_ordered_virtual = list(self.dominance.ordered_nodes())
            root_ordered_virtual = list(roots.dominance.ordered_nodes())

            if len(curr_ordered_virtual) != len(root_ordered_virtual):
                raise ValueError("Number of virtual nodes is not the same between current and root TorchGraph")
            for curr_node, root_node in zip(curr_ordered_virtual, root_ordered_virtual):
                if curr_node.name != root_node.name:
                    raise ValueError("Virtual node order does not match between current and root TorchGraph")

            uid_map: dict[ConcreteNode, ConcreteNode] = {
                root_node: curr_node for curr_node, root_node in zip(curr_tg.execution_order, root_tg.execution_order)
            }

            vid_map: dict[VirtualNode, VirtualNode] = {
                root_node: curr_node
                for curr_node, root_node in zip(self.dominance.ordered_nodes(), roots.dominance.ordered_nodes())
            }

            def map_from_root(subgraph: Subgraph) -> Subgraph:
                return Subgraph(
                    inputs=tuple(uid_map[n] for n in subgraph.inputs),
                    outputs=tuple(uid_map[n] for n in subgraph.outputs),
                    enclosed_region=frozenset(uid_map[n] for n in subgraph.enclosed_region),
                )

            def map_from_root_virtual(cut: Cut) -> Cut:
                return Cut(
                    split=frozenset(vid_map[n] for n in cut.split),
                    join=frozenset(vid_map[n] for n in cut.join),
                )

            selected_partitions = map(
                lambda p: Partition(
                    cut=map_from_root_virtual(p.cut),
                    subgraphs=frozenset(map(map_from_root, p.subgraphs)),
                ),
                selected_partitions,
            )

        # the sort makes the export deterministic
        all_subgraphs = list(self._carve_subgraphs(selected_partitions))
        all_subgraphs = sorted(all_subgraphs, key=lambda a: -len(a.enclosed_region))
        idx2char = [str(chr(ord("A") + idx)) for idx, _ in enumerate(all_subgraphs)]

        to_fx = self.torch_graph.to_fx
        mod = self.torch_graph.graph_module
        sort = self.torch_graph.sort_execution_order

        components: dict[ComponentName, fx.GraphModule] = {
            idx2char[idx]: utils.extract_subgraph(
                mod,
                list(map(to_fx, sort(subgraph.enclosed_region))),
                list(map(to_fx, subgraph.inputs)),
                list(map(to_fx, subgraph.outputs)),
            )
            for idx, subgraph in enumerate(all_subgraphs)
        }

        # for x in components.values():
        #     print("preexport code")
        #     print(x.code)

        metadata: dict[ComponentName, ComponentMetadata] = {
            idx2char[idx]: ComponentMetadata(
                name=idx2char[idx],
                version_hash=utils.hash_model_architecture(components[idx2char[idx]]),
                input_parameters=tuple(
                    node.name for node in components[idx2char[idx]].graph.nodes if node.op == "placeholder"
                ),
                output_parameters=tuple(map(lambda n: n.name, subgraph.outputs)),
            )
            for idx, subgraph in enumerate(all_subgraphs)
        }

        entrypoints: list[Entrypoint] = [
            Entrypoint(name=idx2char[idx])
            for idx, producer in enumerate(all_subgraphs)
            if not any(
                len(set(producer.inputs) & set(candidate.outputs)) > 0
                for candidate in all_subgraphs
                if candidate != producer
            )
        ]

        dfg = {
            idx2char[pidx]: [
                DownstreamNode(name=idx2char[cidx], mapping=[(c.name, c.name) for c in common])
                for cidx, consumer in enumerate(all_subgraphs)
                if len(common := set(consumer.inputs) & set(producer.outputs)) > 0
            ]
            for pidx, producer in enumerate(all_subgraphs)
        }

        return Switchboard(layout=Layout(metadata=metadata, entrypoints=entrypoints, dfg=dfg), components=components)

    def _carve_subgraphs(self, selected_partitions: Iterable[Partition]) -> Iterable[Subgraph]:
        remaining_nodes = set(self.torch_graph.execution_order)
        for partition in selected_partitions:
            for subgraph in partition.subgraphs:
                assertions.subset(subgraph.enclosed_region, remaining_nodes)
                remaining_nodes -= subgraph.enclosed_region
                yield subgraph

        for sg in self._get_subgraphs(frozenset(remaining_nodes), self._adjacent):
            yield Subgraph(
                inputs=self._subgraph_inputs(sg),
                outputs=self._subgraph_outputs(sg),
                enclosed_region=sg,
            )

    # [debug]: cut region:  ['hidden_states_51'] → ['hidden_states_63']
    def _adjacent(self, node: ConcreteNode) -> frozenset[ConcreteNode]:
        # if node.name == "mask":
        #     a = self.torch_graph.get_successors(node)
        #     b = self.torch_graph.get_predecessors(node)
        #     print("[debug]: getting adjacent for mask", a)
        #     print("[debug]: getting adjacent for mask", b)
        #     print("[debug]: getting adjacent for mask", a.union(b))
        return self.torch_graph.get_successors(node).union(self.torch_graph.get_predecessors(node))

    def _get_subgraphs(self, node_set: frozenset[T], adjacent: Callable[[T], frozenset[T]]) -> Iterable[frozenset[T]]:
        """Return all disjoint subgraphs in the given node set. Node set must be a subset of the range of adjacent().

        Note: all return values from adjacent() are filtered to be within node_set. adjacent() must return all the node
        ids that are directly reachable from the given node in the **original** graph.
        """
        visited = set()
        for node in node_set:
            if node not in visited:
                # print("[debug]: considering node ", node)
                component: set[T] = set()
                worklist = [node]
                visited.add(node)

                while worklist:
                    current = worklist.pop()
                    # print("[debug]:   visiting node ", current)
                    component.add(current)
                    for successor in adjacent(current):
                        if successor in node_set and successor not in visited:
                            visited.add(successor)
                            worklist.append(successor)
                yield frozenset(component)

    def _get_sese_region(self) -> frozenset[Cut]:
        """Return all single code cuts (A, B) where A dominates B and B post-dominates A."""

        def lazy_iterable():
            for node_a, node_b in itertools.combinations(self.dominance.ordered_nodes(), 2):
                is_d = node_a in self.dominance.dominators(node_b)
                is_p = node_b in self.dominance.post_dominators(node_a)
                if is_d and is_p:
                    yield Cut(frozenset({node_a}), frozenset({node_b}))

        return frozenset(lazy_iterable())

    def _get_enclosed_region(self, source: ConcreteNodeSet, sink: ConcreteNodeSet) -> ConcreteNodeSet:
        """Get the region between source and sink nodes, **including** parameter nodes. Parameter nodes
        are constexpr-foldable. source node and sink node are included in the region."""

        def next(n: ConcreteNode) -> frozenset[ConcreteNode]:
            if n in source:
                return self.torch_graph.get_successors(n)
            elif n in sink:
                return self.torch_graph.get_predecessors(n)
            else:
                return self.torch_graph.get_successors(n).union(self.torch_graph.get_predecessors(n))

        return self._flood_fill_helper(source, sink, next)

    def _get_core_region(self, cut: Cut) -> VirtualNodeSet:
        """Get the region between source and sink nodes, not including parameter nodes. Parameter nodes
        are constexpr-foldable. source node and sink node are included in the region."""
        return self._flood_fill_helper(cut.join, cut.split, self.dominance.get_successors)

    def _flood_fill_helper(
        self,
        source: frozenset[T],
        sink: frozenset[T],
        next: Callable[[T], frozenset[T]],
    ) -> frozenset[T]:
        """Flood fill from source to sink using the provided `next` function to get adjacent nodes. Source and Sink cannot overlap."""
        assertions.disjoint(source, sink)

        frontier_set = set(source)
        settled_set = set(sink)

        while frontier_set:
            if (n := frontier_set.pop()) not in settled_set:
                settled_set.add(n)
                for succ in next(n):
                    if succ not in settled_set and succ not in frontier_set:
                        frontier_set.add(succ)

        assertions.subset(source, settled_set)
        assertions.subset(sink, settled_set)
        return frozenset(settled_set)

    def _subgraph_inputs(self, subgraph_nodes: ConcreteNodeSet) -> tuple[ConcreteNode, ...]:
        """Nodes whose predecessors include at least one outside the subgraph."""
        ret = set()

        for n in subgraph_nodes:
            if not self.dominance.has_virtual_repr(n):
                continue

            virtual_n = self.dominance.from_concrete(n)
            if set(self.dominance.get_predecessors(virtual_n)).pop().is_entry():
                # add if this node's parent is virtual entry (speical case)
                ret.add(n)
                continue

            for pred in self.torch_graph.get_predecessors(n) - subgraph_nodes:
                if self.dominance.has_virtual_repr(pred):
                    # must not be a parameter node
                    ret.add(pred)

        # we should return in a deterministic order
        return tuple(sorted(ret, key=lambda x: x.name))

    def _subgraph_outputs(self, subgraph_nodes: ConcreteNodeSet) -> tuple[ConcreteNode, ...]:
        """Nodes with no successors in the subgraph."""
        return tuple(
            sorted(
                [n for n in subgraph_nodes if not (self.torch_graph.get_successors(n) & subgraph_nodes)],
                key=lambda x: x.name,
            )
        )
