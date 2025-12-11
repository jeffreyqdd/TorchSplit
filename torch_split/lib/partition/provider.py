import itertools
import pickle
import json
import uuid
import base64
import io
from collections import defaultdict
from collections.abc import Callable, Iterable, Set
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
import lzma
from typing import Any, Type, TypeVar

import graphviz  # type: ignore
import torch
import torch.fx as fx
from frozendict import frozendict

import torch_split.lib.assertions as assertions
import torch_split.lib.log as logging
import torch_split.lib.utils as utils
from torch_split.lib.ir import ConcreteNode, TorchGraph
from torch_split.lib.partition.dominance import DominanceInformation, VirtualNode

logger = logging.get_logger(__name__)

T = TypeVar("T", bound=VirtualNode | ConcreteNode)
VirtualNodeSet = frozenset[VirtualNode]
ConcreteNodeSet = frozenset[ConcreteNode]


@dataclass(frozen=True)
class Subgraph:
    inputs: ConcreteNodeSet
    """Subset of a cut's split set. Non-paremeter nodes. Should be input into the enclosed region"""

    outputs: ConcreteNodeSet
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
    """enforce the subgraph invariant formed by a cut"""

    cut: Cut
    """the cut that formed this partition"""

    subgraphs: frozenset[Subgraph]
    """the subgraphs formed by this cut"""


@dataclass(frozen=True)
class SwitchboardLayout:
    """Container for partition structure and corresponding GraphModules."""

    structure: dict
    """The JSON structure describing the partition layout"""

    modules: dict[str, fx.GraphModule]
    """Mapping of subgraph IDs to their extracted GraphModules"""

    def interpret(self, **external_inputs: Any) -> Any:
        """Execute the switchboard layout by running subgraphs in order.

        Args:
            **external_inputs: Input tensors keyed by their original node names

        Returns:
            The output from the final subgraph(s)
        """
        # Dictionary to store intermediate results indexed by node name
        intermediate_results: dict[str, Any] = dict(external_inputs)

        # Print initial inputs
        logger.info("[debug] external inputs:")
        for name, tensor in external_inputs.items():
            if isinstance(tensor, torch.Tensor):
                logger.info(f"[debug]   {name}: {tensor.shape}")
            else:
                logger.info(f"[debug]   {name}: {type(tensor)}")

        # Track which subgraphs have been executed
        executed: set[str] = set()

        # Execute entrypoints first
        for entrypoint in self.structure["entrypoint"]:
            subgraph_name = entrypoint["name"]
            input_names = entrypoint["inputs"]

            logger.info(f"[debug] executing entrypoint {subgraph_name}")
            for name in input_names:
                if name in intermediate_results:
                    tensor = intermediate_results[name]
                    if isinstance(tensor, torch.Tensor):
                        logger.info(f"[debug]   input {name}: {tensor.shape}")
                    else:
                        logger.info(f"[debug]   input {name}: {type(tensor)}")

            # Execute the subgraph with positional arguments
            module = self.modules[subgraph_name]
            subgraph_inputs = [intermediate_results[name] for name in input_names]
            outputs = module(*subgraph_inputs)

            logger.info(f"[debug] entrypoint {subgraph_name} completed")

            # Store outputs (handle both single and multiple returns)
            # Find the corresponding DFG node to get output names
            dfg_node = next(
                (n for n in self.structure["dfg"] if n["name"] == subgraph_name), None
            )
            output_names = dfg_node.get("outpus", []) if dfg_node else []

            if isinstance(outputs, (list, tuple)):
                for i, output in enumerate(outputs):
                    if isinstance(output, torch.Tensor):
                        logger.info(f"[debug]   output_{i}: {output.shape}")
                    else:
                        logger.info(f"[debug]   output_{i}: {type(output)}")
                    # Store with original node name if available, otherwise use indexed name
                    key = (
                        output_names[i]
                        if i < len(output_names)
                        else f"{subgraph_name}_output_{i}"
                    )
                    intermediate_results[key] = output
            else:
                if isinstance(outputs, torch.Tensor):
                    logger.info(f"[debug]   output_0: {outputs.shape}")
                else:
                    logger.info(f"[debug]   output_0: {type(outputs)}")
                # Store with original node name if available
                key = output_names[0] if output_names else f"{subgraph_name}_output_0"
                intermediate_results[key] = outputs

            executed.add(subgraph_name)

        # Execute remaining subgraphs following the DFG
        max_iterations = len(self.modules) * 2  # Safety limit to prevent infinite loops
        iteration = 0

        while len(executed) < len(self.modules) and iteration < max_iterations:
            iteration += 1
            found_executable = False

            for node in self.structure["dfg"]:
                subgraph_name = node["name"]

                if subgraph_name in executed:
                    continue

                # Check if all upstream dependencies are satisfied
                input_names = node["inputs"]
                if all(name in intermediate_results for name in input_names):
                    logger.info(f"[debug] executing {subgraph_name}")
                    for name in input_names:
                        if name in intermediate_results:
                            tensor = intermediate_results[name]
                            if isinstance(tensor, torch.Tensor):
                                logger.info(f"[debug]   input {name}: {tensor.shape}")
                            else:
                                logger.info(f"[debug]   input {name}: {type(tensor)}")

                    # Execute the subgraph with positional arguments
                    module = self.modules[subgraph_name]
                    subgraph_inputs = [
                        intermediate_results[name] for name in input_names
                    ]
                    outputs = module(*subgraph_inputs)

                    logger.info(f"[debug] {subgraph_name} completed")

                    # Store outputs
                    dfg_node = next(
                        (
                            n
                            for n in self.structure["dfg"]
                            if n["name"] == subgraph_name
                        ),
                        None,
                    )
                    output_names = dfg_node.get("outpus", []) if dfg_node else []

                    if isinstance(outputs, (list, tuple)):
                        for i, output in enumerate(outputs):
                            if isinstance(output, torch.Tensor):
                                logger.info(f"[debug]   output_{i}: {output.shape}")
                            else:
                                logger.info(f"[debug]   output_{i}: {type(output)}")
                            # Store with original node name if available, otherwise use indexed name
                            key = (
                                output_names[i]
                                if i < len(output_names)
                                else f"{subgraph_name}_output_{i}"
                            )
                            intermediate_results[key] = output
                    else:
                        if isinstance(outputs, torch.Tensor):
                            logger.info(f"[debug]   output_0: {outputs.shape}")
                        else:
                            logger.info(f"[debug]   output_0: {type(outputs)}")
                        # Store with original node name if available
                        key = (
                            output_names[0]
                            if output_names
                            else f"{subgraph_name}_output_0"
                        )
                        intermediate_results[key] = outputs

                    # Also store by output node names from downstream connections
                    for downstream in node.get("downstream", []):
                        for input_mapping in downstream.get("inputs", []):
                            output_name = input_mapping["output"]
                            if output_name in intermediate_results:
                                intermediate_results[input_mapping["input"]] = (
                                    intermediate_results[output_name]
                                )

                    executed.add(subgraph_name)
                    found_executable = True

            if not found_executable:
                break

        logger.info(f"[debug] execution complete, executed {len(executed)} subgraphs")

        # Return the final output from the last executed subgraph
        # Find the final node (C in this case - the one with empty downstream)
        for node in self.structure["dfg"]:
            if not node.get("downstream"):  # Node with no downstream is the final node
                final_output_names = node.get("outpus", [])
                if final_output_names:
                    final_key = final_output_names[0]
                    if final_key in intermediate_results:
                        return intermediate_results[final_key]

        # Fallback: return the last output found
        final_outputs = []
        for subgraph_name in reversed(sorted(executed)):
            dfg_node = next(
                (n for n in self.structure["dfg"] if n["name"] == subgraph_name), None
            )
            output_names = dfg_node.get("outpus", []) if dfg_node else []
            if output_names:
                for name in output_names:
                    if name in intermediate_results:
                        final_outputs.append(intermediate_results[name])

        return final_outputs[0] if len(final_outputs) == 1 else final_outputs

    def save_switchboard(self, output_path: Path) -> None:
        """Save partition data to a single proprietary directory"""
        output_path = output_path.with_suffix(".tspartd")
        output_path.mkdir(parents=True, exist_ok=True)

        json_data, graph_modules = self.structure, self.modules
        json_path = output_path / "structure.json"

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        for module_id, graph_module in graph_modules.items():
            module_path = output_path / f"{module_id}.pt"
            utils.save_graph(graph_module, module_path)

    @staticmethod
    def load_switchboard(filename: Path) -> "SwitchboardLayout":
        """Load partition data from a proprietary file (.tspart)."""
        assertions.file_extension(filename, ".tspartd")

        with open(filename / "structure.json", "r") as f:
            json_data = json.load(f)

        graph_modules = {}
        for module_file in filename.glob("*.pt"):
            assertions.file_extension(module_file, ".pt")
            graph_module = utils.load_graph(module_file)
            graph_modules[module_file.stem] = graph_module

        return SwitchboardLayout(structure=json_data, modules=graph_modules)


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
                for node in self._flood_fill_helper(
                    source, (cut.split | children) - source, next
                ):
                    visited_count[node] += 1

            # after we remove "children" from interacted nodes count, we are left with refined "cut.split" nodes
            refined_split: VirtualNodeSet = frozenset(
                filter(
                    lambda n: n not in children
                    and visited_count[n] >= max(len(children) - 1, 2),
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
            candidate_cut = Cut(
                split=frozenset([split_node]), join=frozenset([join_node])
            )
            enclosed_region = (
                self._get_core_region(candidate_cut) - candidate_cut.split
            ) - candidate_cut.join

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

            concrete_join: ConcreteNodeSet = ConcreteNodeSet(
                map(VirtualNode.to_concrete, cut.join)
            )

            # remove the join nodes so that our subgraph code actually will return disjoint subgraphs
            # print("[debug]: cut region: ", [n.name for n in cut.split], "→", [n.name for n in cut.join])
            # print("[debug]:     concrete split: ", [n.name for n in concrete_split])
            # print("[debug]:     concrete join: ", [n.name for n in concrete_join])
            # REMINDER: There is no need to substract concrete_split
            concrete_region = (
                self._get_enclosed_region(frozenset(concrete_split), concrete_join)
                - concrete_join
            )
            # print("[debug]:     er: ", concrete_region)
            subgraphs: list[ConcreteNodeSet] = list(
                self._get_subgraphs(frozenset(concrete_region), self._adjacent)
            )

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

    def carve_subgraphs(
        self, selected_partitions: Iterable[Partition]
    ) -> SwitchboardLayout:
        remaining_nodes = set(self.torch_graph.execution_order)
        all_subgraphs = []
        for partition in selected_partitions:
            for subgraph in partition.subgraphs:
                assertions.subset(subgraph.enclosed_region, remaining_nodes)
                remaining_nodes -= subgraph.enclosed_region
                all_subgraphs.append(subgraph)

        all_subgraphs += list(
            Subgraph(
                inputs=self._subgraph_inputs(sg),
                outputs=self._subgraph_outputs(sg),
                enclosed_region=sg,
            )
            for sg in self._get_subgraphs(frozenset(remaining_nodes), self._adjacent)
        )

        return self._create_json(all_subgraphs)

    def _create_json(self, all_subgraphs: list[Subgraph]) -> SwitchboardLayout:
        """Transform all_subgraphs list into nested hierarchical structure."""

        graph_modules: dict[str, fx.GraphModule] = {
            str(chr(ord("A") + idx)): utils.extract_subgraph(
                self.torch_graph.graph_module,
                list(
                    map(
                        self.torch_graph.to_fx,
                        self.torch_graph.sort_execution_order(subgraph.enclosed_region),
                    )
                ),
                list(map(self.torch_graph.to_fx, subgraph.inputs)),
                list(map(self.torch_graph.to_fx, subgraph.outputs)),
            )
            for idx, subgraph in enumerate(all_subgraphs)
        }

        id_to_char = {
            id(sg): str(chr(ord("A") + idx)) for idx, sg in enumerate(all_subgraphs)
        }

        # Build entrypoint section - subgraphs with no producers
        entrypoint = []
        dfg = []

        for producer in all_subgraphs:
            # Check if this subgraph has any producers
            has_producer = any(
                len(producer.inputs & candidate.outputs) > 0
                for candidate in all_subgraphs
                if candidate != producer
            )

            if not has_producer:
                entrypoint.append(
                    {
                        "name": id_to_char[id(producer)],
                        "inputs": list(map(lambda n: n.name, producer.inputs)),
                    }
                )

        # Build DFG section - all subgraphs with their downstream connections
        dfg = [
            {
                "name": id_to_char[id(producer)],
                "inputs": list(map(lambda n: n.name, producer.inputs)),
                "outpus": list(map(lambda n: n.name, producer.outputs)),
                "downstream": [
                    {
                        "to": id_to_char[id(consumer)],
                        "inputs": list(
                            [
                                {"output": inp.name, "input": inp.name}
                                for inp in consumer.inputs & producer.outputs
                            ]
                        ),
                    }
                    for consumer in all_subgraphs
                    if len(consumer.inputs & producer.outputs) > 0
                ],
            }
            for producer in all_subgraphs
        ]

        structure = {"entrypoint": entrypoint, "dfg": dfg}

        return SwitchboardLayout(structure=structure, modules=graph_modules)

    # [debug]: cut region:  ['hidden_states_51'] → ['hidden_states_63']
    def _adjacent(self, node: ConcreteNode) -> frozenset[ConcreteNode]:
        # if node.name == "mask":
        #     a = self.torch_graph.get_successors(node)
        #     b = self.torch_graph.get_predecessors(node)
        #     print("[debug]: getting adjacent for mask", a)
        #     print("[debug]: getting adjacent for mask", b)
        #     print("[debug]: getting adjacent for mask", a.union(b))
        return self.torch_graph.get_successors(node).union(
            self.torch_graph.get_predecessors(node)
        )

    def _get_subgraphs(
        self, node_set: frozenset[T], adjacent: Callable[[T], frozenset[T]]
    ) -> Iterable[frozenset[T]]:
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
            for node_a, node_b in itertools.combinations(
                self.dominance.ordered_nodes(), 2
            ):
                is_d = node_a in self.dominance.dominators(node_b)
                is_p = node_b in self.dominance.post_dominators(node_a)
                if is_d and is_p:
                    yield Cut(frozenset({node_a}), frozenset({node_b}))

        return frozenset(lazy_iterable())

    def _get_enclosed_region(
        self, source: ConcreteNodeSet, sink: ConcreteNodeSet
    ) -> ConcreteNodeSet:
        """Get the region between source and sink nodes, **including** parameter nodes. Parameter nodes
        are constexpr-foldable. source node and sink node are included in the region."""

        def next(n: ConcreteNode) -> frozenset[ConcreteNode]:
            if n in source:
                return self.torch_graph.get_successors(n)
            elif n in sink:
                return self.torch_graph.get_predecessors(n)
            else:
                return self.torch_graph.get_successors(n).union(
                    self.torch_graph.get_predecessors(n)
                )

        return self._flood_fill_helper(source, sink, next)

    def _get_core_region(self, cut: Cut) -> VirtualNodeSet:
        """Get the region between source and sink nodes, not including parameter nodes. Parameter nodes
        are constexpr-foldable. source node and sink node are included in the region."""
        return self._flood_fill_helper(
            cut.join, cut.split, self.dominance.get_successors
        )

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

    # #     def generate_tree(
    # #         self,
    # #         min_compute_share: float = 0.10,
    # #     ) -> tuple[
    # #         "PartitionProvider",
    # #         list["PartitionProvider.PartitionCandidate"],
    # #         dict[
    # #             "PartitionProvider.PartitionCandidate",
    # #             set["PartitionProvider.PartitionCandidate"],
    # #         ],
    # #     ]:
    # #         logger.info("Creating partitions | minimum compute share %f", min_compute_share)

    # #         source = self._dominance.dom_root()
    # #         sink = self._dominance.pdom_root()

    # #         entire_graph = self._get_partition_from_cut(PartitionProvider.Cut(source, sink))
    # #         entire_graph_execution_time = self.total_execution_time(entire_graph)

    # #         # outline: enumerate -> eval -> prune (invalid/subsumed/intersecting) -> pack into ≤ gpu_budget -> rank

    # #         # only consider candidates that meet the minimum compute share
    # #         partition_candidates = list(
    # #             filter(
    # #                 lambda pc: self._highest_empirical_weight(
    # #                     pc.execution_time_ms, entire_graph_execution_time, 0
    # #                 )
    # #                 >= 0,
    # #                 self._generate_all_partition_candidates(),
    # #             )
    # #         )

    # #         # sort, prioritizing highest compute share
    # #         partition_candidates = sorted(
    # #             partition_candidates,
    # #             key=lambda pc: -self._highest_empirical_weight(
    # #                 pc.execution_time_ms, entire_graph_execution_time, 0
    # #             ),
    # #         )

    # #         # prune intersecting candidates
    # #         accepted: list["PartitionProvider.PartitionCandidate"] = []
    # #         for p in partition_candidates:
    # #             fraction = self._highest_empirical_weight(
    # #                 p.execution_time_ms, entire_graph_execution_time, 0
    # #             )
    # #             # if fraction < min_compute_share:
    # #             #     continue

    # #             if (
    # #                 not any(
    # #                     self.intersects(p.partition, a.partition)
    # #                     and (
    # #                         not (
    # #                             self.subsumes(p.partition, a.partition)
    # #                             or self.subsumes(a.partition, p.partition)
    # #                         )
    # #                     )
    # #                     for a in accepted
    # #                 )
    # #                 and len(p.partition.subgraphs) > 1
    # #             ):
    # #                 print(
    # #                     "accepted partition candidate:",
    # #                     p.partition.cut.source.name,
    # #                     "→",
    # #                     p.partition.cut.sink.name,
    # #                     fraction,
    # #                 )
    # #                 accepted.append(p)

    # #         # prune subsumed candidates
    # #         logger.info("Selected %d partitions after pruning", len(accepted))

    # #         subsumption: dict[
    # #             PartitionProvider.PartitionCandidate,
    # #             set[PartitionProvider.PartitionCandidate],
    # #         ] = {}
    # #         for i, a in enumerate(accepted):
    # #             for j, b in enumerate(accepted):
    # #                 if i != j and self.subsumes(a.partition, b.partition):
    # #                     subsumption.setdefault(a, set()).add(b)
    # #         for p in list(subsumption):
    # #             t, stack = set(), list(subsumption[p])
    # #             while stack:
    # #                 c = stack.pop()
    # #                 for g in subsumption.get(c, ()):
    # #                     if g not in t:
    # #                         t.add(g)
    # #                         stack.append(g)
    # #             subsumption[p] -= t
    # #         roots = [
    # #             p
    # #             for p in subsumption
    # #             if p not in {c for v in subsumption.values() for c in v}
    # #         ]

    # #         logger.info("Selecting partitions now; this might take a moment...")

    # #         return (self, roots, subsumption)

    # #     def _highest_empirical_weight(
    # #         self,
    # #         partition: frozendict[int, tuple[float, float]],
    # #         total: frozendict[int, tuple[float, float]],
    # #         zscore: float,
    # #     ) -> float:
    # #         assert partition.keys() == total.keys(), (
    # #             "Partition and total must have the same batch sizes"
    # #         )

    # #         best = 0.0
    # #         for batch_size in partition.keys():
    # #             part_avg, part_std = partition[batch_size]
    # #             total_avg, total_std = total[batch_size]
    # #             best = max(
    # #                 best,
    # #                 (part_avg + part_std * zscore)
    # #                 / max(total_avg - total_std * zscore, 1e-8),
    # #             )

    # #         return best

    # #         return frozenset(settled)
    # #     @lru_cache(maxsize=16384)
    # #     def is_disjoint(
    # #         self, a: "PartitionProvider.Partition", b: "PartitionProvider.Partition"
    # #     ) -> bool:
    # #         """Regions are disjoint (safe to co-exist)."""
    # #         return len(a.total_enclosed_region & b.total_enclosed_region) == 0

    # #     @lru_cache(maxsize=16384)
    # #     def total_execution_time(
    # #         self, partition: "PartitionProvider.Partition"
    # #     ) -> frozendict[int, tuple[float, float]]:
    # #         """Estimate total execution time of a partition by summing subgraph times.

    # #         Args:
    # #             partition (PartitionProvider.Partition): assumes sequential execution of subgraphs

    # #         Returns:
    # #             tuple[float, float]: avg execution time, std execution time
    # #         """

    # #         ret: defaultdict[int, list[int]] = defaultdict(lambda: [0, 0])

    # #         for subgraph in partition.subgraphs:
    # #             # enclosed region must be converted to NodeId
    # #             enclosed_region = subgraph.enclosed_region
    # #             for v_node in enclosed_region:
    # #                 if v_node.is_exit() or v_node.is_entry():
    # #                     continue
    # #                 node = self.torch_graph.concrete_to_node[v_node.to_concrete()]
    # #                 for batch_size, batch_metrics in node.meta[
    # #                     "torch_split_profiling"
    # #                 ].items():
    # #                     ret[int(batch_size)][0] += batch_metrics["avg_time_ms"]
    # #                     ret[int(batch_size)][1] += batch_metrics["std_time_ms"] ** 2

    # #         return frozendict({k: (v[0], v[1] ** 0.5) for k, v in ret.items()})

    # #     @lru_cache(maxsize=16384)
    # #     def maximum_memory_usage(
    # #         self, partition: "PartitionProvider.Partition"
    # #     ) -> tuple[float, float]:
    # #         """Estimate maximum memory usage of a partition by taking a maximum across nodes in the partition

    # #         Args:
    # #             partition (PartitionProvider.Partition): partition to evaluate

    # #         Returns:
    # #             tuple[float, float]: avg memory usage, std memory usage
    # #         """

    # #         max_avg_memory_bytes = 0.0
    # #         max_std_memory_bytes = 0.0

    # #         for subgraph in partition.subgraphs:
    # #             for v_node in subgraph.enclosed_region:
    # #                 node = self.torch_graph.concrete_to_node[v_node.to_concrete()]
    # #                 node_avg_mem = node.meta["torch_split_profiling"]["avg_memory_bytes"]
    # #                 node_std_mem = node.meta["torch_split_profiling"]["std_memory_bytes"]

    # #                 max_avg_memory_bytes = max(node_avg_mem, max_avg_memory_bytes)
    # #                 max_std_memory_bytes = max(node_std_mem, max_std_memory_bytes)

    # #         return max_avg_memory_bytes, max_std_memory_bytes

    def _subgraph_inputs(self, subgraph_nodes: ConcreteNodeSet) -> ConcreteNodeSet:
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

        return frozenset(ret)

    def _subgraph_outputs(self, subgraph_nodes: ConcreteNodeSet) -> ConcreteNodeSet:
        """Nodes with no successors in the subgraph."""
        return frozenset(
            n
            for n in subgraph_nodes
            if not (self.torch_graph.get_successors(n) & subgraph_nodes)
        )
