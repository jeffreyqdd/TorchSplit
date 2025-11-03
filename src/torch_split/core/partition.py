import uuid
from collections import defaultdict
from collections.abc import Iterable, Set
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import graphviz  # type: ignore

import torch_split.logging as logging
from torch_split.core.dominance import DominanceInformation, VirtualNodeId
from torch_split.core.ir import TorchGraph

logger = logging.get_logger(__name__)

NodeId = uuid.UUID


class PartitionProvider:
    """
    Plans graph partitions using dominance-aware cut candidates.
    Responsibilities:
      - build dominance info from a TorchGraph
      - enumerate/validate cuts (inputs, outputs)
      - compute regions for cuts (node sets)
      - compare cuts: subsumption / intersection / disjointness
      - select a maximal non-overlapping set by cost (branch-parallel first)
    """

    @dataclass(frozen=True)
    class Cut:
        """A preliminary cut candidate defined by source and sink nodes."""

        source: VirtualNodeId
        sink: VirtualNodeId

    @dataclass(frozen=True)
    class Partition:
        """A boundary described by entering nodes (inputs) and exiting nodes (outputs). A cut
        is valid if combined, every input node dominates every output node, and every output
        node post-dominates every input node.

        In other words, all paths from inputs to exit must pass through an output node,
        and all paths from entry to outputs must pass through an input node.
        """

        @dataclass(frozen=True)
        class Subgraph:
            inputs: frozenset[NodeId]
            """input node ids that have no incoming edges from within the subcomponent"""

            outputs: frozenset[NodeId]
            """output node ids that have no outgoing edges within the subcomponent"""

            enclosed_region: frozenset[NodeId]
            """all node ids in this subcomponent including inputs and outputs"""

        cut: "PartitionProvider.Cut"

        subgraphs: frozenset["PartitionProvider.Partition.Subgraph"]
        """complete and disjoint"""

        @property
        def source(self) -> VirtualNodeId:
            return self.cut.source

        @property
        def sink(self) -> VirtualNodeId:
            return self.cut.sink

        @cached_property
        def total_enclosed_region(self) -> frozenset[NodeId]:
            """All node ids in this partition (union of all subgraph regions)."""
            return frozenset().union(*(sg.enclosed_region for sg in self.subgraphs))

        def __post_init__(self):
            if bool(self.source) != bool(self.sink):
                raise ValueError("Both source and sink must be None or both must be non-None")

    @dataclass(frozen=True)
    class PartitionEval:
        """Cached evaluation of a partition; enables ranking & pruning."""

        partition: "PartitionProvider.Partition"
        """the partition being evaluated"""

        bytes_in: int
        """cumulative size of activations into the partition"""

        bytes_out: int
        """cumulative size of activations out of the partition"""

        flops: float
        """total flops in the region"""

        peak_mem_bytes: int  # estimated peak per microbatch
        """total peak memory usage in bytes"""

    # -----------------------------
    # Construction
    # -----------------------------
    def __init__(self, torch_graph: TorchGraph):
        self._dominance = DominanceInformation(torch_graph)

    @property
    def torch_graph(self) -> TorchGraph:
        return self._dominance.torch_graph

    # -----------------------------
    # Public API Visualization
    # -----------------------------
    def visualize_dataflow(self, export_path: Path, include_parameters: bool = False):
        """export the dataflow graph as a pdf"""
        export_path.absolute().mkdir(parents=True, exist_ok=True)
        dot = graphviz.Digraph(name="Dataflow Graph")
        dot.attr(nodesep="0.1", ranksep="0.3")
        self.torch_graph.render_graph(dot, include_parameters=include_parameters)
        dot.render(export_path / "dataflow_graph", format="pdf")

    def visualize_dominance(self, export_path: Path):
        """exports four graphs: dominance tree, post-dominance tree, reverse dominance tree, reverse post-dominance tree"""

        export_path.absolute().mkdir(parents=True, exist_ok=True)

        dom_gv = graphviz.Digraph(name="Dominance Set")
        dom_gv.attr(nodesep="0.1", ranksep="0.3")
        self._dominance.render_graph(self._dominance.dom, dom_gv, flatten=True)
        dom_gv.render(export_path / "dominance_set", format="pdf")

        dom_tree_gv = graphviz.Digraph(name="Dominance Tree")
        dom_tree_gv.attr(nodesep="0.1", ranksep="0.3")
        self._dominance.render_graph(self._dominance.dom_tree, dom_tree_gv)
        dom_tree_gv.render(export_path / "dominance_tree", format="pdf")

        pdom_tree_gv = graphviz.Digraph(name="Post Dominance Tree")
        pdom_tree_gv.attr(nodesep="0.1", ranksep="0.3")
        self._dominance.render_graph(self._dominance.pdom_tree, pdom_tree_gv)
        pdom_tree_gv.render(export_path / "post_dominance_tree", format="pdf")

        rev_dom_tree_gv = graphviz.Digraph(name="Reverse Dominance Tree")
        rev_dom_tree_gv.attr(nodesep="0.1", ranksep="0.3")
        self._dominance.render_graph(self._dominance.reverse_dom_tree, rev_dom_tree_gv)
        rev_dom_tree_gv.render(export_path / "rev_dominance_tree", format="pdf")

        rev_pdom_tree_gv = graphviz.Digraph(name="Reverse Post Dominance Tree")
        rev_pdom_tree_gv.attr(nodesep="0.1", ranksep="0.3")
        self._dominance.render_graph(self._dominance.reverse_pdom_tree, rev_pdom_tree_gv)
        rev_pdom_tree_gv.render(export_path / "rev_post_dominance_tree", format="pdf")

    def create_partition(
        self,
        gpu_budget: int = 0,
        strategy: str = "branch_parallel",  # future: "hybrid", "pipeline"
        top_k: int = 1,
        alpha: float = 1.0,  # compute imbalance weight
        beta: float = 1.0,  # communication weight
        gamma: float = 1000.0,  # memory violation penalty
        beta_tp: float = 10.0,  # TP disruption penalty (if used)
    ) -> list["PartitionProvider.PartitionEval"]:
        # outline: enumerate -> eval -> prune (invalid/subsumed/intersecting) -> pack into ≤ gpu_budget -> rank

        sorted_partition_evaluations = sorted(self._generate_all_partition_candidates(), key=lambda x: -x.flops)
        accepted: list[PartitionProvider.PartitionEval] = []
        for p in sorted_partition_evaluations:
            if not any(self.intersects(p.partition, a.partition) for a in accepted):
                accepted.append(p)

        accepted = sorted(accepted, key=lambda x: -len(x.partition.total_enclosed_region))

        for a in accepted:
            logger.info(
                "Accepted partition with %s %s subgraphs, with %d nodes",
                self._dominance.name_of(a.partition.source),
                self._dominance.name_of(a.partition.sink),
                len(a.partition.total_enclosed_region),
            )

        # for p in accepted:
        #     # Pop until we find a region that subsumes this one
        #     while stack and not self.subsumes(stack[-1].partition, p.partition):
        #         stack.pop()

        #     parent = stack[-1] if stack else None
        #     parent_of[p] = parent

        #     if parent:
        #         region_tree.setdefault(parent, []).append(p)
        #     else:
        #         region_tree.setdefault(None, []).append(p)  # root-level region

        #     stack.append(p)

        # # Step 3: Optionally log the resulting hierarchy
        # def _log_tree(node: PartitionProvider.PartitionEval | None, depth: int = 0):
        #     for child in region_tree.get(node, []):
        #         indent = "  " * depth
        #         logger.info(
        #             "%sRegion %s→%s | %d nodes | %.2f MFLOPs",
        #             indent,
        #             self._dominance.name_of(child.partition.source),
        #             self._dominance.name_of(child.partition.sink),
        #             len(child.partition.total_enclosed_region),
        #             child.flops / 1e6,
        #         )
        #         _log_tree(child, depth + 1)

        # logger.info("=== Region Tree ===")
        # _log_tree(None)

        # accepted is a set of of Partitions which form a laminar family, meaning, that each pair of sets are
        # either disjoint or related by subsumption

        # if keep:
        #     accepted.add(evaluated_partition)
        # else:
        #     accepted.difference_update(removed)
        # logger.debug(
        #     "Partition with %s %s subgraphs, flops=%.2f",
        #     self._dominance.name_of(partition.partition.source),
        #     self._dominance.name_of(partition.partition.sink),
        #     partition.flops,
        # )

        return []

    # # -----------------------------
    # # Candidate generation & evaluation
    # # -----------------------------
    # def enumerate_branch_parallel_regions(self) -> List[Tuple[NodeId, NodeId, List[FrozenSet[NodeId]]]]:
    #     """
    #     Detect fan-out/fan-in regions using dominance:
    #       returns a list of (entry, exit, branches) where branches are disjoint subgraphs.
    #     Use as the basis for ENTRY/EXIT cut candidates.
    #     """
    #     raise NotImplementedError

    # def enumerate_candidates(self, mode: str = "branch_parallel") -> List["PartitionProvider.Cut"]:
    #     """
    #     Return canonicalized cut candidates for the selected mode.
    #     For branch_parallel, produce ENTRY and EXIT cuts around each parallel region.
    #     """
    #     raise NotImplementedError

    # @lru_cache(maxsize=16384)
    # def canonicalize(self, cut: "PartitionProvider.Cut") -> "PartitionProvider.Cut":
    #     """
    #     Normalize inputs/outputs to a canonical boundary:
    #       - drop redundant inputs dominated by other inputs
    #       - drop redundant outputs post-dominated by other outputs
    #       - sort/freeze for stable hashing
    #     Prevents duplicate logical cuts.
    #     """
    #     raise NotImplementedError

    # @lru_cache(maxsize=16384)
    # def evaluate(self, cut: "PartitionProvider.Cut") -> "PartitionProvider.CutEval":
    #     """
    #     Build the region via dominance-closure and compute boundary + cost metrics.
    #     Use dominance/post-dominance for O(1) validity checks where possible.
    #     """
    #     raise NotImplementedError

    # # -----------------------------
    # # Validity & relations
    # # -----------------------------

    @lru_cache(maxsize=16384)
    def valid_cut(self, cut: "PartitionProvider.Cut") -> bool:
        """Fast validity check for a cut. Every input must dominate every output, and every
        output must post-dominate every input. Furthermore, source and sink cannot be directly connected.
        """
        pdoms = self._dominance.pdom[cut.source]
        doms = self._dominance.dom[cut.sink]

        if not (cut.sink in pdoms and cut.source in doms):
            return False

        successors = self._dominance.successors_of(cut.source)
        predecessors = self._dominance.predecessors_of(cut.sink)

        if cut.source in predecessors or cut.sink in successors:
            return False

        return True

    @lru_cache(maxsize=16384)
    def enclosed_region(self, sese: "PartitionProvider.Cut") -> frozenset[VirtualNodeId]:
        """return the set of nodes enclosed by the single-entry-single-exit region, excluding the source and sink nodes"""

        if not self.valid_cut(sese):
            raise ValueError("Cannot compute region for invalid cut")

        reachable = {*self._dominance.predecessors_of(sese.sink)}
        worklist = [*self._dominance.successors_of(sese.source)]

        while worklist:
            virtual_node = worklist.pop()

            if virtual_node in reachable:
                continue

            reachable.add(virtual_node)
            for successor in self._dominance.successors_of(virtual_node):
                if successor not in reachable:
                    worklist.append(successor)

        return frozenset(reachable)

    @lru_cache(maxsize=16384)
    def subsumes(self, a: "PartitionProvider.Partition", b: "PartitionProvider.Partition") -> bool:
        """returns true if b's region is a subset of a's region"""
        return b.total_enclosed_region <= a.total_enclosed_region

    @lru_cache(maxsize=16384)
    def intersects(self, a: "PartitionProvider.Partition", b: "PartitionProvider.Partition") -> bool:
        """returns true if regions overlap, but neither subsumes the other"""
        not_empty = len(a.total_enclosed_region & b.total_enclosed_region) != 0
        return not_empty and not self.subsumes(a, b) and not self.subsumes(b, a)

    @lru_cache(maxsize=16384)
    def is_disjoint(self, a: "PartitionProvider.Partition", b: "PartitionProvider.Partition") -> bool:
        """Regions are disjoint (safe to co-exist)."""
        return len(a.total_enclosed_region & b.total_enclosed_region) == 0

    # -----------------------------
    # Selection / ranking
    # -----------------------------
    # def cost(
    #     self, cuts: Sequence["PartitionProvider.CutEval"], alpha: float, beta: float, gamma: float, beta_tp: float = 0.0
    # ) -> float:
    #     """
    #     Aggregate cost for a plan (or single cut):
    #       α * compute_bottleneck + β * boundary_comm + γ * memory_violations + β_tp * tp_disruptions
    #     For branch-parallel, compute_bottleneck is max(region.flops).
    #     """

    #     raise NotImplementedError

    # def select_maximal_non_overlapping(
    #     self,
    #     candidates: Sequence["PartitionProvider.Cut"],
    #     gpu_budget: int,
    #     alpha: float,
    #     beta: float,
    #     gamma: float,
    #     beta_tp: float = 0.0,
    # ) -> List["PartitionProvider.CutEval"]:
    #     """
    #     Greedy packing:
    #       - sort by (valid, cost asc)
    #       - add cut if it is disjoint from chosen set
    #       - skip if subsumed by an already chosen maximal cut
    #       - stop at gpu_budget (one region per GPU in branch-parallel)
    #     """
    #     raise NotImplementedError

    def _get_partition_from_cut(self, cut: "PartitionProvider.Cut") -> "PartitionProvider.Partition":
        if not self.valid_cut(cut):
            raise ValueError("Cannot create cut from invalid SESE")

        # get all nodes in the enclosed region, excluding the source and sink nodes
        enclosed_region = self.enclosed_region(cut)
        logger.debug("[dim]Analyzing enclosed region with %d nodes[/]", len(enclosed_region))

        # create an undirected graph of the enclosed region
        undirected_graph: dict[VirtualNodeId, set[VirtualNodeId]] = defaultdict(set)
        for node_uid in enclosed_region:
            for succ in self._dominance.successors_of(node_uid):
                if succ in enclosed_region:
                    undirected_graph[node_uid].add(succ)
                    undirected_graph[succ].add(node_uid)

        # flood fill the undirected graph to find disjoint subgraphs

        visited = set()
        subgraphs: list[set[VirtualNodeId]] = []

        for node in enclosed_region:
            if node not in visited:
                component: set[VirtualNodeId] = set()
                worklist = [node]
                visited.add(node)

                while worklist:
                    current = worklist.pop()
                    component.add(current)
                    for successor in undirected_graph[current]:
                        if successor not in visited:
                            visited.add(successor)
                            worklist.append(successor)
                subgraphs.append(component)

        logger.debug("[dim]Found %d disjoint subgraphs in partition[/]", len(subgraphs))

        subcomponents: set[PartitionProvider.Partition.Subgraph] = set()
        for subgraph in subgraphs:
            input_uids = self._dominance.safe_unwrap_iterable(self._find_component_inputs(subgraph))
            output_uids = self._dominance.safe_unwrap_iterable(self._find_component_outputs(subgraph))
            subcomponents.add(
                PartitionProvider.Partition.Subgraph(
                    frozenset(input_uids),
                    frozenset(output_uids),
                    frozenset(self._dominance.safe_unwrap_iterable(subgraph)),
                )
            )

        return PartitionProvider.Partition(cut, frozenset(subcomponents))

    def _find_component_inputs(self, subgraph_nodes: Set[VirtualNodeId]) -> frozenset[VirtualNodeId]:
        """Find nodes in component with no incoming edges from within the component."""

        # inputs have no in edges, meaning no predecessors in the subgraph
        inputs = set()

        for node in subgraph_nodes:
            if len(self._dominance.predecessors_of(node) & subgraph_nodes) == 0:
                inputs.add(node)

        return frozenset(inputs)

    def _find_component_outputs(self, component_nodes: set[VirtualNodeId]) -> frozenset[VirtualNodeId]:
        """Find nodes in component with no outgoing edges within the component."""

        # outputs have no out edges, meaning no successors in the subgraph
        outputs = set()

        for node in component_nodes:
            if len(self._dominance.successors_of(node) & component_nodes) == 0:
                outputs.add(node)

        return frozenset(outputs)

    def _evaluate_partition(self, partition: "PartitionProvider.Partition") -> "PartitionProvider.PartitionEval":
        """Evaluate a partition to compute metrics like bytes_in, bytes_out, flops, and peak_mem_bytes."""

        # TODO (jq54): implement accurate computation of these metrics from traces
        bytes_in = 0
        bytes_out = 0
        flops = 0.0
        peak_mem_bytes = 0

        for subgraph in partition.subgraphs:
            # Compute bytes_in and bytes_out based on inputs and outputs
            for _ in subgraph.inputs:
                bytes_in += 1
                # bytes_in += self.torch_graph.get_activation_size(input_node)

            for _ in subgraph.outputs:
                bytes_in += 1
                # bytes_out += self.torch_graph.get_activation_size(output_node)

            # Compute flops and peak memory for the enclosed region
            for _ in subgraph.enclosed_region:
                flops += 1
                peak_mem_bytes += 1
                # flops += self.torch_graph.get_flops(node)
                # peak_mem_bytes = max(peak_mem_bytes, self.torch_graph.get_peak_memory(node))

        return PartitionProvider.PartitionEval(
            partition=partition,
            bytes_in=bytes_in,
            bytes_out=bytes_out,
            flops=flops,
            peak_mem_bytes=peak_mem_bytes,
        )

    @logging.timer(name="Partition generation")
    def _generate_all_partition_candidates(
        self,
    ) -> Iterable["PartitionProvider.PartitionEval"]:
        logger.info(
            "[bold cyan]Starting partition generation[/] [dim]for graph with %d nodes[/]",
            len(list(self._dominance.nodes())),
        )

        partitions = []
        total_cut_count = 0
        valid_cut_count = 0
        processed_sources = 0

        # look at dominance relationships to find a source and sink node that form a SESE region
        source_nodes = list(self._dominance.nodes())
        logger.debug(
            "[dim]Exploring %d potential source nodes for SESE regions[/]",
            len(source_nodes),
        )

        if len(source_nodes) > 1000:
            logger.warning(
                "[yellow]Large graph detected (%d nodes) - partition generation may take some time[/]",
                len(source_nodes),
            )

        for source_node in source_nodes:
            processed_sources += 1
            successor_nodes = self._dominance.successors_of(source_node)

            if not successor_nodes:
                continue

            # Log progress for large graphs
            if len(source_nodes) > 100 and processed_sources % max(1, len(source_nodes) // 10) == 0:
                logger.debug(
                    "[cyan]Progress[/]: %d/%d source nodes processed [dim](%.1f%%)[/]",
                    processed_sources,
                    len(source_nodes),
                    100.0 * processed_sources / len(source_nodes),
                )
            successor_nodes = self._dominance.successors_of(source_node)

            if not successor_nodes:
                continue

            # SESE regions contain of a source node that dominates a sink node
            # and the sink node post-dominates the source node
            candidate_pdom_uids = set.intersection(
                set(self._dominance.nodes()),
                *(self._dominance.pdom[b] for b in successor_nodes),
            )
            dominated_pdom_uids = set(filter(lambda n: source_node in self._dominance.dom[n], candidate_pdom_uids))

            for sink_node in dominated_pdom_uids:
                total_cut_count += 1
                candidate_cut = PartitionProvider.Cut(source_node, sink_node)
                if self.valid_cut(candidate_cut) is False:
                    continue

                valid_cut_count += 1
                source_name = self._dominance.name_of(source_node)
                sink_name = self._dominance.name_of(sink_node)

                logger.debug(
                    "[cyan]Processing SESE[/]: [yellow]%s[/] -> [yellow]%s[/]",
                    source_name,
                    sink_name,
                )
                partition = self._get_partition_from_cut(candidate_cut)

                for idx, subgraph in enumerate(partition.subgraphs):
                    input_names = [self.torch_graph.name_from_uid(uid) for uid in subgraph.inputs]
                    output_names = [self.torch_graph.name_from_uid(uid) for uid in subgraph.outputs]

                    # Format names with truncation for readability
                    inputs_display = ", ".join(input_names[:3]) + ("..." if len(input_names) > 3 else "")
                    outputs_display = ", ".join(output_names[:3]) + ("..." if len(output_names) > 3 else "")

                    logger.debug(
                        "  [green]Subgraph %d[/]: [blue]inputs[/]=[cyan]%s[/], [magenta]outputs[/]=[cyan]%s[/] [dim](%d nodes)[/]",
                        idx + 1,
                        inputs_display,
                        outputs_display,
                        len(subgraph.enclosed_region),
                    )

                partitions.append(partition)

        logger.info(
            "[bold green]Partition generation complete![/] "
            "[blue]Found %d valid partitions[/] [dim]out of %d candidates[/]",
            len(partitions),
            total_cut_count,
        )

        # Log LRU cache statistics for performance analysis
        valid_cut_info = self.valid_cut.cache_info()
        enclosed_region_info = self.enclosed_region.cache_info()

        logger.debug(
            "[dim]LRU Cache Stats - valid_cut: hits=%d, misses=%d, hit_rate=%.1f%%, size=%d[/]",
            valid_cut_info.hits,
            valid_cut_info.misses,
            100.0 * valid_cut_info.hits / (valid_cut_info.hits + valid_cut_info.misses)
            if (valid_cut_info.hits + valid_cut_info.misses) > 0
            else 0.0,
            valid_cut_info.currsize,
        )

        logger.debug(
            "[dim]LRU Cache Stats - enclosed_region: hits=%d, misses=%d, hit_rate=%.1f%%, size=%d[/]",
            enclosed_region_info.hits,
            enclosed_region_info.misses,
            100.0 * enclosed_region_info.hits / (enclosed_region_info.hits + enclosed_region_info.misses)
            if (enclosed_region_info.hits + enclosed_region_info.misses) > 0
            else 0.0,
            enclosed_region_info.currsize,
        )

        # TODO: move this step into the generation process so I at least have logs
        return map(self._evaluate_partition, partitions)
