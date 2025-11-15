import uuid
from collections import defaultdict, deque
from collections.abc import Iterable, Set
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import graphviz  # type: ignore
from frozendict import frozendict

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
    class PartitionCandidate:
        """Cached evaluation of a partition; enables ranking & pruning."""

        partition: "PartitionProvider.Partition"
        """the partition being evaluated"""

        execution_time_ms: frozendict[int, tuple[float, float]]

        # bytes_in: frozendict[int, tuple[float, float]]
        # """cumulative size of activations into the partition"""

        # bytes_out: frozendict[int, tuple[float, float]]
        # """cumulative size of activations out of the partition"""

        # signifiance: float = 0.0
        # """importance score of this partition (higher is more important), based on computational """

        # peak_mem_bytes: int  # estimated peak per microbatch
        # """total peak memory usage in bytes"""

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

    def generate_tree(
        self,
        min_compute_share: float = 0.10,
    ) -> tuple[
        "PartitionProvider",
        dict[
            "PartitionProvider.PartitionCandidate",
            set["PartitionProvider.PartitionCandidate"],
        ],
    ]:
        logger.info("Creating partitions | minimum compute share %f", min_compute_share)

        source = self._dominance.dom_root()
        sink = self._dominance.pdom_root()

        entire_graph = self._get_partition_from_cut(PartitionProvider.Cut(source, sink))
        entire_graph_execution_time = self.total_execution_time(entire_graph)

        # outline: enumerate -> eval -> prune (invalid/subsumed/intersecting) -> pack into ≤ gpu_budget -> rank

        # only consider candidates that meet the minimum compute share
        partition_candidates = list(
            filter(
                lambda pc: self._highest_empirical_weight(pc.execution_time_ms, entire_graph_execution_time, 0) >= 0,
                self._generate_all_partition_candidates(),
            )
        )

        # sort, prioritizing highest compute share
        partition_candidates = sorted(
            partition_candidates,
            key=lambda pc: -self._highest_empirical_weight(pc.execution_time_ms, entire_graph_execution_time, 0),
        )

        # prune intersecting candidates
        accepted: list["PartitionProvider.PartitionCandidate"] = []
        for p in partition_candidates:
            fraction = self._highest_empirical_weight(p.execution_time_ms, entire_graph_execution_time, 0)
            if fraction < min_compute_share:
                continue

            if not any(
                self.intersects(p.partition, a.partition) and not self.subsumes(p.partition, a.partition)
                for a in accepted
            ):
                accepted.append(p)

        # prune subsumed candidates
        logger.info("Selected %d partitions after pruning", len(accepted))
        for a in accepted:
            logger.debug(
                "Accepted partition %s %s wth %d subgraphs, with %d nodes | compute share: %f%%",
                self._dominance.name_of(a.partition.source),
                self._dominance.name_of(a.partition.sink),
                len(a.partition.subgraphs),
                len(a.partition.total_enclosed_region),
                self._highest_empirical_weight(a.execution_time_ms, entire_graph_execution_time, 0),
            )

        subsumption: dict[
            PartitionProvider.PartitionCandidate,
            set[PartitionProvider.PartitionCandidate],
        ] = {}
        for i, a in enumerate(accepted):
            for j, b in enumerate(accepted):
                if i != j and self.subsumes(a.partition, b.partition):
                    subsumption.setdefault(a, set()).add(b)
        for p in list(subsumption):
            t, stack = set(), list(subsumption[p])
            while stack:
                c = stack.pop()
                for g in subsumption.get(c, ()):
                    if g not in t:
                        t.add(g)
                        stack.append(g)
            subsumption[p] -= t
        roots = [p for p in subsumption if p not in {c for v in subsumption.values() for c in v}]

        # # log resulting hierarchy
        def _log_tree(node: PartitionProvider.PartitionCandidate, depth: int = 0):
            indent = "  " * depth
            logger.debug(
                "%sRegion %s→%s | %d nodes",
                indent,
                self._dominance.name_of(node.partition.source),
                self._dominance.name_of(node.partition.sink),
                len(node.partition.total_enclosed_region),
            )

            for child in subsumption.get(node, []):
                _log_tree(child, depth + 1)

        for root in roots:
            _log_tree(root)

        logger.info("Selecting partitions now; this might take a moment...")

        return (self, subsumption)

    def _highest_empirical_weight(
        self,
        partition: frozendict[int, tuple[float, float]],
        total: frozendict[int, tuple[float, float]],
        zscore: float,
    ) -> float:
        assert partition.keys() == total.keys(), "Partition and total must have the same batch sizes"

        best = 0.0
        for batch_size in partition.keys():
            part_avg, part_std = partition[batch_size]
            total_avg, total_std = total[batch_size]
            best = max(
                best,
                (part_avg + part_std * zscore) / max(total_avg - total_std * zscore, 1e-8),
            )

        return best

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

    # -----------------------------
    # Validity & relations
    # -----------------------------

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

    @lru_cache(maxsize=16384)
    def total_execution_time(self, partition: "PartitionProvider.Partition") -> frozendict[int, tuple[float, float]]:
        """Estimate total execution time of a partition by summing subgraph times.

        Args:
            partition (PartitionProvider.Partition): assumes sequential execution of subgraphs

        Returns:
            tuple[float, float]: avg execution time, std execution time
        """

        ret: defaultdict[int, list[int]] = defaultdict(lambda: [0, 0])

        for subgraph in partition.subgraphs:
            for node_uid in subgraph.enclosed_region:
                node = self.torch_graph.node_from_uid(node_uid)
                for batch_size, batch_metrics in node.node.meta["torch_split_profiling"].items():
                    ret[int(batch_size)][0] += batch_metrics["avg_time_ms"]
                    ret[int(batch_size)][1] += batch_metrics["std_time_ms"] ** 2

        return frozendict({k: (v[0], v[1] ** 0.5) for k, v in ret.items()})

    @lru_cache(maxsize=16384)
    def maximum_memory_usage(self, partition: "PartitionProvider.Partition") -> tuple[float, float]:
        """Estimate maximum memory usage of a partition by taking a maximum across nodes in the partition

        Args:
            partition (PartitionProvider.Partition): partition to evaluate

        Returns:
            tuple[float, float]: avg memory usage, std memory usage
        """

        max_avg_memory_bytes = 0.0
        max_std_memory_bytes = 0.0

        for subgraph in partition.subgraphs:
            for node_uid in subgraph.enclosed_region:
                node = self.torch_graph.node_from_uid(node_uid)
                node_avg_mem = node.node.meta["torch_split_profiling"]["avg_memory_bytes"]
                node_std_mem = node.node.meta["torch_split_profiling"]["std_memory_bytes"]

                max_avg_memory_bytes = max(node_avg_mem, max_avg_memory_bytes)
                max_std_memory_bytes = max(node_std_mem, max_std_memory_bytes)

        return max_avg_memory_bytes, max_std_memory_bytes

    @lru_cache(maxsize=16384)
    def _cached_network_traffic(self, node_uids: frozenset[uuid.UUID]) -> float:
        """Get cached output size for a node.

        Args:
            node_uid: Unique identifier for the node

        Returns:
            float: output size in bytes
        """
        total_size = 0.0
        for node_uid in node_uids:
            node = self.torch_graph.node_from_uid(node_uid)
            total_size += node.node.meta["torch_split_profiling"]["max_output_size_bytes"]
        return total_size

    def network_traffic(self, partition: "PartitionProvider.Partition") -> float:
        """Estimate network traffic for a partition by summing input and output sizes.

        Args:
            partition (PartitionProvider.Partition): partition to evaluate

        Returns:
            float: total network traffic in bytes
        """
        total_traffic = 0.0
        for subgraph in partition.subgraphs:
            total_traffic += self._cached_network_traffic(subgraph.inputs)
            total_traffic += self._cached_network_traffic(subgraph.outputs)

        return total_traffic

    # -----------------------------
    # Selection / ranking
    # -----------------------------

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

    def _evaluate_partition(self, partition: "PartitionProvider.Partition") -> "PartitionProvider.PartitionCandidate":
        """Evaluate a partition to compute metrics like bytes_in, bytes_out, flops, and peak_mem_bytes."""

        # TODO (jq54): implement accurate computation of these metrics from traces
        flops = 0.0

        for subgraph in partition.subgraphs:
            for _ in subgraph.enclosed_region:
                flops += 1

        return PartitionProvider.PartitionCandidate(
            partition=partition,
            execution_time_ms=self.total_execution_time(partition),
        )

    @logging.timer(name="Partition generation")
    def _generate_all_partition_candidates(
        self,
    ) -> Iterable["PartitionProvider.PartitionCandidate"]:
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
                partition = self._get_partition_from_cut(candidate_cut)

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
