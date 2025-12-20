"""This module captures the computation graph of a PyTorch model using torch.fx"""

import importlib
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import graphviz  # type: ignore
import torch.fx as fx
from frozendict import frozendict

import torch_split.core.assertions as assertions
import torch_split.log as logging

logger = logging.get_logger(__name__)

cond_mod = importlib.import_module("torch._higher_order_ops.cond")


# setattr(fx.Node, "__hash__", lambda self: id(self))
# setattr(fx.Node, "__eq__", lambda self, other: self is other)
# setattr(fx.Node, "__neq__", lambda self, other: self is not other)


def _flatten_arguments(items) -> Iterable:
    """Flatten nested arguments into a single iterable."""
    if isinstance(items, fx.Node):
        items = [items]

    for item in items:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            for sub in _flatten_arguments(item):
                yield sub
        else:
            yield item


@dataclass(frozen=True)
class ConcreteNode:
    uuid: uuid.UUID
    name: str

    @staticmethod
    def from_fx(n: fx.Node) -> "ConcreteNode":
        return ConcreteNode(uuid.uuid4(), n.name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(self.uuid)


@dataclass(frozen=True)
class TorchGraph:
    label: str
    """human readable name of this subgraph"""

    graph_module: fx.GraphModule
    """the original fx.GraphModule from which this TorchGraph was constructed"""

    execution_order: tuple[ConcreteNode, ...]
    """Original construction order of nodes (UUIDs) in the fx graph."""

    concrete_to_node: frozendict[ConcreteNode, fx.Node]
    """map from a concrete node to its fx.Node representation"""

    node_to_concrete: frozendict[fx.Node, ConcreteNode]
    """map from a fx.Node to its concrete node representation"""

    consumers: frozendict[ConcreteNode, frozenset[ConcreteNode]]
    """map from a node to the set of nodes that consume its output"""

    producers: frozendict[ConcreteNode, frozenset[ConcreteNode]]
    """map from a node to the set of nodes that produce its inputs"""

    parameters: frozendict[ConcreteNode, frozenset[str]]
    """map from a node to the set of parameter names it depends on"""

    inputs: frozenset[ConcreteNode]
    """represents the input nodes of this subgraph"""

    outputs: frozenset[ConcreteNode]
    """represents the nodes required by the output instruction"""

    @staticmethod
    def from_fx_graph(gm: fx.GraphModule, label: str = "root") -> "TorchGraph":
        """Construct a TorchGraph from a torch.fx.GraphModule."""

        assertions.static_single_assignment(cast(Sequence[fx.Node], gm.graph.nodes))
        assertions.topological_order(cast(Sequence[fx.Node], gm.graph.nodes))

        logger.info("starting graph construction for [cyan]%s[/]", label)

        fun = ConcreteNode.from_fx
        g_execution_order: tuple[ConcreteNode, ...] = tuple(map(fun, gm.graph.nodes))
        g_concrete_to_node: dict[ConcreteNode, fx.Node] = {}
        g_node_to_concrete: dict[fx.Node, ConcreteNode] = {}

        g_parameters: dict[ConcreteNode, set[str]] = defaultdict(set)

        g_consumers: defaultdict[ConcreteNode, set[ConcreteNode]] = defaultdict(set)
        g_producers: defaultdict[ConcreteNode, set[ConcreteNode]] = defaultdict(set)

        g_inputs: set[ConcreteNode] = set()
        g_outputs: set[ConcreteNode] = set()

        def remap_args(n: fx.Node) -> ConcreteNode | None:
            try:
                return g_node_to_concrete.get(n, None)
            except TypeError:
                return None

        for idx, (fx_node, concrete_node) in enumerate(zip(gm.graph.nodes, g_execution_order), 1):
            logger.debug(
                "[bold]%s[/] [%d/%d] %s → %s",
                fx_node.op,
                idx,
                len(gm.graph.nodes),
                fx_node.target,
                fx_node.name,
            )

            if fx_node.name != concrete_node.name:
                raise RuntimeError(f"fx.Node '{fx_node.name}' does not match ConcreteNode '{concrete_node.name}'")

            if not isinstance(concrete_node, ConcreteNode):
                raise TypeError("Expected ConcreteNode")

            if not isinstance(fx_node, fx.Node):
                raise TypeError("Expected fx.Node")

            g_concrete_to_node[concrete_node] = fx_node
            g_node_to_concrete[fx_node] = concrete_node
            normalized_args = fx_node.args + tuple(fx_node.kwargs.values())

            if fx_node.op == "placeholder":
                # represents a function input
                #   - name      : value to assign to
                #   - target    : the name of the input argument
                g_inputs.add(concrete_node)
                pass
            elif fx_node.op == "get_attr":
                assert not fx_node.is_impure(), "get_attr must be pure"
                g_parameters[concrete_node].add(str(fx_node.target))
            elif fx_node.op == "call_function" or fx_node.op == "call_module" or fx_node.op == "call_method":
                for arg in filter(
                    lambda n: n is not None,
                    map(remap_args, _flatten_arguments(normalized_args)),
                ):
                    # can't be None because of filter
                    assert arg is not None
                    g_producers[concrete_node].add(arg)
                    g_consumers[arg].add(concrete_node)
                    logger.debug(
                        "    [green]→[/] connecting data node [cyan]%s[/]",
                        arg.name,
                    )
            elif fx_node.op == "output":
                for arg in filter(
                    lambda n: n is not None,
                    map(remap_args, _flatten_arguments(normalized_args)),
                ):
                    # can't be None because of filter
                    assert arg is not None
                    g_producers[concrete_node].add(arg)
                    g_consumers[arg].add(concrete_node)
                    logger.debug(
                        "    [green]→[/] connecting output node [cyan]%s[/]",
                        arg.name,
                    )
                g_outputs.add(concrete_node)
            else:
                raise RuntimeError(f"unmatched op: {fx_node.op} of target type {type(fx_node.target)}")

        return TorchGraph(
            label=label,
            graph_module=gm,
            concrete_to_node=frozendict(g_concrete_to_node),
            node_to_concrete=frozendict(g_node_to_concrete),
            execution_order=g_execution_order,
            consumers=frozendict({k: frozenset(v) for k, v in g_consumers.items()}),
            producers=frozendict({k: frozenset(v) for k, v in g_producers.items()}),
            parameters=frozendict({k: frozenset(v) for k, v in g_parameters.items()}),
            inputs=frozenset(g_inputs),
            outputs=frozenset(g_outputs),
        )

    def to_fx(self, c_node: ConcreteNode) -> fx.Node:
        """Get the fx.Node corresponding to a ConcreteNode."""
        if c_node not in self.concrete_to_node:
            raise KeyError(f"Node {c_node} not found in graph")
        return self.concrete_to_node[c_node]

    def sort_execution_order(self, nodes: Iterable[ConcreteNode]) -> list[ConcreteNode]:
        """Sort a list of concrete nodes according to the original execution order."""
        order_index = {node: idx for idx, node in enumerate(self.execution_order)}
        return sorted(nodes, key=lambda n: order_index[n])

    def get_roots(self) -> frozenset[ConcreteNode]:
        """Get the root nodes of the graph (with no predecessors)."""
        roots: set[ConcreteNode] = set()
        for n in self.concrete_to_node.keys():
            if not self.get_predecessors(n):
                roots.add(n)
        return frozenset(roots)

    def get_successors(self, n: ConcreteNode) -> frozenset[ConcreteNode]:
        """Get the successor nodes of a given concrete node."""
        if n not in self.concrete_to_node:
            raise KeyError(f"Node {n} not found in graph")
        return self.consumers.get(n, frozenset())

    def get_predecessors(self, n: ConcreteNode) -> frozenset[ConcreteNode]:
        """Get the predecessor nodes of a given concrete node."""
        if n not in self.concrete_to_node:
            raise KeyError(f"Node {n} not found in graph")
        return self.producers.get(n, frozenset())

    def render_graph(self, graph: graphviz.Digraph, include_parameters: bool = False):
        """Render the graph using graphviz"""

        for concrete_node in self.concrete_to_node.keys():
            fillcolor = "orange" if concrete_node in self.outputs else "white"
            fillcolor = "lightgreen" if concrete_node in self.inputs else fillcolor
            style = "filled"
            shape = "box"
            label = concrete_node.name
            graph.node(
                str(concrete_node.uuid),
                label=label,
                shape=shape,
                style=style,
                fillcolor=fillcolor,
            )

        for src, dests in self.consumers.items():
            for dest in dests:
                graph.edge(str(src.uuid), str(dest.uuid))

        if include_parameters:
            for src, param_names in self.parameters.items():
                for param_name in param_names:
                    param_node_id = f"param_{src.uuid}_{param_name}"
                    graph.node(
                        param_node_id,
                        label=param_name,
                        shape="box",
                        style="filled",
                        fillcolor="lightblue",
                    )
                    graph.edge(param_node_id, str(src.uuid), style="dashed")

    def annotate_with_profiling_data(self, batch_size: int, profiling_data: dict, device_data: dict):
        """Annotate the graph nodes with profiling data."""
        for _, node in self.concrete_to_node.items():
            if node_data := profiling_data.get(node.name, None):
                if not hasattr(node.meta, "torch_split_profiling"):
                    node.meta["torch_split_profiling"] = {}
                node.meta["torch_split_profiling"][batch_size] = node_data
            else:
                logger.error("No profiling data found for node %s", node.name)
                raise RuntimeError(f"No profiling data for node {node.name}")
