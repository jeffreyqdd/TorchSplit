"""This module captures the computation graph of a PyTorch model using torch.fx"""

import importlib
import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import graphviz  # type: ignore
import torch
import torch.fx as fx
from frozendict import frozendict

import torch_split.logging as logging
from torch_split.client import SplitClient
from torch_split.utils import capture_graph

logger = logging.get_logger(__name__)

cond_mod = importlib.import_module("torch._higher_order_ops.cond")


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


def _to_string(module) -> str:
    if isinstance(module, str):
        return module
    else:
        return module.__module__ + "." + module.__name__


@dataclass(frozen=True)
class TorchGraph:
    @dataclass(frozen=True)
    class Node:
        node: fx.Node
        is_phi: bool = field(default=False)

    @dataclass(frozen=True)
    class Parameter:
        name: str
        parameter: torch.Tensor

    label: str
    """human readable name of this subgraph"""

    trace: fx.GraphModule
    """the original fx.GraphModule from which this TorchGraph was constructed"""

    parameters: frozendict[uuid.UUID, "Parameter | TorchGraph"]
    """key is the variable to assign to and the value is the parameter object"""

    nodes: frozendict[uuid.UUID, Node]
    """mapping of node UUIDs to nodes"""

    node_dataflow: frozendict[uuid.UUID, frozenset[uuid.UUID]]
    """represents the dataflow edges where keys are variables, and values are variables that use the key's data"""

    reverse_node_dataflow: frozendict[uuid.UUID, frozenset[uuid.UUID]]
    """inverse of node_dataflow where keys are variables, and values are variables that produce data for the key"""

    parameter_dataflow: frozendict[uuid.UUID, frozenset[uuid.UUID]]
    """represents the dataflow edges where keys are parameter names and values are sets of destination nodes"""

    inputs: frozenset[uuid.UUID]
    """represents the input nodes of this subgraph"""

    outputs: frozenset[uuid.UUID]
    """represents the nodes required by the output node"""

    return_node: uuid.UUID
    """the output node of this subgraph"""

    @staticmethod
    def name_from_node(node: "TorchGraph.Node | TorchGraph") -> str:
        if isinstance(node, TorchGraph):
            return f"subgraph_{node.label}"
        else:
            return node.node.name

    @staticmethod
    def from_client(client: SplitClient, label: str = "root") -> "TorchGraph":
        return TorchGraph.from_fx_graph(capture_graph(client), label)

    def annotate_with_profiling_data(self, batch_size: int, profiling_data: dict):
        """Annotate the graph nodes with profiling data."""
        for _, node in self.nodes.items():
            if node_data := profiling_data.get(node.node.name, None):
                if not hasattr(node.node.meta, "torch_split_profiling"):
                    node.node.meta["torch_split_profiling"] = {}
                node.node.meta["torch_split_profiling"][batch_size] = node_data
            else:
                logger.error("No profiling data found for node %s", node.node.name)
                raise RuntimeError(f"No profiling data for node {node.node.name}")

    @staticmethod
    def from_fx_graph(gm: fx.GraphModule, label: str = "root") -> "TorchGraph":
        logger.info("[bold blue]Starting graph construction[/] for [cyan]%s[/]", label)
        graph = gm.graph
        # --- basic containers ---
        graph_nodes: dict[uuid.UUID, TorchGraph.Node] = {}
        graph_node_uid: dict[str, uuid.UUID] = {}
        graph_parameters: dict[uuid.UUID, TorchGraph.Parameter | TorchGraph] = {}
        graph_parameter_uid: dict[str, uuid.UUID] = {}

        # --- dataflow edges ---
        graph_node_dataflow: defaultdict[uuid.UUID, set[uuid.UUID]] = defaultdict(set)
        graph_reverse_node_dataflow: defaultdict[uuid.UUID, set[uuid.UUID]] = defaultdict(set)
        graph_parameter_dataflow: defaultdict[uuid.UUID, set[uuid.UUID]] = defaultdict(set)

        # --- inputs and outputs ---
        graph_inputs: set[uuid.UUID] = set()
        graph_outputs: set[uuid.UUID] = set()
        graph_return_node: uuid.UUID | None = None

        logger.debug("[dim]Graph has %d nodes to process[/]", len(graph.nodes))

        for node_idx, node in enumerate(graph.nodes, 1):
            if not isinstance(node, fx.Node):
                raise TypeError("Expected fx.Node")
            if node.op == "placeholder":
                # represents a function input
                #   - name      : value to assign to
                #   - target    : the name of the input argument

                logger.debug(
                    "[bold green]PLACEHOLDER[/] [%d/%d] %s → [dim]%s[/]",
                    node_idx,
                    len(graph.nodes),
                    node.name,
                    node.target,
                )

                current_node_uid = uuid.uuid4()
                graph_node_uid[node.name] = current_node_uid
                graph_nodes[current_node_uid] = TorchGraph.Node(node)
                graph_inputs.add(current_node_uid)

                logger.debug(
                    "  [dim]Added input node with UUID: %s[/]",
                    str(current_node_uid)[:8],
                )

            elif node.op == "get_attr":
                assert not node.is_impure(), "get_attr must be pure"
                assert isinstance(node.target, str)

                attribute = getattr(gm, node.target)
                if isinstance(attribute, fx.GraphModule):
                    # node.name is the variable to assign to
                    # node.target is the name of the subgraph attribute
                    logger.debug(
                        "[bold yellow]GET_ATTR[/] [%d/%d] subgraph %s → %s",
                        node_idx,
                        len(graph.nodes),
                        node.target,
                        node.name,
                    )

                    subgraph_node = TorchGraph.from_fx_graph(attribute, node.target)

                    new_uuid = uuid.uuid4()
                    graph_parameter_uid[node.name] = new_uuid
                    graph_parameters[new_uuid] = subgraph_node
                    logger.debug("  [dim]Added parameter with UUID: %s[/]", str(new_uuid)[:8])

                elif isinstance(attribute, torch.Tensor):
                    # node.name is the variable to assign to
                    # node.target is the name of the parameter attribute
                    logger.debug(
                        "[bold yellow]GET_ATTR[/] [%d/%d] parameter %s → %s",
                        node_idx,
                        len(graph.nodes),
                        node.target,
                        node.name,
                    )
                    new_uuid = uuid.uuid4()
                    graph_parameter_uid[node.name] = new_uuid
                    graph_parameters[new_uuid] = TorchGraph.Parameter(node.target, attribute)
                    logger.debug("  [dim]Added parameter with UUID: %s[/]", str(new_uuid)[:8])
                else:
                    raise TypeError(f"unhandled attribute type: {type(attribute)}")
            elif node.op == "call_function" and isinstance(node.target, cond_mod.CondOp):
                # only look at args[0] which contains data dependencies; the other should be handled by get_attr
                logger.debug(
                    "[bold blue]CALL_FUNCTION[/] [%d/%d] cond → %s",
                    node_idx,
                    len(graph.nodes),
                    node.name,
                )
                current_node_uid = uuid.uuid4()
                graph_node_uid[node.name] = current_node_uid
                graph_nodes[current_node_uid] = TorchGraph.Node(node, is_phi=True)
                for data_source in _flatten_arguments(node.args):
                    if not isinstance(data_source, fx.Node):
                        logger.debug("  [dim]ignoring non-node argument: %s[/]", data_source)
                        continue
                    if source_node_uid := graph_node_uid.get(data_source.name, None):
                        # source (node) => dest (call function node)
                        logger.debug(
                            "    [green]→[/] connecting data node [cyan]%s[/]",
                            data_source.name,
                        )
                        graph_node_dataflow[source_node_uid].add(current_node_uid)
                        graph_reverse_node_dataflow[current_node_uid].add(source_node_uid)
                    elif source_parameter_uid := graph_parameter_uid.get(data_source.name, None):
                        # source (parameter) => dest (call function node)
                        logger.debug(
                            "    [blue]→[/] connecting parameter [cyan]%s[/]",
                            data_source.name,
                        )
                        source_parameter_uid = graph_parameter_uid[data_source.name]
                        graph_parameter_dataflow[source_parameter_uid].add(current_node_uid)
                    else:
                        raise RuntimeError(f"data source {data_source.name} not found")
            elif node.op == "call_function":
                # a call function applies a free function to some values
                #   - name      : value to assign to
                #   - target    : the applied function
                #   - args      : positional arguments to the function
                #   - kwargs    : keyword arguments to the function
                # use the args and kwargs to build dataflow edges
                logger.debug(
                    "[bold blue]CALL_FUNCTION[/] [%d/%d] %s → %s",
                    node_idx,
                    len(graph.nodes),
                    _to_string(node.target),
                    node.name,
                )

                current_node_uid = uuid.uuid4()
                graph_node_uid[node.name] = current_node_uid
                graph_nodes[current_node_uid] = TorchGraph.Node(node)

                for data_source in _flatten_arguments(node.args + tuple(node.kwargs.values())):
                    if not isinstance(data_source, fx.Node):
                        logger.debug("  [dim]ignoring non-node argument: %s[/]", data_source)
                        continue

                    if source_node_uid := graph_node_uid.get(data_source.name, None):
                        # source (node) => dest (call function node)
                        logger.debug(
                            "    [green]→[/] connecting data node [cyan]%s[/]",
                            data_source.name,
                        )
                        graph_node_dataflow[source_node_uid].add(current_node_uid)
                        graph_reverse_node_dataflow[current_node_uid].add(source_node_uid)
                    elif source_parameter_uid := graph_parameter_uid.get(data_source.name, None):
                        # source (parameter) => dest (call function node)
                        logger.debug(
                            "    [blue]→[/] connecting parameter [cyan]%s[/]",
                            data_source.name,
                        )
                        source_parameter_uid = graph_parameter_uid[data_source.name]
                        graph_parameter_dataflow[source_parameter_uid].add(current_node_uid)
                    else:
                        raise RuntimeError(f"data source {data_source.name} not found")

            elif node.op == "call_module":
                # a call_module applies a module's forward() method to given arguments
                #   - name      : value to assign to
                #   - target    : the fully-qualified name of the module in the module hierarchy to
                #   - args      : positional arguments to invoke the module on, excluding the self argument
                #   - kwargs    : keyword arguments to invoke the module on, excluding the self argument
                # use the args and kwargs to build dataflow edges
                current_node_uid = uuid.uuid4()
                graph_node_uid[node.name] = current_node_uid
                graph_nodes[current_node_uid] = TorchGraph.Node(node)

                # TODO (jq54) make more rigorous for torch split hints
                # sm = gm.get_submodule(node.target)
                # if hasattr(sm, "_hello_world"):
                #     logger.error("Node %s has dynamo hint: %s", node.name, getattr(sm, "_hello_world"))

                logger.debug(
                    "[bold magenta]CALL_MODULE[/] [%d/%d] %s → %s",
                    node_idx,
                    len(graph.nodes),
                    node.target,
                    node.name,
                )
                for data_source in _flatten_arguments(node.args + tuple(node.kwargs.values())):
                    if not isinstance(data_source, fx.Node):
                        # there is no data dependency on this argument
                        continue
                    if source_node_uid := graph_node_uid.get(data_source.name, None):
                        # source (node) => dest (call module node)
                        logger.debug(
                            "    [green]→[/] connecting data node [cyan]%s[/]",
                            data_source.name,
                        )
                        graph_node_dataflow[source_node_uid].add(current_node_uid)
                        graph_reverse_node_dataflow[current_node_uid].add(source_node_uid)
                    elif source_parameter_uid := graph_parameter_uid.get(data_source.name, None):
                        # source (parameter) => dest (call module node)
                        logger.debug(
                            "    [blue]→[/] connecting parameter [cyan]%s[/]",
                            data_source.name,
                        )
                        source_parameter_uid = graph_parameter_uid[data_source.name]
                        graph_parameter_dataflow[source_parameter_uid].add(current_node_uid)
                    else:
                        raise RuntimeError(f"data source {data_source.name} not found")
            elif node.op == "call_method":
                # a call_method applies a method of an object to given arguments
                #   - name      : value to assign to
                #   - target    : the name of the method to call
                #   - args      : positional arguments to the method, including the self argument
                #   - kwargs    : keyword arguments to the method
                # print(node.name, node.target, node.args, node.type)
                assert not node.is_impure(), "get_attr must be pure"
                logger.debug(
                    "[bold cyan]CALL_METHOD[/] [%d/%d] %s → %s",
                    node_idx,
                    len(graph.nodes),
                    node.target,
                    node.name,
                )
                current_node_uid = uuid.uuid4()
                graph_node_uid[node.name] = current_node_uid
                graph_nodes[current_node_uid] = TorchGraph.Node(node)
                for data_source in _flatten_arguments(node.args + tuple(node.kwargs.values())):
                    if not isinstance(data_source, fx.Node):
                        # there is no data dependency on this argument
                        continue
                    if source_node_uid := graph_node_uid.get(data_source.name, None):
                        # source (node) => dest (call method node)
                        logger.debug(
                            "    [green]→[/] connecting data node [cyan]%s[/]",
                            data_source.name,
                        )
                        graph_node_dataflow[source_node_uid].add(current_node_uid)
                        graph_reverse_node_dataflow[current_node_uid].add(source_node_uid)
                    elif source_parameter_uid := graph_parameter_uid.get(data_source.name, None):
                        # source (parameter) => dest (call method node)
                        logger.debug(
                            "    [blue]→[/] connecting parameter [cyan]%s[/]",
                            data_source.name,
                        )
                        source_parameter_uid = graph_parameter_uid[data_source.name]
                        graph_parameter_dataflow[source_parameter_uid].add(current_node_uid)
                    else:
                        raise RuntimeError(f"data source {data_source.name} not found")
            elif node.op == "output":
                # contains the output in arg[0] attribute
                logger.debug(
                    "[bold white]OUTPUT[/] [%d/%d] → %s",
                    node_idx,
                    len(graph.nodes),
                    node.name,
                )
                current_node_uid = uuid.uuid4()
                graph_node_uid[node.name] = current_node_uid
                graph_nodes[current_node_uid] = TorchGraph.Node(node)
                for output_source in _flatten_arguments(node.args[0]):
                    assert isinstance(output_source, fx.Node)
                    if source_node_uid := graph_node_uid.get(output_source.name, None):
                        # logger.debug("    [green]→[/] connecting output [cyan]%s[/]", output_source.name)
                        graph_node_dataflow[source_node_uid].add(current_node_uid)
                        graph_reverse_node_dataflow[current_node_uid].add(source_node_uid)
                        graph_outputs.add(source_node_uid)
                        graph_return_node = current_node_uid
                    else:
                        raise RuntimeError(f"data source {output_source.name} not found")
            else:
                raise RuntimeError(f"unmatched op: {node.op} of target type {type(node.target)}")

        logger.info("[bold blue]Graph construction complete[/] for [cyan]%s[/]", label)
        logger.debug(
            "[dim]Summary: %d nodes, %d parameters, %d inputs, %d outputs[/]",
            len(graph_nodes),
            len(graph_parameters),
            len(graph_inputs),
            len(graph_outputs),
        )
        if graph_return_node is None:
            raise RuntimeError("Graph is missing return node")

        return TorchGraph(
            label=label,
            trace=gm,
            parameters=frozendict(graph_parameters),
            nodes=frozendict(graph_nodes),
            node_dataflow=frozendict({k: frozenset(v) for k, v in graph_node_dataflow.items()}),
            reverse_node_dataflow=frozendict({k: frozenset(v) for k, v in graph_reverse_node_dataflow.items()}),
            parameter_dataflow=frozendict({k: frozenset(v) for k, v in graph_parameter_dataflow.items()}),
            inputs=frozenset(graph_inputs),
            outputs=frozenset(graph_outputs),
            return_node=graph_return_node,
        )

    def get_successors(self, uid: uuid.UUID) -> frozenset[uuid.UUID]:
        """Get the successor nodes of a given node UID."""
        return self.node_dataflow.get(uid, frozenset())

    def get_predecessors(self, uid: uuid.UUID) -> frozenset[uuid.UUID]:
        """Get the predecessor nodes of a given node UID."""
        return self.reverse_node_dataflow.get(uid, frozenset())

    def name_from_uid(self, uid: uuid.UUID) -> str:
        return TorchGraph.name_from_node(self.nodes[uid])

    def node_from_uid(self, uid: uuid.UUID) -> "TorchGraph.Node":
        return self.nodes[uid]

    def render_graph(self, graph: graphviz.Digraph, include_parameters: bool = False):
        """Render the graph using graphviz"""

        for uid, node in self.nodes.items():
            fillcolor = "lightgrey" if uid in self.outputs else "white"
            fillcolor = "lightgreen" if uid in self.inputs else fillcolor
            fillcolor = "orange" if uid == self.return_node else fillcolor
            style = "filled"
            shape = "box"
            label = node.node.name

            graph.node(str(uid), label=label, shape=shape, style=style, fillcolor=fillcolor)

        if include_parameters:
            for uid, parameter in self.parameters.items():
                if isinstance(parameter, TorchGraph):
                    subgraph = graph.subgraph(name=f"cluster_{parameter.label}")
                    if subgraph is None:
                        raise RuntimeError(f"Failed to create subgraph for node {uid}")

                    with subgraph as sg:
                        sg.attr("graph", label=parameter.label, style="rounded")
                        parameter.render_graph(sg, include_parameters)

                    fillcolor = "lightblue"
                    style = "filled"
                    shape = "box"
                    label = parameter.label
                    graph.node(
                        str(uid),
                        label=label,
                        shape=shape,
                        style=style,
                        fillcolor=fillcolor,
                    )
                else:
                    fillcolor = "lightblue"
                    style = "filled"
                    shape = "box"
                    label = parameter.name
                    graph.node(
                        str(uid),
                        label=label,
                        shape=shape,
                        style=style,
                        fillcolor=fillcolor,
                    )

        for src, dests in self.node_dataflow.items():
            for dest in dests:
                graph.edge(str(src), str(dest))

        if include_parameters:
            for src, dests in self.parameter_dataflow.items():
                for dest in dests:
                    graph.edge(str(src), str(dest), style="dashed")

    def _get_execution_order(self) -> list[uuid.UUID]:
        """Get a possible execution order of the nodes using a topological sort."""
        in_degree: dict[uuid.UUID, int] = {node_uid: 0 for node_uid in self.nodes.keys()}

        for _, dests in self.node_dataflow.items():
            for dest in dests:
                in_degree[dest] += 1

        zero_in_degree = [node_uid for node_uid, degree in in_degree.items() if degree == 0]
        execution_order: list[uuid.UUID] = []

        while zero_in_degree:
            current = zero_in_degree.pop()
            execution_order.append(current)

            for neighbor in self.node_dataflow.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in_degree.append(neighbor)

        if len(execution_order) != len(self.nodes):
            raise RuntimeError("Graph has at least one cycle; topological sort not possible")

        return execution_order

    # def test(self):
    #     g = fx.Graph()
    #     env = {}
    #     execution_order = self._get_execution_order()

    #     def remap(x):
    #         # Replace references to old nodes with the corresponding new nodes
    #         return fx.map_arg(x, lambda n: env.get(n, n))

    #     for node_uid in execution_order:
    #         old = self.nodes[node_uid].node
    #         assert isinstance(old, fx.Node)

    #         # RuntimeError: Argument 'l__self___core_logit_scale' of Node 'exp' does not belong to this Graph, but was used as an argument!
    #         # If you are copying nodes from another graph, make sure to use ``arg_transform`` on node_copy() to remap values
    #         # graph():
    #         if old.op == "placeholder":
    #             # Use the placeholder's target (its name) so args align with specs
    #             new = g.placeholder(old.target)
    #         elif old.op == "get_attr":
    #             new = g.get_attr(old.target)
    #         elif old.op == "call_function":
    #             new = g.call_function(old.target, remap(old.args), remap(old.kwargs))
    #         elif old.op == "call_module":
    #             new = g.call_module(old.target, remap(old.args), remap(old.kwargs))
    #         elif old.op == "call_method":
    #             new = g.call_method(old.target, remap(old.args), remap(old.kwargs))
    #         elif old.op == "output":
    #             # In FX, output always has a single positional arg that can be any pytree.
    #             # Do NOT wrap it in an extra tuple; remap its contents structurally.
    #             assert len(old.args) == 1
    #             val = remap(old.args[0])

    #             # --- Force tensor return: unwrap singleton list/tuple ---
    #             if isinstance(val, (list, tuple)) and len(val) == 1 and isinstance(val[0], fx.Node):
    #                 val = val[0]

    #             g.output(val)
    #             break  # ensure nothing comes after output
    #         else:
    #             raise RuntimeError(f"Unsupported op {old.op}")

    #         env[old] = new

    #     # Build GraphModule; make sure it can resolve any call_module targets
    #     gm = fx.GraphModule(self.trace, g)

    #     # Preserve input/output pytrees so codegen matches the original signature/result
    #     if hasattr(self.trace, "_in_spec"):
    #         gm._in_spec = self.trace._in_spec
    #     if hasattr(self.trace, "_out_spec"):
    #         gm._out_spec = self.trace._out_spec

    #     # Optional: prune any unreachable nodes and recompile
    #     gm.graph.eliminate_dead_code()
    #     gm.recompile()

    #     print(gm.code)
    #     return gm
