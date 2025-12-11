import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional, cast

import torch
import torch._dynamo as dynamo
import torch._dynamo.exc as dynamo_exc
import torch.fx as fx
import torch.nn as nn

import torch_split.lib.assertions as assertions
import torch_split.lib.log as logging


def capture_graph(m: nn.Module) -> Callable[..., fx.GraphModule]:
    def graph_module_generator(*args, **kwargs) -> fx.GraphModule:
        try:
            dynamo.config.dynamic_shapes = False
            export_result = dynamo.export(m, tracing_mode="concrete")(*args, **kwargs)
            if isinstance(export_result, dynamo.eval_frame.ExportResult):
                gm = export_result.graph_module
                return gm
            raise ValueError("Export did not return an ExportResult")
        except dynamo_exc.TorchRuntimeError as e:
            logging.log_exception(e)
            raise RuntimeError("ensure model and example inputs are on the same device")

    return graph_module_generator


def extract_subgraph(
    gm: fx.GraphModule, n: list[fx.Node], i: Optional[list[fx.Node]], o: Optional[list[fx.Node]] = None
) -> fx.GraphModule:
    if o is None:
        o = [n[-1]]

    assertions.static_single_assignment(cast(Sequence[fx.Node], gm.graph.nodes))
    assertions.topological_order(cast(Sequence[fx.Node], gm.graph.nodes))
    assertions.subset(o, n)

    new_graph = fx.Graph()
    env: dict[str, fx.Node] = {}
    placeholder_names: list[str] = []

    def remap_args(n: fx.Node):
        if node := env.get(n.name):
            return node
        else:
            assert i is None or n in i, f"Node {n} not in inputs {i}"
            new_node = new_graph.placeholder(n.name)
            placeholder_names.append(n.name)
            env[n.name] = new_node
            return new_node

    added_output = False
    for node in n:
        if node.op == "output":
            added_output = True
        new_node = new_graph.node_copy(node, remap_args)
        env[node.name] = new_node

    if not added_output:
        new_graph.output([env[output.name] for output in o])
    new_gm = fx.GraphModule(gm, new_graph)
    new_gm.eval()
    new_gm.graph.eliminate_dead_code()
    new_gm.recompile()
    new_gm.graph.lint()
    return new_gm


def save_graph(gm: fx.GraphModule, filename: Path):
    assertions.file_extension(filename, ".pt")
    torch.save(gm, filename)


def load_graph(filename: Path) -> fx.GraphModule:
    assertions.file_extension(filename, ".pt")
    gm = torch.load(filename, weights_only=False)
    assertions.is_type(gm, fx.GraphModule)
    return gm


def hash_model_architecture(model: torch.nn.Module) -> str:
    """Generate a hash representing the architecture of a PyTorch model."""
    modules = []
    for name, module in model.named_modules():
        if name == "":
            continue
        module_info = {
            "name": name,
            "class": module.__class__.__name__,
            "params": {k: tuple(v.shape) for k, v in module._parameters.items() if v is not None},
        }
        modules.append(module_info)

    arch_str = json.dumps({"class": model.__class__.__name__, "modules": modules}, sort_keys=True)
    return hashlib.sha256(arch_str.encode()).hexdigest()
