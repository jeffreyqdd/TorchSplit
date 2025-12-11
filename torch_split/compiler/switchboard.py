from collections import defaultdict
from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any
from rich.markup import escape

import torch
import torch.fx as fx
from pydantic import BaseModel
from dataclasses import dataclass

from torch_split.lib import assertions, log, utils

logger = log.get_logger(__name__)


ComponentName = str


class ComponentMetadata(BaseModel):
    name: ComponentName
    version_hash: str
    input_parameters: tuple[str, ...]
    output_parameters: tuple[str, ...]


class Entrypoint(BaseModel):
    name: ComponentName


class DownstreamNode(BaseModel):
    name: ComponentName
    mapping: list[tuple[str, str]]


class SwitchboardLayout(BaseModel):
    metadata: dict[ComponentName, ComponentMetadata]
    entrypoints: list[Entrypoint]
    dfg: dict[ComponentName, list[DownstreamNode]]


@dataclass(frozen=True)
class Switchboard:
    layout: SwitchboardLayout
    components: dict[ComponentName, fx.GraphModule]

    def get_model(self, name: ComponentName) -> fx.GraphModule:
        return self.components[name]

    def save(self, output_path: Path):
        """Save this switchboard to the given path"""
        output_path = output_path.with_suffix(".tspartd")
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "structure.json", "w") as f:
            json.dump(self.layout.model_dump(), f, indent=2)

        for module_name, graph_module in self.components.items():
            module_path = output_path / f"{module_name}.pt"
            utils.save_graph(graph_module, module_path)

    @staticmethod
    def load(path: Path) -> "Switchboard":
        """load a switchboard from the given path"""
        assertions.file_extension(path, ".tspartd")

        with open(path / "structure.json", "r") as f:
            layout = SwitchboardLayout.model_validate(json.load(f))

        components: dict[ComponentName, fx.GraphModule] = {}
        for name, meta in layout.metadata.items():
            assert name == meta.name, "should never fail"
            components[name] = utils.load_graph(path / f"{name}.pt")

            # validate hashes
            component_hash = utils.hash_model_architecture(components[name])
            assert component_hash == meta.version_hash, f"hash mismatch for component {name}"

        return Switchboard(layout, components)

    def interpret(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the switchboard by running components in topological order."""

        def map_to_position_args(inputs: Iterable[str], kwargs: dict[str, Any]) -> tuple[Any, ...]:
            """Map a list of input parameter names to positional args from kwargs."""
            return tuple(kwargs[name] for name in inputs)

        def to_string(param: torch.Tensor | Any) -> str:
            if isinstance(param, torch.Tensor):
                return f"tensor shape={list(param.shape)} dtype={param.dtype}"
            else:
                return type(param).__name__

        # outputs[i] is a dictionary of available parameters in kwargs format to component i
        outputs: dict[str, dict[str, Any]] = defaultdict(dict)
        results: dict[str, Any] = {}

        logger.info("starting switchboard interpretation")
        for arg_name, arg_value in kwargs.items():
            logger.info("  [green]→[/] input  [cyan]%-20s[/] (type: %s)", arg_name, to_string(arg_value))

            for entrypoint in self.layout.entrypoints:
                if arg_name in self.layout.metadata[entrypoint.name].input_parameters:
                    outputs[entrypoint.name][arg_name] = arg_value

        # iterate until no more components can be executed, or until max iterations reached
        # the idea is to "consume" arguments and put outputs into downstream components' input dicts
        changed = True
        iterations = 0
        max_iterations = len(self.layout.metadata) * 2

        while changed and iterations < max_iterations:
            changed = False

            # iterate through all models in the dfg field and see if we can run them
            for component_name, downstream_nodes in self.layout.dfg.items():
                component = self.get_model(component_name)
                meta = self.layout.metadata[component_name]

                # check if we have all inputs for this component
                if not all(i in outputs[component_name] for i in meta.input_parameters):
                    continue

                logger.info("executing component [bold blue]%s[/]", component_name)
                changed = True
                component_inputs = map_to_position_args(meta.input_parameters, outputs[component_name])
                component_outputs = component(*component_inputs)
                results[component_name] = component_outputs
                outputs[component_name].clear()

                # if len(meta.output_parameters) == 1:
                #     component_outputs = (component_outputs,)

                for input_name, input_value in zip(meta.input_parameters, component_inputs):
                    logger.info("  [green]→[/] input  [cyan]%-20s[/] (type: %s)", input_name, to_string(input_value))
                for output_name, output_value in zip(meta.output_parameters, component_outputs):
                    for downstream in downstream_nodes:
                        if output_name in map(lambda x: x[0], downstream.mapping):
                            outputs[downstream.name][output_name] = output_value
                            logger.info(
                                "  [blue]←[/] output [cyan]%-20s[/] (type: %s) (to: %s)",
                                output_name,
                                to_string(output_value),
                                downstream.name,
                            )
        return results
