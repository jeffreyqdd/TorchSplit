import gc
import json
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.fx as fx
from pydantic import BaseModel

from torch_split import log
from torch_split.core import assertions, utils

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


class Layout(BaseModel):
    metadata: dict[ComponentName, ComponentMetadata]
    entrypoints: list[Entrypoint]
    dfg: dict[ComponentName, list[DownstreamNode]]


@dataclass(frozen=True)
class Switchboard:
    layout: Layout
    components: dict[ComponentName, fx.GraphModule]
    _param_cache: dict[tuple[ComponentName, str], str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def get_model(self, name: ComponentName) -> fx.GraphModule:
        return self.components[name]

    def to_device(self, device: torch.device):
        """Move all components to the given device"""
        for name in self.components:
            self.components[name] = self.components[name].to(device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def eval(self):
        """Set all components to eval mode"""
        for name in self.components:
            self.components[name].eval()

    def save(self, output_path: Path):
        """Save this switchboard to the given path"""
        output_path = output_path.with_suffix(".tspartd")
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "structure.json", "w") as f:
            json.dump(self.layout.model_dump(), f, indent=2)

        for module_name, graph_module in self.components.items():
            module_path = output_path / f"{module_name}.pt"
            utils.save_graph(graph_module, module_path)

    def discard_except(self, keep: list[ComponentName]):
        """discard all components except those in 'keep'"""
        for name in list(self.components.keys()):
            if name not in keep:
                del self.components[name]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def call_component(self, name: ComponentName, *args, **kwargs):
        """Call a component by name with the given arguments."""
        if name not in self.components:
            raise ValueError(f"Component '{name}' not found. Available: {list(self.components.keys())}")

        component = self.get_model(name)
        meta = self.layout.metadata[name]
        input_names = meta.input_parameters

        # Map kwargs to positional args based on metadata
        call_args = []
        for i, p_name in enumerate(input_names):
            if i < len(args):
                call_args.append(args[i])
            elif p_name in kwargs:
                call_args.append(kwargs[p_name])
            else:
                # Try fuzzy matching for the parameter name with caching
                cache_key = (name, p_name)
                if cache_key in self._param_cache:
                    resolved_p_name = self._param_cache[cache_key]
                    if resolved_p_name in kwargs:
                        call_args.append(kwargs[resolved_p_name])
                        continue

                matches = difflib.get_close_matches(p_name, kwargs.keys(), n=1, cutoff=0.6)
                if matches:
                    resolved_p_name = matches[0]
                    logger.warning(
                        "Parameter '%s' not found for component '%s', using closest match '%s'",
                        p_name,
                        name,
                        resolved_p_name,
                    )
                    self._param_cache[cache_key] = resolved_p_name
                    call_args.append(kwargs[resolved_p_name])
                else:
                    raise ValueError(f"Component '{name}' missing required input parameter: '{p_name}'")
        outputs = component(*call_args)

        output_names = meta.output_parameters
        if len(output_names) == 1:
            return {output_names[0]: outputs}
        else:
            return {output_names[i]: outputs[i] for i in range(len(output_names))}

    @staticmethod
    def load(path: Path, load_only: Optional[list[str]] = None) -> "Switchboard":
        """load a switchboard from the given path"""
        assertions.file_extension(path, ".tspartd")

        with open(path / "structure.json", "r") as f:
            layout = Layout.model_validate(json.load(f))

        components: dict[ComponentName, fx.GraphModule] = {}
        for name, meta in layout.metadata.items():
            if load_only is not None and name not in load_only:
                continue
            assert name == meta.name, "should never fail"
            components[name] = utils.load_graph(path / f"{name}.pt")
            components[name].eval()

            # validate hashes
            component_hash = utils.hash_model_architecture(components[name])
            assert component_hash == meta.version_hash, f"hash mismatch for component {name}"

        return Switchboard(layout=layout, components=components)

    def depends_on(self, src: ComponentName) -> list[ComponentName]:
        """Return a list of components that depend on the given component"""
        dependents = []

        for node, downstreams in self.layout.dfg.items():
            for downstream in downstreams:
                if downstream.name == src:
                    dependents.append(node)

        # if src is an entrypoint, it need a special marker
        for entrypoint in self.layout.entrypoints:
            if entrypoint.name == src:
                dependents.append("<ENTRYPOINT>")

        return dependents

    def get_topological_order(self) -> list[ComponentName]:
        """Return the components in topological order"""
        visited = set()
        order = []

        def dfs(node: ComponentName):
            if node in visited:
                return
            visited.add(node)
            for downstream in self.layout.dfg.get(node, []):
                dfs(downstream.name)
            order.append(node)

        for entrypoint in self.layout.entrypoints:
            dfs(entrypoint.name)

        order.reverse()
        return order
