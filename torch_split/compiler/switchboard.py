import json
from dataclasses import dataclass
from pathlib import Path

import torch.fx as fx
from pydantic import BaseModel

from torch_split.compiler import assertions, log, utils

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
            components[name].eval()
            components[name].compile()

            # validate hashes
            component_hash = utils.hash_model_architecture(components[name])
            assert component_hash == meta.version_hash, f"hash mismatch for component {name}"

        return Switchboard(layout, components)
