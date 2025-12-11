from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any
import torch
from .compiler.switchboard import Switchboard

from opentelemetry import trace


def _estimate_tensor_size(obj) -> int:
    """Calculate tensor size in bytes"""
    if torch.is_tensor(obj):
        return obj.nbytes
    elif isinstance(obj, (list, tuple)):
        return sum(_estimate_tensor_size(o) for o in obj)
    elif isinstance(obj, dict):
        return sum(_estimate_tensor_size(v) for v in obj.values())
    return 0


class SwitchboardRuntime:
    def __init__(self, switchboard_path: Path):
        self.switchboard = Switchboard.load(switchboard_path)
        self.tracer = trace.get_tracer("ts.runtime")

    def call(self, component_name: str, *args, **kwargs):
        with self.tracer.start_as_current_span(f"SwitchboardRuntime.call:{component_name}") as span:
            model = self.switchboard.get_model(component_name)
            span.add_event("fetched_model")

            output = model(*args, **kwargs)
            span.add_event("executed_model")

            input_size = _estimate_tensor_size(args) + _estimate_tensor_size(kwargs.values())
            output_size = _estimate_tensor_size(output)
            span.set_attribute("input_size_bytes", input_size)
            span.set_attribute("output_size_bytes", output_size)
            span.add_event("attributes_recorded")

            return output

    def interpret(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the switchboard by running components in topological order."""
        with self.tracer.start_as_current_span("SwitchboardRuntime.interpret") as span:
            layout = self.switchboard.layout

            def map_to_position_args(inputs: Iterable[str], kwargs: dict[str, Any]) -> tuple[Any, ...]:
                """Map a list of input parameter names to positional args from kwargs."""
                return tuple(kwargs[name] for name in inputs)

            # outputs[i] is a dictionary of available parameters in kwargs format to component i
            outputs: dict[str, dict[str, Any]] = defaultdict(dict)
            results: dict[str, Any] = {}

            for arg_name, arg_value in kwargs.items():
                for entrypoint in layout.entrypoints:
                    if arg_name in layout.metadata[entrypoint.name].input_parameters:
                        outputs[entrypoint.name][arg_name] = arg_value

            # iterate until no more components can be executed, or until max iterations reached
            # the idea is to "consume" arguments and put outputs into downstream components' input dicts
            changed = True
            iterations = 0
            max_iterations = len(layout.metadata) * 2

            while changed and iterations < max_iterations:
                changed = False

                # iterate through all models in the dfg field and see if we can run them
                for component_name, downstream_nodes in layout.dfg.items():
                    meta = layout.metadata[component_name]

                    # check if we have all inputs for this component
                    if not all(i in outputs[component_name] for i in meta.input_parameters):
                        continue

                    changed = True
                    component_inputs = map_to_position_args(meta.input_parameters, outputs[component_name])
                    component_outputs = self.call(component_name, *component_inputs)
                    results[component_name] = component_outputs
                    outputs[component_name].clear()

                    for output_name, output_value in zip(meta.output_parameters, component_outputs):
                        for downstream in downstream_nodes:
                            if output_name in map(lambda x: x[0], downstream.mapping):
                                outputs[downstream.name][output_name] = output_value

            return results
