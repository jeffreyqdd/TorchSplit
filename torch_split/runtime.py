from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional
import torch
import psutil  # type: ignore
from .core.switchboard import Switchboard
from opentelemetry import metrics, trace


def _estimate_tensor_size(obj) -> int:
    """Calculate tensor size in bytes"""
    if torch.is_tensor(obj):
        return obj.nbytes
    elif isinstance(obj, (list, tuple)):
        return sum(_estimate_tensor_size(o) for o in obj)
    elif isinstance(obj, dict):
        return sum(_estimate_tensor_size(v) for v in obj.values())
    return 0


def _get_gpu_utilization_for_model(model) -> int:
    return 0


def _get_cpu_utilization() -> float:
    return psutil.cpu_percent()


def _get_dram_utilization() -> int:
    return psutil.virtual_memory().used // (1024 * 1024)


def _normalize_call_inputs(value):
    if isinstance(value, Mapping):
        return (), value
    elif isinstance(value, (tuple, list)):
        return tuple(value), {}
    else:  # singular value
        return (value,), {}


class SwitchboardRuntime:
    def __init__(self, switchboard: Switchboard, sampling_interval: int = 4, debug: bool = False):
        self.switchboard = switchboard
        self.tracer = trace.get_tracer("ts.runtime")
        self.meter = metrics.get_meter("ts.runtime")
        self.sampling_interval = sampling_interval
        self.call_count: defaultdict[str, int] = defaultdict(lambda: 0)
        self.debug = debug

        # self.cpu_gauge = self.meter.create_gauge("cpu_utilization", unit="%", description="CPU usage percentage")
        # self.memory_gauge = self.meter.create_gauge("memory_usage", unit="MB", description="Memory usage in MB")
        # gpu_gauge = self.meter.create_gauge("gpu_utilization", unit="%", description="GPU usage percentage")

    @staticmethod
    def from_path(
        switchboard_path: Path, load_only: Optional[list[str]] = None, sampling_interval: int = 4, debug: bool = False
    ) -> "SwitchboardRuntime":
        sb = Switchboard.load(switchboard_path, load_only=load_only)
        return SwitchboardRuntime(sb, sampling_interval=sampling_interval, debug=debug)

    def call(self, component_name: str, *args, **kwargs):
        with self.tracer.start_as_current_span(f"SwitchboardRuntime.call:{component_name}") as span:
            output = self.switchboard.call_component(component_name, *args, **kwargs)
            span.add_event("call.executed_model")

            input_size = _estimate_tensor_size(args) + _estimate_tensor_size(kwargs.values())
            output_size = _estimate_tensor_size(output)
            span.set_attribute("call.input_size_bytes", input_size)
            span.set_attribute("call.output_size_bytes", output_size)
            span.add_event("call.attributes_recorded")

            return output

    def interpret(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the switchboard by running components in topological order."""

        data = {}
        data["<ENTRYPOINT>"] = kwargs

        with self.tracer.start_as_current_span("SwitchboardRuntime.interpret") as span:
            topological_order = self.switchboard.get_topological_order()

            for component_name in topological_order:
                if self.debug:
                    print(f"Executing component: {component_name}")
                    print(f"Inputs: {self.switchboard.layout.metadata[component_name].input_parameters}")
                    print(f"Outputs: {self.switchboard.layout.metadata[component_name].output_parameters}")
                dependencies = self.switchboard.depends_on(component_name)

                args_list = []
                kwargs_dict = {}
                for dep in dependencies:
                    dep_args, dep_kwargs = _normalize_call_inputs(data[dep])
                    args_list.extend(dep_args)
                    kwargs_dict.update(dep_kwargs)

                span.add_event("interpret.unpacked_inputs")
                data[component_name] = self.call(component_name, *args_list, **kwargs_dict)

        return data

    def _should_sample(self, identifier: str) -> bool:
        self.call_count[identifier] += 1
        return (self.call_count[identifier] % self.sampling_interval) == 0
