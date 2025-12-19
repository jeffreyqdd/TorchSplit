from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional
import torch
import psutil  # type: ignore
from .core.switchboard import Switchboard
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


from opentelemetry import trace, metrics


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


def setup_tracing():
    """
    resource = Resource.create({"service.name": "torchsplit-runtime"})
    trace_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(trace_provider)

    metric_exporter = OTLPMetricExporter(endpoint="http://localhost:4318/v1/metrics")
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    meter_provider = MeterProvider(metric_readers=[metric_reader], resource=resource)
    metrics.set_meter_provider(meter_provider)
    """
    pass


setup_tracing()


class SwitchboardRuntime:
    def __init__(self, switchboard_path: Path, load_only: Optional[list[str]] = None, sampling_interval: int = 4):
        self.switchboard = Switchboard.load(switchboard_path, load_only=load_only)
        self.tracer = trace.get_tracer("ts.runtime")
        self.meter = metrics.get_meter("ts.runtime")
        self.sampling_interval = sampling_interval
        self.call_count: defaultdict[str, int] = defaultdict(lambda: 0)

        self.cpu_gauge = self.meter.create_gauge("cpu_utilization", unit="%", description="CPU usage percentage")
        self.memory_gauge = self.meter.create_gauge("memory_usage", unit="MB", description="Memory usage in MB")
        gpu_gauge = self.meter.create_gauge("gpu_utilization", unit="%", description="GPU usage percentage")

    @staticmethod
    def from_switchboard(switchboard: Switchboard) -> "SwitchboardRuntime":
        obj = SwitchboardRuntime.__new__(SwitchboardRuntime, switchboard=switchboard)

    def _map_to_position_args(self, inputs: Iterable[str], kwargs: dict[str, Any]) -> tuple[Any, ...]:
        """Map a list of input parameter names to positional args from kwargs."""
        return tuple(kwargs[name] for name in inputs)

    def call(self, component_name: str, *args, **kwargs):
        with self.tracer.start_as_current_span(f"SwitchboardRuntime.call:{component_name}") as span:
            model = self.switchboard.get_model(component_name)
            span.add_event("fetched_model")

            if kwargs:
                model_meta = self.switchboard.layout.metadata[component_name]
                args = args + self._map_to_position_args(model_meta.input_parameters, kwargs)
                span.add_event("remapped_keyword_args")

            output = model(*args)
            span.add_event("executed_model")

            input_size = _estimate_tensor_size(args) + _estimate_tensor_size(kwargs.values())
            output_size = _estimate_tensor_size(output)
            span.set_attribute("input_size_bytes", input_size)
            span.set_attribute("output_size_bytes", output_size)
            span.add_event("attributes_recorded")

            # if self._should_sample(component_name):
            #     self.cpu_gauge.set(_get_cpu_utilization())
            #     self.memory_gauge.set(_get_dram_utilization())
            #     span.add_event("metrics_recorded")

            return output

    def interpret(self, debug: bool = False, **kwargs: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Execute the switchboard by running components in topological order."""
        with self.tracer.start_as_current_span("SwitchboardRuntime.interpret") as span:
            layout = self.switchboard.layout

            # outputs[i] is a dictionary of available parameters in kwargs format to component i
            outputs: dict[str, dict[str, Any]] = defaultdict(dict)
            results: dict[str, Any] = {}
            intermediates: dict[str, dict[str, Any]] = {}

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
                    component_inputs = self._map_to_position_args(meta.input_parameters, outputs[component_name])
                    component_outputs = self.call(component_name, *component_inputs)
                    results[component_name] = component_outputs
                    intermediates[component_name] = outputs[component_name].copy()
                    outputs[component_name].clear()

                    if debug:
                        print(f"Executed component: {component_name}")
                        for i, o in zip(meta.input_parameters, component_inputs):
                            if torch.is_tensor(o):
                                print(f"  Input {i}: {type(o)}, shape={o.shape}, dtype={o.dtype}")
                            else:
                                print(f"  Input {i}: {type(o)}")
                        for o_name, o_value in zip(meta.output_parameters, component_outputs):
                            if torch.is_tensor(o_value):
                                print(
                                    f"  Output {o_name}: {type(o_value)}, shape={o_value.shape}, dtype={o_value.dtype}"
                                )
                            else:
                                print(f"  Output {o_name}: {type(o_value)}")

                    for output_name, output_value in zip(meta.output_parameters, component_outputs):
                        for downstream in downstream_nodes:
                            if output_name in map(lambda x: x[0], downstream.mapping):
                                outputs[downstream.name][output_name] = output_value

            if self._should_sample("interpret"):
                self.cpu_gauge.set(_get_cpu_utilization())
                self.memory_gauge.set(_get_dram_utilization())
                span.add_event("metrics_recorded")

            return results, intermediates

    def _should_sample(self, identifier: str) -> bool:
        self.call_count[identifier] += 1
        return (self.call_count[identifier] % self.sampling_interval) == 0
