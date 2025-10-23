"""Enhanced tracing that instruments FX graph nodes directly for accurate partition analysis."""

import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.fx as fx

from torch_split.client import TorchSplitClient
from torch_split.core import ir
from torch_split.logging import get_logger

logger = get_logger(__name__)


def _estimate_tensor_size(obj) -> int:
    """Calculate tensor size in bytes"""
    if torch.is_tensor(obj):
        return obj.nbytes
    elif isinstance(obj, (list, tuple)):
        return sum(_estimate_tensor_size(o) for o in obj)
    elif isinstance(obj, dict):
        return sum(_estimate_tensor_size(v) for v in obj.values())
    return 0


@dataclass(frozen=True)
class RunAttribute:
    batch_size: int
    attributes: dict[str, Any]


class InstrumentedModel(fx.Interpreter):
    def __init__(self, client: TorchSplitClient):
        device = client.target_device()
        gm = ir.capture_graph(client)
        gm.to(device)

        super().__init__(gm)
        self._total_runtime_secs: list[float] = []
        self._node_runtime_secs: dict[fx.Node, float] = {}
        self._device = device

    def run(self, *args, **kwargs):
        # time execution
        # time_start = time.perf_counter_ns()
        ret = super().run(*args, **kwargs)
        # elapsed_time = time.perf_counter_ns() - time_start
        # get generated objects
        return ret

    def run_node(self, n: fx.Node):
        torch.cuda.reset_peak_memory_stats(self._device)
        time_start = time.perf_counter_ns()
        ret = super().run_node(n)

        time_elapsed_ns = time.perf_counter_ns() - time_start
        peak_memory_used_bytes = torch.cuda.memory_stats(self._device)["reserved_bytes.all.peak"]
        output_size_bytes = _estimate_tensor_size(ret)

        print(
            f"Node {n.name}: time {time_elapsed_ns / 1e6:.3f} ms, "
            f"peak memory {peak_memory_used_bytes / 1e6:.3f} MB, "
            f"output size {output_size_bytes / 1e6:.3f} MB"
        )

        return ret


class Profiler:
    def __init__(self, client: TorchSplitClient):
        self._instrumented_model = InstrumentedModel(client)
        self._run_attribute: Optional[RunAttribute] = None

    def __enter__(self, batch_size: int, **kwargs):
        self._run_attribute = RunAttribute(batch_size=batch_size, attributes=kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        self._run_attribute = None

    def profile_inference(self, *args, **kwargs):
        if not self._run_attribute:
            raise RuntimeError("InstrumentedModel.profile_inference must be called within a 'with' context")


# class FXGraphTracer:
#     """Traces FX graph execution at the node level for accurate performance analysis."""

#     def __init__(self, graph_module: fx.GraphModule, node_uuid_mapping: Dict[str, uuid.UUID]):
#         """
#         Args:
#             graph_module: The FX GraphModule to trace
#             node_uuid_mapping: Mapping from FX node names to UUIDs used in TorchGraph
#         """
#         self.graph_module = graph_module
#         self.node_uuid_mapping = node_uuid_mapping
#         self.trace_data: Dict[uuid.UUID, NodeTraceData] = {}
#         self._tracer: trace.Tracer | None = None
#         self._setup_tracing()

#     def _setup_tracing(self):
#         """Setup OpenTelemetry tracing"""
#         resource = Resource.create({"service.name": "fx_graph_tracer"})
#         trace.set_tracer_provider(TracerProvider(resource=resource))
#         self._tracer = trace.get_tracer(__name__)

#     def add_exporter(self, exporter_type: str, **kwargs):
#         """Add trace exporters"""
#         if exporter_type == "otlp" and "endpoint" in kwargs:
#             otlp_exporter = OTLPSpanExporter(endpoint=kwargs["endpoint"])
#             trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))  # type: ignore
#         elif exporter_type == "file" and "path" in kwargs:
#             file_exporter = MsgpackFileTraceExporter(Path(kwargs["path"]))
#             trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(file_exporter))  # type: ignore


#     def _create_pre_hook(self, original_node: fx.Node):
#         """Create a pre-execution hook for a node"""

#         def pre_hook():
#             node_uuid = self.node_uuid_mapping.get(original_node.name)
#             if node_uuid and self._tracer:
#                 span = self._tracer.start_span(f"{original_node.op}_{original_node.name}", start_time=time.time_ns())
#                 span.set_attribute("node.uuid", str(node_uuid))
#                 span.set_attribute("node.name", original_node.name)
#                 span.set_attribute("node.op", original_node.op)
#                 span.set_attribute("node.target", str(original_node.target))

#                 # Store span for post-hook
#                 if not hasattr(self, "_active_spans"):
#                     self._active_spans = {}
#                 self._active_spans[original_node.name] = span

#         return pre_hook

#     def _create_post_hook(self, original_node: fx.Node):
#         """Create a post-execution hook for a node"""

#         def post_hook(output):
#             if hasattr(self, "_active_spans") and original_node.name in self._active_spans:
#                 span = self._active_spans.pop(original_node.name)

#                 # Calculate output size
#                 output_size = self._tensor_size(output)
#                 span.set_attribute("output.size.bytes", output_size)

#                 # End span and record trace data
#                 end_time = time.time_ns()
#                 span.end(end_time=end_time)

#                 # Store trace data for later use
#                 node_uuid = self.node_uuid_mapping.get(original_node.name)
#                 if node_uuid:
#                     start_time = span.start_time if hasattr(span, "start_time") else end_time
#                     duration = end_time - start_time

#                     self.trace_data[node_uuid] = NodeTraceData(
#                         node_uuid=node_uuid,
#                         node_name=original_node.name,
#                         node_op=original_node.op,
#                         node_target=str(original_node.target),
#                         output_size_bytes=output_size,
#                         duration_ns=duration,
#                         flops=self._estimate_flops(original_node, output_size),
#                     )

#             return output

#         return post_hook

#     @staticmethod
#     def get_trace_data(self) -> Dict[uuid.UUID, NodeTraceData]:
#         """Get collected trace data"""
#         return self.trace_data.copy()
