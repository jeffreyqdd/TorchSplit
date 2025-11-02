"""Profile PyTorch Graph using the Interpreter pattern."""

import time
from dataclasses import dataclass

import torch
import torch.fx as fx


def _estimate_tensor_size(obj) -> int:
    """Calculate tensor size in bytes"""
    if torch.is_tensor(obj):
        return obj.nbytes
    elif isinstance(obj, (list, tuple)):
        return sum(_estimate_tensor_size(o) for o in obj)
    elif isinstance(obj, dict):
        return sum(_estimate_tensor_size(v) for v in obj.values())
    return 0


@dataclass
class ProfilerResult:
    time_ns: list[int] = []
    peak_memory_usage_bytes: list[int] = []


class InstrumentedModel(fx.Interpreter):
    def __init__(self, gm: fx.GraphModule, device: torch.device):
        super().__init__(gm)
        self._device = device
        self._node_runtime_secs: dict[fx.Node, dict[int, ProfilerResult]] = {}
        """Map from fx.Node to a dict of batch_size and ProfilerResult"""

        if self._device.type == "cuda":
            self._profiler_reset = lambda: torch.cuda.reset_peak_memory_stats(self._device)

    def run_node(self, n: fx.Node):
        # Synchronize before timing to ensure clean start
        if self._device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self._device)
            # GPU utilization
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            time_start = time.perf_counter_ns()

        # Run the node
        ret = super().run_node(n)

        # Synchronize and measure
        if self._device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            time_elapsed_ms = start_event.elapsed_time(end_event)
            peak_memory_used_bytes = torch.cuda.max_memory_reserved(self._device)
        else:
            time_elapsed_ms = (time.perf_counter_ns() - time_start) / 1e6
            peak_memory_used_bytes = 0

        # Estimate output size
        output_size_bytes = _estimate_tensor_size(ret)

        print(
            f"Node {n.name:<30}: "
            f"time {time_elapsed_ms:.3f} ms, "
            f"peak memory {peak_memory_used_bytes / 1e6:.3f} MB, "
            f"output size {output_size_bytes / 1e6:.3f} MB"
        )

        return ret

    # def run(self, *args, **kwargs):
    #     # time execution
    #     # time_start = time.perf_counter_ns()
    #     ret = super().run(*args, **kwargs)
    #     # elapsed_time = time.perf_counter_ns() - time_start
    #     # get generated objects
    #     return ret    #     return ret
