import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
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


def _get_device_type(obj) -> torch.device | None:
    """calculate the device type of a tensor or nested structure of tensors"""
    if torch.is_tensor(obj):
        return obj.device
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            device = _get_device_type(o)
            if device is not None:
                return device
    elif isinstance(obj, dict):
        for v in obj.values():
            device = _get_device_type(v)
            if device is not None:
                return device

    return None


class DeviceAnnotator(fx.Interpreter):
    """Propagate device information through the FX graph. The device of each node's output is stored in node.meta['torch_split_device']."""

    def __init__(self, gm: fx.GraphModule):
        super().__init__(gm)
        self.device_map: dict[str, str] = {}

    def run_node(self, n):
        result = super().run_node(n)

        if device := _get_device_type(result):
            self.device_map[n.name] = str(device)
            n.meta["torch_split_device"] = device
        else:
            # safe to put CPU since no tensors are involved
            # and everything on the CPU can be ignored because CPU + RAM is not a resource constraint
            self.device_map[n.name] = "cpu"
            n.meta["torch_split_device"] = torch.device("cpu")

        return result

    def get_json(self):
        return json.dumps(self.device_map, indent=2)


class RuntimeAnnotator(fx.Interpreter):
    """Annotate the FX graph with runtime profiling information."""

    @dataclass
    class ProfileResult:
        device: torch.device
        time_ms: list[float] = field(default_factory=list)
        output_size_bytes: list[int] = field(default_factory=list)
        peak_memory_usage_bytes: list[int] = field(default_factory=list)

    def __init__(self, gm: fx.GraphModule):
        super().__init__(gm)
        self._inside_ctx = False
        self._node_statistics: dict[str, RuntimeAnnotator.ProfileResult] = {}
        self._summary: dict[str, dict[str, Any]] = {}

    def run(self, *args, **kwargs):
        if not self._inside_ctx:
            raise RuntimeError("RuntimeAnnotator.run must be called inside a context manager specifying batch size")
        ret = super().run(*args, **kwargs)
        return ret

    def run_node(self, n: fx.Node):
        if n.meta["torch_split_device"].type == "cuda":
            start_hook = self._cuda_start_hook
            stop_hook = self._cuda_stop_hook
        elif n.meta["torch_split_device"].type == "mps":
            start_hook = self._mps_start_hook
            stop_hook = self._mps_stop_hook
        else:
            start_hook = self._cpu_start_hook
            stop_hook = self._cpu_stop_hook

        start_hook(n.meta["torch_split_device"])

        ret = super().run_node(n)

        time_elapsed_ms, peak_memory_used_bytes = stop_hook(n.meta["torch_split_device"])
        output_size_bytes = _estimate_tensor_size(ret)

        # get corresponding node
        if n.name not in self._node_statistics:
            self._node_statistics[n.name] = RuntimeAnnotator.ProfileResult(device=n.meta["torch_split_device"])

        profiler_result = self._node_statistics[n.name]
        profiler_result.time_ms.append(time_elapsed_ms)
        profiler_result.peak_memory_usage_bytes.append(peak_memory_used_bytes)
        profiler_result.output_size_bytes.append(output_size_bytes)

        return ret

    def get_json(self):
        return json.dumps(self._summary, indent=2)

    ### context manager to specify batch size
    def __call__(self, batch_size: int):
        self._batch_size = batch_size
        return self

    def __enter__(self):
        self._inside_ctx = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._inside_ctx = False
        self._summary = {}

        for node_name, profiler_result in self._node_statistics.items():
            avg_time = sum(profiler_result.time_ms) / len(profiler_result.time_ms)
            avg_output_size = sum(profiler_result.output_size_bytes) / len(profiler_result.output_size_bytes)
            avg_peak_memory = sum(profiler_result.peak_memory_usage_bytes) / len(
                profiler_result.peak_memory_usage_bytes
            )

            # calculate std_dev
            time_std = np.std(profiler_result.time_ms)
            output_size_std = np.std(profiler_result.output_size_bytes)
            peak_memory_std = np.std(profiler_result.peak_memory_usage_bytes)

            self._summary[node_name] = {
                "avg_time_ms": avg_time,
                "avg_output_size_bytes": avg_output_size,
                "avg_peak_memory_usage_bytes": avg_peak_memory,
                "std_time_ms": float(time_std),
                "std_output_size_bytes": float(output_size_std),
                "std_peak_memory_usage_bytes": float(peak_memory_std),
                "min_time_ms": min(profiler_result.time_ms),
                "max_time_ms": max(profiler_result.time_ms),
                "min_output_size_bytes": min(profiler_result.output_size_bytes),
                "max_output_size_bytes": max(profiler_result.output_size_bytes),
                "min_peak_memory_usage_bytes": min(profiler_result.peak_memory_usage_bytes),
                "max_peak_memory_usage_bytes": max(profiler_result.peak_memory_usage_bytes),
            }

    ### device profiling hooks
    def _cuda_start_hook(self, device: torch.device):
        torch.cuda.synchronize()
        self._cuda_start_event = torch.cuda.Event(enable_timing=True)
        self._cuda_end_event = torch.cuda.Event(enable_timing=True)

        self._cuda_start_event.record()

    def _cuda_stop_hook(self, device: torch.device):
        self._cuda_end_event.record()
        torch.cuda.synchronize()

        time_elapsed_ms = self._cuda_start_event.elapsed_time(self._cuda_end_event)
        peak_memory_used_bytes = torch.cuda.max_memory_reserved(device)

        return time_elapsed_ms, peak_memory_used_bytes

    def _mps_start_hook(self, device: torch.device):
        torch.mps.synchronize()
        self._mps_start_event = torch.mps.Event(enable_timing=True)
        self._mps_end_event = torch.mps.Event(enable_timing=True)

        self._mps_start_event.record()

    def _mps_stop_hook(self, device: torch.device):
        self._mps_end_event.record()
        torch.mps.synchronize()

        time_elapsed_ms = self._mps_start_event.elapsed_time(self._mps_end_event)
        peak_memory_used_bytes = torch.mps.driver_allocated_memory()

        return time_elapsed_ms, peak_memory_used_bytes

    def _cpu_start_hook(self, device: torch.device):
        self._cpu_start_time = time.perf_counter_ns()

    def _cpu_stop_hook(self, device: torch.device):
        time_elapsed_ms = (time.perf_counter_ns() - self._cpu_start_time) / 1e6
        peak_memory_used_bytes = 0

        return time_elapsed_ms, peak_memory_used_bytes
