import json
import time
import warnings
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
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

    def run(self, *args, **kwargs) -> "DeviceAnnotator":
        super().run(*args, **kwargs)
        return self

    def get_json(self):
        return json.dumps(self.device_map, indent=2)


class RuntimeAnnotator(fx.Interpreter):
    """Annotate the FX graph with runtime profiling information."""

    class Mode(Enum):
        WARMUP = 0
        LATENCY = 1
        MEMORY = 2
        NETWORK = 3

        @staticmethod
        def profiling_modes() -> Iterable["RuntimeAnnotator.Mode"]:
            return [
                RuntimeAnnotator.Mode.LATENCY,
                RuntimeAnnotator.Mode.MEMORY,
                RuntimeAnnotator.Mode.NETWORK,
            ]

    @dataclass
    class ProfileResult:
        device: torch.device
        time_ms: list[float] = field(default_factory=list)
        output_size_bytes: list[int] = field(default_factory=list)
        peak_memory_usage_bytes: list[int] = field(default_factory=list)

    @dataclass
    class Measurement:
        time_ms: float = 0.0
        peak_memory_bytes: int = 0

    def __init__(self, gm: fx.GraphModule):
        super().__init__(gm)
        self._inside_ctx = False
        self._node_statistics: dict[str, RuntimeAnnotator.ProfileResult] = {}
        self._summary: dict[str, dict[str, Any]] = {}
        self._mode = RuntimeAnnotator.Mode.WARMUP
        self._num_nodes = len(gm.graph.nodes)
        self._profilers = {
            RuntimeAnnotator.Mode.LATENCY: {
                "cuda": self._cuda_latency,
                "mps": self._mps_latency,
                "cpu": self._cpu_latency,
            },
            RuntimeAnnotator.Mode.MEMORY: {
                "cuda": self._cuda_memory,
                "mps": self._mps_memory,
                "cpu": self._cpu_memory,
            },
            RuntimeAnnotator.Mode.NETWORK: {
                "cuda": self._noop_profiler,
                "mps": self._noop_profiler,
                "cpu": self._noop_profiler,
            },
            RuntimeAnnotator.Mode.WARMUP: {
                "cuda": self._noop_profiler,
                "mps": self._noop_profiler,
                "cpu": self._noop_profiler,
            },
        }

    def run(self, *args, **kwargs):
        if not self._inside_ctx:
            raise RuntimeError("RuntimeAnnotator.run must be called inside a context manager specifying batch size")
        ret = super().run(*args, **kwargs)
        return ret

    def run_node(self, n: fx.Node):
        if self._mode == RuntimeAnnotator.Mode.WARMUP:
            return super().run_node(n)

        device: torch.device = n.meta["torch_split_device"]
        ctx = self._mode_profilers[device.type]

        with ctx(device) as measurement:
            ret = super().run_node(n)

        # get corresponding node
        self._node_statistics.setdefault(
            n.name,
            RuntimeAnnotator.ProfileResult(device=n.meta["torch_split_device"]),
        )

        profile_result = self._node_statistics[n.name]
        if self._mode == RuntimeAnnotator.Mode.NETWORK:
            profile_result.output_size_bytes.append(_estimate_tensor_size(ret))
        elif self._mode == RuntimeAnnotator.Mode.LATENCY:
            profile_result.time_ms.append(measurement.time_ms)
        elif self._mode == RuntimeAnnotator.Mode.MEMORY:
            profile_result.peak_memory_usage_bytes.append(measurement.peak_memory_bytes)

        return ret

    def get_json(self):
        return json.dumps(self._summary, indent=2)

    ### context manager to specify batch size
    def set_mode(self, mode: "RuntimeAnnotator.Mode"):
        self._mode = mode
        self._mode_profilers = self._profilers[mode]

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

    ### device profiling context managers
    @contextmanager
    def _cuda_latency(self, device: torch.device):
        measurement = RuntimeAnnotator.Measurement()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        try:
            yield measurement
        finally:
            end_event.record()
            torch.cuda.synchronize()
            measurement.time_ms = start_event.elapsed_time(end_event)

    @contextmanager
    def _cuda_memory(self, device: torch.device):
        measurement = RuntimeAnnotator.Measurement()
        torch.cuda.memory._record_memory_history(device=device, clear_history=True)
        torch.cuda.synchronize(device)
        try:
            yield measurement
        finally:
            torch.cuda.synchronize()
            device_traces: list[list[Any]] = torch.cuda.memory._snapshot(device=device)["device_traces"]
            total_memory_bytes = 0
            for trace_list in device_traces:
                for trace_entry in trace_list:
                    if trace_entry["action"] == "alloc":
                        total_memory_bytes += trace_entry["size"]
                    elif trace_entry["action"] == "free_requested":
                        total_memory_bytes -= trace_entry["size"]
                    elif trace_entry["action"] == "segment_alloc":
                        total_memory_bytes += trace_entry["size"]
                    elif trace_entry["action"] == "segment_free":
                        total_memory_bytes -= trace_entry["size"]

            measurement.peak_memory_bytes = total_memory_bytes

    @contextmanager
    def _mps_latency(self, device: torch.device):
        measurement = RuntimeAnnotator.Measurement()
        torch.mps.synchronize()
        start_ns = time.perf_counter_ns()
        try:
            yield measurement
        finally:
            torch.mps.synchronize()
            measurement.time_ms = (time.perf_counter_ns() - start_ns) / 1e6

    @contextmanager
    def _mps_memory(self, device: torch.device):
        measurement = RuntimeAnnotator.Measurement()
        torch.mps.synchronize()
        try:
            yield measurement
        finally:
            torch.mps.synchronize()
            measurement.peak_memory_bytes = torch.mps.driver_allocated_memory() // self._num_nodes
            warnings.warn("Metal memory profiling is approximate")

    @contextmanager
    def _cpu_latency(self, device: torch.device):
        measurement = RuntimeAnnotator.Measurement()
        start_ns = time.perf_counter_ns()
        try:
            yield measurement
        finally:
            measurement.time_ms = (time.perf_counter_ns() - start_ns) / 1e6

    @contextmanager
    def _cpu_memory(self, device: torch.device):
        measurement = RuntimeAnnotator.Measurement()
        try:
            yield measurement
        finally:
            measurement.peak_memory_bytes = 0
            warnings.warn("CPU memory profiling is not supported")

    @contextmanager
    def _noop_profiler(self, device: torch.device):
        measurement = RuntimeAnnotator.Measurement()
        try:
            yield measurement
        finally:
            pass
