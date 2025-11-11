import json
import time
from contextlib import contextmanager
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

    @dataclass
    class _Measurement:
        time_ms: float = 0.0
        peak_memory_bytes: int = 0

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
        device: torch.device = n.meta["torch_split_device"]

        # Choose device-specific profiling context
        if device.type == "cuda":
            ctx = self._cuda_profile
        elif device.type == "mps":
            ctx = self._mps_profile
        else:
            ctx = self._cpu_profile

        with ctx(device) as meas:
            ret = super().run_node(n)

        time_elapsed_ms = meas.time_ms
        peak_memory_used_bytes = meas.peak_memory_bytes
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

    ### device profiling context managers
    @contextmanager
    def _cuda_profile(self, device: torch.device):
        meas = RuntimeAnnotator._Measurement()
        torch.cuda.reset_max_memory_allocated(device)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        try:
            yield meas
        finally:
            end_event.record()
            torch.cuda.synchronize()
            meas.time_ms = start_event.elapsed_time(end_event)
            meas.peak_memory_bytes = torch.cuda.max_memory_allocated(device)

    @contextmanager
    def _mps_profile(self, device: torch.device):
        meas = RuntimeAnnotator._Measurement()
        torch.mps.synchronize()
        start_ns = time.perf_counter_ns()
        try:
            yield meas
        finally:
            torch.mps.synchronize()
            meas.time_ms = (time.perf_counter_ns() - start_ns) / 1e6
            # Best available metric on MPS: total driver-allocated memory
            meas.peak_memory_bytes = torch.mps.driver_allocated_memory()

    @contextmanager
    def _cpu_profile(self, device: torch.device):
        meas = RuntimeAnnotator._Measurement()
        start_ns = time.perf_counter_ns()
        try:
            yield meas
        finally:
            meas.time_ms = (time.perf_counter_ns() - start_ns) / 1e6
            meas.peak_memory_bytes = 0
