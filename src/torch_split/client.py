import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn


def _estimate_tensor_size(obj) -> int:
    """Calculate tensor size in bytes"""
    if torch.is_tensor(obj):
        return obj.nbytes
    elif isinstance(obj, (list, tuple)):
        return sum(_estimate_tensor_size(o) for o in obj)
    elif isinstance(obj, dict):
        return sum(_estimate_tensor_size(v) for v in obj.values())
    return 0


class InstrumentedModule(fx.Interpreter):
    """Annotate the FX graph with runtime profiling information."""

    @dataclass
    class ProfileResult:
        device: torch.device
        time_ms: list[float] = field(default_factory=list)
        output_size_bytes: list[int] = field(default_factory=list)
        peak_memory_usage_bytes: list[int] = field(default_factory=list)

    def __init__(self, gm: fx.GraphModule):
        super().__init__(gm)

        self._batch_size = 0
        self._inside_ctx = False
        self._node_statistics: dict[fx.Node, dict[int, InstrumentedModule.ProfileResult]] = {}
        self._summary = {}
        """Map from fx.Node to a dict of batch_size and ProfilerResult"""

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
        if n not in self._node_statistics:
            self._node_statistics[n] = {}

        if self._batch_size not in self._node_statistics[n]:
            self._node_statistics[n][self._batch_size] = InstrumentedModule.ProfileResult(
                device=n.meta["torch_split_device"]
            )

        profiler_result = self._node_statistics[n][self._batch_size]
        profiler_result.time_ms.append(time_elapsed_ms)
        profiler_result.peak_memory_usage_bytes.append(peak_memory_used_bytes)
        profiler_result.output_size_bytes.append(output_size_bytes)

        return ret

    def export_to_file(self, filepath: Path):
        with open(filepath, "w") as f:
            json.dump(self._summary, f, indent=2)

    ### context manager to specify batch size
    def __call__(self, batch_size: int):
        self._batch_size = batch_size
        return self

    def __enter__(self):
        self._inside_ctx = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._inside_ctx = False
        self._batch_size = 0
        self._summary = {}

        # annotate all the nodes
        assert isinstance(self.graph, fx.Graph), "Expected fx.Graph"
        for node in self.graph.nodes:
            if not isinstance(node, fx.Node):
                raise TypeError("Expected fx.Node")

            if node not in self._node_statistics:
                continue

            # create a dictionary mapping from batch size to
            # average time, average output size, average peak memory usage, standard deviation of time, standard deviation of peak memory usage, std of output size
            batch_size_stats = {}
            for batch_size, profiler_result in self._node_statistics[node].items():
                avg_time = sum(profiler_result.time_ms) / len(profiler_result.time_ms)
                avg_output_size = sum(profiler_result.output_size_bytes) / len(profiler_result.output_size_bytes)
                avg_peak_memory = sum(profiler_result.peak_memory_usage_bytes) / len(
                    profiler_result.peak_memory_usage_bytes
                )

                # calculate std_dev
                time_std = np.std(profiler_result.time_ms)
                output_size_std = np.std(profiler_result.output_size_bytes)
                peak_memory_std = np.std(profiler_result.peak_memory_usage_bytes)

                batch_size_stats[batch_size] = {
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

            self._summary[node.name] = batch_size_stats

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

    def _mps_start_hook(self, _: torch.device):
        torch.mps.synchronize()
        self._mps_start_event = torch.mps.Event(enable_timing=True)
        self._mps_end_event = torch.mps.Event(enable_timing=True)

        self._mps_start_event.record()

    def _mps_stop_hook(self, _: torch.device):
        self._mps_end_event.record()
        torch.mps.synchronize()

        time_elapsed_ms = self._mps_start_event.elapsed_time(self._mps_end_event)
        peak_memory_used_bytes = torch.mps.driver_allocated_memory()

        return time_elapsed_ms, peak_memory_used_bytes

    def _cpu_start_hook(self, _: torch.device):
        self._cpu_start_time = time.perf_counter_ns()

    def _cpu_stop_hook(self, _: torch.device):
        time_elapsed_ms = (time.perf_counter_ns() - self._cpu_start_time) / 1e6
        peak_memory_used_bytes = 0

        return time_elapsed_ms, peak_memory_used_bytes


class SplitClient(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_example_inputs(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Return example inputs for the model."""
        raise NotImplementedError("SplitClient.get_example_inputs is not implemented")

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the model."""
        raise NotImplementedError("SplitClient.get_model is not implemented")

    @abstractmethod
    def run_benchmark(self, module: InstrumentedModule):
        """Run benchmark on the provided module."""
        raise NotImplementedError("SplitClient.run_benchmark is not implemented")

    def setup_benchmark(self, im: InstrumentedModule) -> InstrumentedModule:
        self.run_benchmark(im)
        return im
