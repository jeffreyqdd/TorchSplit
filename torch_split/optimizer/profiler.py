import gc
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

import psutil
import pynvml
import torch

import torch_split.log as logging
from torch_split.core import PartitionTemplate, SplitClient, Switchboard, batch_compiler, get_partition_template
from torch_split.core.switchboard import ComponentName

logger = logging.get_logger(__name__)

NVML_INITIALIZED = False
if not NVML_INITIALIZED and torch.cuda.is_available():
    pynvml.nvmlInit()
    NVML_INITIALIZED = True


def _is_ram_backed(path: Path):
    RAM_FS_TYPES = {"tmpfs", "ramfs"}
    path = path.resolve()
    best_match = None
    for part in psutil.disk_partitions(all=True):
        mount = Path(part.mountpoint).resolve()
        if path == mount or mount in path.parents:
            if best_match is None or len(str(mount)) > len(str(best_match.mountpoint)):
                best_match = part
    if best_match is None:
        raise RuntimeError(f"No mount point found for {path}")
    return best_match.fstype.lower() in RAM_FS_TYPES


def _reached_soft_dram_limit(percent: float) -> bool:
    return psutil.virtual_memory().percent >= percent * 100


def _normalize_call_inputs(value):
    if isinstance(value, Mapping):
        return (), value
    elif isinstance(value, (tuple, list)):
        return tuple(value), {}
    else:  # singular value
        return (value,), {}


def _apply_recursive(obj: Any, fn: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _apply_recursive(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        res = [_apply_recursive(v, fn) for v in obj]
        return tuple(res) if isinstance(obj, tuple) else res
    return fn(obj)


def _move_to_device(obj: Any, device: torch.device, **kwargs) -> Any:
    return _apply_recursive(obj, lambda x: x.to(device, **kwargs) if hasattr(x, "to") else x)


def _pin_memory(obj: Any) -> Any:
    return _apply_recursive(obj, lambda x: x.pin_memory() if hasattr(x, "pin_memory") else x)


class Profiler:
    def __init__(
        self,
        client_factory: Callable[[], SplitClient],
        artifacts_dir: Path,
        max_batch_size: int = 512,
        cache_dir: Optional[Path] = None,
        dram_limit_percent: float = 0.1,
    ):
        self.read_only_client: SplitClient = client_factory()
        self.client_factory = client_factory
        self.artifacts_dir = artifacts_dir
        self.dram_limit_percent = dram_limit_percent
        self.switchboard_cache: dict[int, dict[str, Switchboard]] = {}
        self.max_batch_size = max_batch_size

        if cache_dir is not None:
            self.cache_dir = cache_dir
        elif sys.platform == "linux":
            self.cache_dir = Path("/dev/shm/torchsplit_profiler_cache")
        else:
            self.cache_dir = artifacts_dir / "profiler_cache"

        # map batch size to component name to list of outputs from that component
        self._data: defaultdict[int, defaultdict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))

        if not _is_ram_backed(self.cache_dir):
            logger.warning("Cache directory (%s) is not on a RAM-backed filesystem. ", str(self.cache_dir))

        with logging.suppress_logs():
            self.template = get_partition_template(self.client_factory())

    def _load_from_cache_or_generate_to_cpu(
        self, batch_size: int, load_only: Optional[list[str]] = None
    ) -> Switchboard:
        device_memory_fractions = {
            i: torch.cuda.get_per_process_memory_fraction(device=i) for i in range(torch.cuda.device_count())
        }

        try:
            # set memory fraction to 100%, so that generation runs without OOM
            # but make sure to reset back to original values later
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(1.0, device=i)

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / f"switchboard_bs{batch_size}_{self.read_only_client.get_name()}.tspartd"

            if cache_path.exists():
                switchboard = Switchboard.load(cache_path)
            else:
                with logging.suppress_logs():
                    switchboard = batch_compiler(self.client_factory(), self.template, batch_size)
                    switchboard.save(cache_path)

            if load_only is not None:
                switchboard.discard_except(load_only)
            return switchboard

        finally:
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(device_memory_fractions[i], device=i)

    def _initialize_inputs(self, batch_sizes: list[int], num_iterations: int) -> bool:
        """initializes the 'inputs' key in self._data for each batch size and moves the tensors to RAM"""
        total_steps = len(batch_sizes) * num_iterations
        with logging.progress_bar("Initializing Inputs", total=total_steps, transient=True) as (progress, task_id):
            for bs in batch_sizes:
                dl = iter(self.read_only_client.get_dataloader(bs))
                for _ in range(num_iterations):
                    if _reached_soft_dram_limit(self.dram_limit_percent):
                        logger.warning(
                            "Reached soft DRAM limit (%.2f%%) during initialization. Stopping further input initialization",
                            self.dram_limit_percent * 100,
                        )
                        progress.stop_task(task_id=task_id)
                        progress.stop()
                        return False
                    data = _move_to_device(next(dl), torch.device("cpu"))
                    if torch.cuda.is_available():
                        data = _pin_memory(data)
                    self._data[bs]["<ENTRYPOINT>"].append(data)
                    progress.update(task_id, advance=1)
        gc.collect()
        gc.collect()
        return True

    def _profile_component(
        self,
        component_name: str,
        batch_size: int,
        num_warmup: int,
        num_iterations: int,
        device: torch.device,
        save_outputs: bool = False,
    ):
        """Profiles a single component for a given batch size."""
        cpu_device = torch.device("cpu")
        total_iterations = num_warmup + num_iterations

        switchboard = self._load_from_cache_or_generate_to_cpu(batch_size, load_only=[component_name])

        # load component and move to device
        dependencies = switchboard.depends_on(component_name)
        switchboard.to_device(device)
        switchboard.eval()

        # Collect inputs for all iterations
        all_inputs = []
        for it in range(total_iterations):
            args, kwargs = [], {}
            for dep in dependencies:
                a, k = _normalize_call_inputs(self._data[batch_size][dep][it])
                args.extend(a)
                kwargs.update(k)
            all_inputs.append((args, kwargs))

        # run warmup + profiling iterations
        if device.type == "cuda":
            nvml_device_handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)

            with torch.no_grad():
                for j in range(0, num_warmup):
                    gpu_args, gpu_kwargs = _move_to_device(all_inputs[j], device, non_blocking=True)
                    output = switchboard.call_component(component_name, *gpu_args, **gpu_kwargs)
                    self._data[batch_size][component_name].append(_move_to_device(output, cpu_device))

                # setup profilers
                time_ms_list: list[float] = [0.0] * num_iterations
                peak_mem_mb_list: list[int] = [0] * num_iterations

                for j in range(0, num_iterations):
                    global_idx = j + num_warmup

                    gpu_args, gpu_kwargs = _move_to_device(all_inputs[global_idx], device, non_blocking=True)

                    # start recording
                    torch.cuda.reset_peak_memory_stats(device=device)
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    starter.record()

                    # do forward pass
                    output = switchboard.call_component(component_name, *gpu_args, **gpu_kwargs)

                    # end recording and consolidate
                    ender.record()
                    torch.cuda.synchronize()

                    time_ms_list[j] = starter.elapsed_time(ender)
                    peak_mem_mb_list[j] = torch.cuda.max_memory_allocated(device=device) // (2**20)

                    # move output to CPU
                    if save_outputs:
                        self._data[batch_size][component_name].append(_move_to_device(output, cpu_device))

                # Separate utilization pass to avoid CPU bottlenecks and get a realistic reading
                # We run the component in a tight loop for a short duration
                util_pct = 0
                if num_iterations > 0:
                    stress_args, stress_kwargs = _move_to_device(all_inputs[num_warmup], device)
                    torch.cuda.synchronize()
                    start_util = time.time()
                    while time.time() - start_util < 0.2:  # Run for 200ms
                        _ = switchboard.call_component(component_name, *stress_args, **stress_kwargs)
                        # TODO: util_pct is not very reliable.
                        util_pct = max(int(pynvml.nvmlDeviceGetUtilizationRates(nvml_device_handle).gpu), util_pct)
                    torch.cuda.synchronize()

                return time_ms_list, max(peak_mem_mb_list), util_pct

        else:
            raise RuntimeError("CPU profiling not yet implemented")

    def profile(self, num_warmup: int, num_iterations: int):
        # TODO: assumes homogenous devices; will need to update this to run model on different devices
        SYSTEM_MEMORY_GB = {
            "total": psutil.virtual_memory().total >> 30,
            "available": psutil.virtual_memory().available >> 30,
        }
        DEVICE_MEMORY_GB = {
            device_idx: (torch.cuda.get_device_properties(device=device_idx).total_memory >> 20) // 1000
            for device_idx in range(torch.cuda.device_count())
        }
        MEMORY_BUCKETS = {
            i for system_mem in DEVICE_MEMORY_GB.values() for i in range(1, system_mem + 1) if system_mem % i == 0
        }
        USABLE_DRAM = SYSTEM_MEMORY_GB["available"] * self.dram_limit_percent

        CURRENT_DEVICE = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

        TOTAL_ITERATIONS = num_warmup + num_iterations

        # Consolidate memory information logging
        gpu_info = ", ".join([f"GPU{i}={mem}GB" for i, mem in DEVICE_MEMORY_GB.items()])

        logger.info(
            "System Specs:\n"
            "  • DRAM: total=%d GB, available=%d GB, soft limit=%.2f GB\n"
            "  • GPU memory: %s\n"
            "  • GPU buckets (GB): %s",
            SYSTEM_MEMORY_GB["total"],
            SYSTEM_MEMORY_GB["available"],
            USABLE_DRAM,
            gpu_info,
            sorted(MEMORY_BUCKETS),
        )

        # Get topological order once (using BS 1)
        current_batch_size = 1
        topological_order = self._load_from_cache_or_generate_to_cpu(current_batch_size).get_topological_order()
        oom_occurred_on_largest_memory_slice = False

        # memory constraint, batch size, component name -> profiling results
        result_dict: dict[int, dict[int, dict[str, Any]]] = {}

        # iterate over decreasing memory sizes to guarantee that the model can execute the current batch size with the available memory
        while not oom_occurred_on_largest_memory_slice:
            ok = self._initialize_inputs(batch_sizes=[current_batch_size], num_iterations=TOTAL_ITERATIONS)

            if not ok:
                break

            total_steps = len(MEMORY_BUCKETS) * len(topological_order)
            with logging.progress_bar(f"Profiling BS={current_batch_size}", total=total_steps) as (progress, task_id):
                for idx, memory_contraint in enumerate(reversed(sorted(MEMORY_BUCKETS))):
                    torch.cuda.set_per_process_memory_fraction(
                        memory_contraint / DEVICE_MEMORY_GB[CURRENT_DEVICE.index]
                    )

                    for component_name in topological_order:
                        progress.update(
                            task_id,
                            description=f"Profiling {component_name} (BS={current_batch_size}, Mem={memory_contraint}GB)",
                        )
                        try:
                            with logging.suppress_logs():
                                time_ms_list, peak_mem_mb, util_pct = self._profile_component(
                                    component_name=component_name,
                                    batch_size=current_batch_size,
                                    num_warmup=num_warmup,
                                    num_iterations=num_iterations,
                                    device=CURRENT_DEVICE,
                                    save_outputs=(
                                        idx == 0
                                    ),  # only save the output for the first idx (largest memory bucket)
                                )

                            result_dict.setdefault(memory_contraint, {}).setdefault(current_batch_size, {})[
                                component_name
                            ] = {
                                "elapsed_time_ms": time_ms_list,
                                "peak_memory_mb": peak_mem_mb,
                                "utilization_pct": util_pct,
                            }

                            if _reached_soft_dram_limit(self.dram_limit_percent):
                                oom_occurred_on_largest_memory_slice = True  # force exit
                                break

                        except torch.cuda.OutOfMemoryError:
                            if idx == 0:
                                oom_occurred_on_largest_memory_slice = True
                                break
                        finally:
                            progress.update(task_id, advance=1)

                    if oom_occurred_on_largest_memory_slice:
                        break

                self._data[current_batch_size].clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()

            if not oom_occurred_on_largest_memory_slice:
                current_batch_size *= 2

            if current_batch_size > self.max_batch_size:
                break

        json.dump(
            result_dict,
            open(self.artifacts_dir / f"profiling_results_{self.read_only_client.get_name()}.json", "w"),
            indent=2,
        )
