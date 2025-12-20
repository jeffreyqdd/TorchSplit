#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import sys
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import torch_split.log as logging
from torch_split.core import SplitClient, batch_compiler, get_partition_template, utils
from torch_split.optimizer import Profiler, Solver

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*_register_pytree_node.*",
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*resume_download.*",
)

sys.path.insert(0, ".")


logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_hash: str
    artifact_directory: Path
    split_client: SplitClient


@contextmanager
def assert_transaction(model_spec: ModelSpec, no_cache: bool = False):
    try:
        architecture_hash_file = model_spec.artifact_directory / "architecture.hash"
        architecture_hash_file.touch(exist_ok=True)

        with open(architecture_hash_file, "r") as f:
            previous_hash = f.readline().strip()
            require_update = previous_hash != model_spec.model_hash

        if no_cache or not require_update:
            yield False
        else:
            yield True
    except Exception as e:
        raise e
    else:
        with open(architecture_hash_file, "w") as f:
            f.write(model_spec.model_hash + "\n")


def _load_attr_from_module_path(module_path: str) -> Any:
    module_path, class_name = module_path.split(":")
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)(), lambda: getattr(module, class_name)()

    except Exception:
        logger.exception("failed to load '%s:%s'", module_path, class_name)
        sys.exit(1)


def export_fun(program_args: argparse.Namespace):
    if not program_args.o.endswith(".tspartd"):
        print("output path must have .tspartd extension")
        sys.exit(1)

    batch_size: int = program_args.b
    output_dir = Path(program_args.o)
    split_tuple: tuple[SplitClient, Callable[[], SplitClient]] = _load_attr_from_module_path(program_args.model)
    split_interface, _ = split_tuple

    model = split_interface.get_model()
    model_name = model.__class__.__name__
    model_hash = utils.hash_model_architecture(model)

    logger.info("model [cyan]%s[/] [dim](hash %s...)[/] batch size = %d", model_name, model_hash[:12], batch_size)
    pt = get_partition_template(split_interface)
    compiled = batch_compiler(split_interface, pt, batch_size)
    compiled.save(output_dir)
    logger.info("exported switchboard to %s", str(output_dir))


def export_parser(parser: argparse.ArgumentParser):
    # check if on linux
    if sys.platform == "linux" or sys.platform == "linux2":
        default_path = "/dev/shm/switchboard.tspartd"
    else:
        default_path = f"{os.getcwd()}/switchboard.tspartd"
    parser.add_argument(
        "-b",
        type=int,
        default=1,
        help="batch size to use for export (default=1)",
    )
    parser.add_argument(
        "-o",
        type=str,
        default=default_path,
        help=f"output directory for the exported switchboard (default={default_path})",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Module path to TorchSplitClient interface instance e.g. 'src.main:client'",
    )
    parser.set_defaults(func=export_fun)


def optimizer_fun(program_args: argparse.Namespace):
    artifact_dir = Path(program_args.output).absolute()
    debug_mode: bool = program_args.s

    split_client, client_factory = _load_attr_from_module_path(program_args.model)
    model = split_client.get_model()
    model_name = model.__class__.__name__
    model_hash = utils.hash_model_architecture(model)

    logger.info("model [cyan]%s[/] [dim](hash %s...)[/]", model_name, model_hash[:12])
    logger.info("artifact directory: %s", str(artifact_dir))

    if not artifact_dir.exists():
        artifact_dir.mkdir(parents=True, exist_ok=True)
        profiler = Profiler(client_factory, artifact_dir, dram_limit_percent=0.6, max_batch_size=program_args.max_batch)
        profiler.profile(num_warmup=32, num_iterations=64)

    if debug_mode:
        logger.info("debug mode enabled: intermediates will be saved to artifact directory")
        template = get_partition_template(split_client)
        template.provider.visualize_dataflow(artifact_dir / "visualizations", True)

    # read profiling_results.json
    profiling_file = artifact_dir / f"profiling_results_{split_client.get_name()}.json"
    if not profiling_file.exists():
        logger.error("inconsistent cache. Try deleting file and rerunning command.", str(artifact_dir))
        sys.exit(1)

    with open(profiling_file, "r") as f:
        profiling_data = json.load(f)

    DEVICE_MEMORY_GB = {
        device_idx: (torch.cuda.get_device_properties(device=device_idx).total_memory >> 20) // 1000
        for device_idx in range(torch.cuda.device_count())
    }

    # cap it at 4GB minimum or solver takes too long
    MEMORY_BUCKETS = list(
        filter(
            lambda x: x >= 4,
            sorted(
                {i for system_mem in DEVICE_MEMORY_GB.values() for i in range(1, system_mem + 1) if system_mem % i == 0}
            ),
        )
    )

    solver = Solver(
        num_gpus=4,
        memory_slices=MEMORY_BUCKETS,
        profile_result=profiling_data,
    )

    locked, assignment, utilization = solver.solve_leximin_for_batch(batch_size=1)

    logger.info(f"Pipeline Throughput: {min(locked.values()):.2f}")
    logger.info("Leximin Model Throughputs:")
    for model in locked:
        logger.info(f" {model}: {locked[model]:.2f}")

    logger.info("Assignments:")
    for (model, node, c), count in assignment.items():
        logger.info(f" - Model {model} assigned {count}x to GPU{node} with memory slice {c}GB")

    logger.info("Node Utilization:")
    nodes = list(range(4))
    logger.info(f"Nodes: {nodes}")
    # logger.info(f"Utilization keys: {utilization.keys()}")
    for node in nodes:
        u = 0
        for (model, n, c), count in assignment.items():
            if n == node:
                val = utilization.get(model, {}).get(c, 0)
                # logger.info(f"  GPU{node}: Model {model}, Count {count}, Slice {c}, Util {val}")
                u += count * val
        logger.info(f" - GPU{node}: {u:.1f}%")


def optimizer_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-o",
        "--output",
        default="./ts_bin",
        type=str,
        help="Output path for the optimization and profiling artifacts",
    )

    parser.add_argument(
        "-s",
        default=False,
        action="store_true",
        help="Output intermediates for debugging and visualization",
    )

    parser.add_argument(
        "--max-batch",
        type=int,
        default=128,
        help="Maximum batch size to profile up to (default=128)",
    )

    parser.add_argument(
        "model",
        type=str,
        help="Module path to TorchSplitClient interface instance e.g. 'src.main:client'",
    )

    parser.set_defaults(func=optimizer_fun)


def main():
    parser = argparse.ArgumentParser(description="torchsplit cli")
    subparser = parser.add_subparsers(dest="command", required=True)
    export_parser(subparser.add_parser("export", help="export a model to a switchboard"))
    optimizer_parser(subparser.add_parser("optimize", help="find optimal allocation for a model"))

    program_args = parser.parse_args()
    program_args.func(program_args)

    # )
    # parser.add_argument(
    #     "-c",
    #     "--cache",
    #     type=str,
    #     default=".ts_bin",
    #     help="output path for artifacts and cache",
    # )
    # parser.add_argument(
    #     "-d",
    #     "--dataflow",
    #     action="store_true",
    #     help="render visualization of dataflow graph in the output path",
    # )
    # parser.add_argument(
    #     "--no-cache",
    #     action="store_true",
    #     help="disable caching of model architecture hash",
    # )
    # program_args = parser.parse_args()

    # # Resolve and load user-provided SplitClients
    # target_models: dict[str, ModelSpec] = {}

    # partitions: dict[str, tuple[ModelSpec, provider.PartitionProvider]] = {}
    # for model_name, model_spec in target_models.items():
    #     split_client = model_spec.split_client
    #     a, b, generator = split_client.get_benchmarks(split_client.batch_sizes()[0])
    #     args, kwargs = next(generator)
    #     gm = utils.capture_graph(split_client.get_model())(*args, **kwargs)
    #     tg = ir.TorchGraph.from_fx_graph(gm, label=model_name)
    #     provider_partition = provider.PartitionProvider(tg)
    #     all_partitions = provider_partition.all_partitions()
    #     all_partitions = sorted(all_partitions, key=lambda p: -sum(len(sg.enclosed_region) for sg in p.subgraphs))

    #     selected_partitions = [all_partitions[0]]
    #     layout = provider_partition.create_switchboard(selected_partitions)
    #     layout.save(Path("switchboard.tmp"))
    #     # print(json.dumps(final, indent=2))
    #     # for id, d in data.items():
    #     #     print("------------------")
    #     #     print(id)
    #     #     print(d.code)

    #     # for p in all_partitions[:1]:
    #     #     print("cut: ", [n.name for n in p.cut.split], "→", [n.name for n in p.cut.join])
    #     #     print("  subgraphs count: ", len(p.subgraphs))
    #     #     for idx, subgraph in enumerate(p.subgraphs, 1):
    #     #         print(f"  Subgraph {idx}:")
    #     #         print("      inputs: ", [n.name for n in subgraph.inputs])
    #     #         print("      outputs: ", [n.name for n in subgraph.outputs])
    #     #         print("      enclosed region: ", len(subgraph.enclosed_region))
    #     #     print()

    #     if program_args.dataflow:
    #         output_dir = model_spec.cache_directory
    #         provider_partition.visualize_dataflow(output_dir / "visualizations", True)
    #     # for bs, profiling_data, device_data in assert_model_cache(
    #     #     model_spec, program_args.no_cache
    #     # ):
    #     #     tg.annotate_with_profiling_data(bs, profiling_data, device_data)

    #     # partitions[model_name] = (model_spec, provider.PartitionProvider(tg))

    # # if program_args.dataflow:
    # #     for partition in partitions.values():
    # #         output_dir = partition[0].cache_directory
    # #         partition[1].visualize_dataflow(output_dir / "visualizations", True)

    # # solver.solve(list(map(lambda x: x[1], partitions.values())))

    # # # run partitioning
    # # with logging.timed_execution("solve partitioning problem", logger=logger):
    # #     solutions = partition_provider.solve_partitioning_problem()
    # # logger.info(
    # #     "[bold]Done[/] found %d candidate partition(s)",
    # #     len(solutions) if solutions is not None else 0,
    # # )
    # # # torch_graph_dict: dict[int, core.TorchGraph] = {}
    # # # device_dict: dict[int, str] = {}
    # # # profiling_dict: dict[int, str] = {}


if __name__ == "__main__":
    main()
