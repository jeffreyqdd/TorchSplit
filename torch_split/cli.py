#!/usr/bin/env python3

import argparse
import importlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch_split.core.ir as ir
import torch_split.core.log as logging
import torch_split.core.profiling.annotators as annotators
import torch_split.core.utils as utils
import torch_split.interface as client
from torch_split.core.partition import provider, solver

sys.path.insert(0, ".")


logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_hash: str
    artifact_directory: Path
    split_client: client.SplitClient


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
        return getattr(module, class_name)()

    except Exception:
        logger.exception("failed to load '%s:%s'", module_path, class_name)
        sys.exit(1)


def get_cut(split_interface: client.SplitClient):
    _a, _b, generator = split_interface.get_benchmarks(32)
    args, kwargs = next(generator)
    model = split_interface.get_model()
    model_name = model.__class__.__name__
    gm = utils.capture_graph(model)(*args, **kwargs)
    tg = ir.TorchGraph.from_fx_graph(gm, label=model_name)
    pp = provider.PartitionProvider(tg)
    ap = sorted(pp.all_partitions(), key=lambda p: -sum(len(sg.enclosed_region) for sg in p.subgraphs))
    return [ap[0]], pp


def export_fun(program_args: argparse.Namespace):
    if not program_args.o.endswith(".tspartd"):
        print("output path must have .tspartd extension")
        sys.exit(1)

    batch_size: int = program_args.b
    output_dir = Path(program_args.o)
    split_interface: client.SplitClient = _load_attr_from_module_path(program_args.model)

    model = split_interface.get_model()
    model_name = model.__class__.__name__
    model_hash = utils.hash_model_architecture(model)

    logger.info("model [cyan]%s[/] [dim](hash %s...)[/] batch size = %d", model_name, model_hash[:12], batch_size)
    _a, _b, generator = split_interface.get_benchmarks(batch_size)
    args, kwargs = next(generator)

    gm = utils.capture_graph(model)(*args, **kwargs)
    tg = ir.TorchGraph.from_fx_graph(gm, label=model_name)
    pp = provider.PartitionProvider(tg)
    ap = sorted(pp.all_partitions(), key=lambda p: -sum(len(sg.enclosed_region) for sg in p.subgraphs))
    sb = pp.create_switchboard([ap[0]])
    logger.info("exporting switchboard to %s", str(output_dir))
    sb.save(output_dir)


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
    if program_args.t:
        _load_attr_from_module_path(program_args.t)()

    artifact_dir = Path(program_args.o)
    batch_sizes: list[int] = program_args.b
    memory_sizes: list[int] = program_args.memory
    debug_mode: bool = program_args.s
    split_client: client.SplitClient = _load_attr_from_module_path(program_args.model)

    model = split_client.get_model()
    model_name = model.__class__.__name__
    model_hash = utils.hash_model_architecture(model)

    model_spec = ModelSpec(model_name, model_hash, artifact_dir, split_client)

    logger.info("model [cyan]%s[/] [dim](hash %s...)[/] batch size = %d", model_name, model_hash[:12])
    logger.info("artifact directory: %s", str(artifact_dir))
    logger.info("batch sizes: %s", ", ".join(map(str, batch_sizes)))
    logger.info("memory sizes: %s", ", ".join(map(str, memory_sizes)))

    # with assert_transaction(model_spec, program_args.no_cache) as requires_update:
    #     if not requires_update:
    #         logger.info("model architecture unchanged, skipping optimization")
    #         return

    #     cut, root = get_cut(split_client)
    #     for batch_size in batch_sizes:

    # _a, _b, generator = split_interface.get_benchmarks(batch_size)
    # args, kwargs = next(generator)

    # gm = utils.capture_graph(model)(*args, **kwargs)
    # tg = ir.TorchGraph.from_fx_graph(gm, label=model_name)
    # pp = provider.PartitionProvider(tg)
    # ap = sorted(pp.all_partitions(), key=lambda p: -sum(len(sg.enclosed_region) for sg in p.subgraphs))
    # sb = pp.create_switchboard([ap[0]])
    # logger.info("exporting switchboard to %s", str(output_dir))
    # sb.save(output_dir)


def optimizer_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-t",
        "--trace",
        type=str,
        help="module path e.g. src.provider:console_provider for trace provider hook",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output path for the optimization and profiling artifacts",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Batch size to use for optimization and profiling",
    )

    parser.add_argument(
        "-m",
        "--memory",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Memory slice sizes to use for optimization and profiling",
    )

    parser.add_argument(
        "-s",
        default=False,
        action="store_true",
        help="Output intermediates for debugging and visualization",
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
