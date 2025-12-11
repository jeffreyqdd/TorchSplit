#!/usr/bin/env python3

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch_split.lib.client as client
import torch_split.lib.log as logging
import torch_split.lib.ir as ir
import torch_split.lib.profiling.annotators as annotators
import torch_split.lib.utils as utils
from torch_split.lib.partition import provider, solver

sys.path.insert(0, ".")


logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hash: str
    cache_directory: Path
    split_client: client.SplitClient


def assert_model_cache(model_spec: ModelSpec, no_cache: bool = False):
    model_spec.cache_directory.mkdir(parents=True, exist_ok=True)

    architecture_hash_file = model_spec.cache_directory / "architecture.hash"
    architecture_hash_file.touch(exist_ok=True)

    annotation_directory = model_spec.cache_directory / "annotations"
    annotation_directory.mkdir(parents=True, exist_ok=True)

    profiling_files = {
        bs: annotation_directory / f"profiling_batchsize_{bs}.json" for bs in model_spec.split_client.batch_sizes()
    }

    device_files = {
        bs: annotation_directory / f"device_batchsize_{bs}.json" for bs in model_spec.split_client.batch_sizes()
    }

    with open(architecture_hash_file, "r") as f:
        previous_hash = f.readline().strip()
        require_update = previous_hash != model_spec.hash

    files_missing = any(
        not x.exists() or not y.exists() for x, y in zip(profiling_files.values(), device_files.values())
    )

    if previous_hash != model_spec.hash:
        logger.info("detected model architecture change")
    elif files_missing:
        logger.info("detected missing profiling artifacts")
    elif no_cache:
        logger.info("detected no cache option")

    if require_update or no_cache or files_missing:
        # clean previous artifacts for consistency
        for file in annotation_directory.iterdir():
            try:
                file.unlink()
            except Exception:
                logger.warning("could not remove old artifact: %s", file)

        for bs in model_spec.split_client.batch_sizes():
            profiling_file = profiling_files[bs]
            device_file = device_files[bs]
            warmup_rounds, iterations, generator = model_spec.split_client.get_benchmarks(bs)

            args, kwargs = next(generator)
            gm = utils.capture_graph(model_spec.split_client.get_model())(*args, **kwargs)
            d_annotator = annotators.DeviceAnnotator(gm).run(*args, **kwargs)
            rt_annotator = annotators.RuntimeAnnotator(gm)
            with rt_annotator:
                rt_annotator.set_mode(annotators.RuntimeAnnotator.Mode.WARMUP)
                for _ in range(warmup_rounds):
                    args, kwargs = next(generator)
                    rt_annotator.run(*args, **kwargs)

                for mode in annotators.RuntimeAnnotator.Mode.profiling_modes():
                    rt_annotator.set_mode(mode)
                    for _ in range(iterations):
                        args, kwargs = next(generator)
                        rt_annotator.run(*args, **kwargs)

            with open(device_file, "w") as f:
                f.write(d_annotator.get_json())
            with open(profiling_file, "w") as f:
                f.write(rt_annotator.get_json())

        with open(architecture_hash_file, "w") as f:
            f.write(model_spec.hash + "\n")

    for bs in model_spec.split_client.batch_sizes():
        with open(profiling_files[bs], "r") as f:
            profiling_data = json.load(f)

        with open(device_files[bs], "r") as f:
            device_data = json.load(f)

        yield bs, profiling_data, device_data


def main():
    parser = argparse.ArgumentParser(description="Torch Split CLI")
    parser.add_argument(
        "model",
        type=str,
        help="Path to TorchSplitClient interface instance e.g. './src/main.py:client'",
        nargs="+",
    )
    parser.add_argument(
        "-c",
        "--cache",
        type=str,
        default=".ts_bin",
        help="output path for artifacts and cache",
    )
    parser.add_argument(
        "-d",
        "--dataflow",
        action="store_true",
        help="render visualization of dataflow graph in the output path",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="disable caching of model architecture hash",
    )
    program_args = parser.parse_args()

    # Resolve and load user-provided SplitClients
    target_models: dict[str, ModelSpec] = {}

    for split_client_spec in program_args.model:
        module_path, class_name = split_client_spec.split(":")
        module_path = module_path.replace("/", ".").rstrip(".py").lstrip(".")
        logger.info("loading client %s:%s", module_path, class_name)

        try:
            module = importlib.import_module(module_path)
            split_client: client.SplitClient = getattr(module, class_name)()
        except Exception:
            logger.exception("failed to load SplitClient '%s:%s'", module_path, class_name)
            sys.exit(1)

        model = split_client.get_model()
        model_name = model.__class__.__name__
        model_hash = utils.hash_model_architecture(model)
        model_cache_directory = Path(program_args.cache) / model_name / model_hash

        target_models[model_name] = ModelSpec(
            name=model_name,
            hash=model_hash,
            cache_directory=model_cache_directory,
            split_client=split_client,
        )

        logger.info(
            "model [cyan]%s[/] [dim](hash %s...)[/]\nmodel cache directory [dim](%s...)[/]",
            model_name,
            model_hash[:12],
            str(model_cache_directory)[:40],
        )

    partitions: dict[str, tuple[ModelSpec, provider.PartitionProvider]] = {}
    for model_name, model_spec in target_models.items():
        split_client = model_spec.split_client
        a, b, generator = split_client.get_benchmarks(split_client.batch_sizes()[0])
        args, kwargs = next(generator)
        gm = utils.capture_graph(split_client.get_model())(*args, **kwargs)
        tg = ir.TorchGraph.from_fx_graph(gm, label=model_name)
        provider_partition = provider.PartitionProvider(tg)
        all_partitions = provider_partition.all_partitions()
        all_partitions = sorted(all_partitions, key=lambda p: -sum(len(sg.enclosed_region) for sg in p.subgraphs))

        selected_partitions = [all_partitions[0]]
        layout = provider_partition.create_switchboard(selected_partitions)
        layout.save(Path("/dev/shm/switchboard.tmp"))
        # print(json.dumps(final, indent=2))
        # for id, d in data.items():
        #     print("------------------")
        #     print(id)
        #     print(d.code)

        # for p in all_partitions[:1]:
        #     print("cut: ", [n.name for n in p.cut.split], "→", [n.name for n in p.cut.join])
        #     print("  subgraphs count: ", len(p.subgraphs))
        #     for idx, subgraph in enumerate(p.subgraphs, 1):
        #         print(f"  Subgraph {idx}:")
        #         print("      inputs: ", [n.name for n in subgraph.inputs])
        #         print("      outputs: ", [n.name for n in subgraph.outputs])
        #         print("      enclosed region: ", len(subgraph.enclosed_region))
        #     print()

        if program_args.dataflow:
            output_dir = model_spec.cache_directory
            provider_partition.visualize_dataflow(output_dir / "visualizations", True)
        # for bs, profiling_data, device_data in assert_model_cache(
        #     model_spec, program_args.no_cache
        # ):
        #     tg.annotate_with_profiling_data(bs, profiling_data, device_data)

        # partitions[model_name] = (model_spec, provider.PartitionProvider(tg))

    # if program_args.dataflow:
    #     for partition in partitions.values():
    #         output_dir = partition[0].cache_directory
    #         partition[1].visualize_dataflow(output_dir / "visualizations", True)

    # solver.solve(list(map(lambda x: x[1], partitions.values())))

    # # run partitioning
    # with logging.timed_execution("solve partitioning problem", logger=logger):
    #     solutions = partition_provider.solve_partitioning_problem()
    # logger.info(
    #     "[bold]Done[/] found %d candidate partition(s)",
    #     len(solutions) if solutions is not None else 0,
    # )
    # # torch_graph_dict: dict[int, core.TorchGraph] = {}
    # # device_dict: dict[int, str] = {}
    # # profiling_dict: dict[int, str] = {}


if __name__ == "__main__":
    main()
