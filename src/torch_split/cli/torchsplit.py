import argparse
import importlib
import json
import sys
from pathlib import Path

import torch_split.client as client
import torch_split.core as core
import torch_split.logging as logging
import torch_split.profiling.annotators as annotators
import torch_split.utils as utils

sys.path.insert(0, ".")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch Split CLI")
    parser.add_argument(
        "model",
        type=str,
        help="Path to TorchSplitClient interface instance e.g. './src/main.py:client'",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="torchsplit_bin",
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

    logger = logging.get_logger(__name__)

    module_path, class_name = program_args.model.split(":")
    module_path = module_path.replace("/", ".").rstrip(".py").lstrip(".")
    logger.info("loading '%s' from module: '%s'", class_name, module_path)

    # load module
    module = importlib.import_module(module_path)
    split_client: client.SplitClient = getattr(module, class_name)()
    model_name = split_client.get_model().__class__.__name__
    model_hash = utils.hash_model_architecture(split_client.get_model())

    # create output directory
    output_dir = Path(program_args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    architecture_hash_file = output_dir / "architecture.hash"
    architecture_hash_file.touch(exist_ok=True)

    with open(architecture_hash_file, "r") as f:
        previous_hash = f.readline().strip()
        require_update = previous_hash != model_hash or program_args.no_cache

    logger.info("model: %s", model_name)
    logger.info("model hash: %s", model_hash)
    logger.info("model artifacts: %s", output_dir.absolute())

    if require_update:
        logger.info(
            "model architecture has changed since last run, updating annotations and profiling data"
        )

        annotation_dir = output_dir / "annotations"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        for file in annotation_dir.iterdir():
            file.unlink()

        for batch_size in split_client.batch_sizes():
            logger.debug("capturing graph for batch size %d", batch_size)

            device_file = annotation_dir / f"device_batchsize_{batch_size}.json"
            profiling_file = annotation_dir / f"profiling_batchsize_{batch_size}.json"
            device_file.parent.mkdir(parents=True, exist_ok=True)
            profiling_file.parent.mkdir(parents=True, exist_ok=True)

            # generate graph
            warmup_rounds, iterations, generator = split_client.get_benchmarks(
                batch_size
            )
            args, kwargs = generator.__next__()
            gm = utils.capture_graph(split_client, *args, **kwargs)

            d_annotator = annotators.DeviceAnnotator(gm)
            d_annotator.run(*args, **kwargs)
            rt_annotator = annotators.RuntimeAnnotator(gm)
            with rt_annotator:
                rt_annotator.set_mode(annotators.RuntimeAnnotator.Mode.WARMUP)
                for _ in range(warmup_rounds):
                    args, kwargs = generator.__next__()
                    rt_annotator.run(*args, **kwargs)

                for mode in annotators.RuntimeAnnotator.Mode.profiling_modes():
                    rt_annotator.set_mode(mode)
                    for _ in range(iterations):
                        args, kwargs = generator.__next__()
                        rt_annotator.run(*args, **kwargs)

            # save
            with open(device_file, "w") as f:
                f.write(d_annotator.get_json())

            with open(profiling_file, "w") as f:
                f.write(rt_annotator.get_json())

        # update architecture hash
        with open(architecture_hash_file, "w") as f:
            f.write(model_hash + "\n")

    warmup_rounds, iterations, generator = split_client.get_benchmarks(
        split_client.batch_sizes()[0]
    )
    args, kwargs = generator.__next__()
    gm = utils.capture_graph(split_client, *args, **kwargs)
    torch_graph = core.TorchGraph.from_fx_graph(gm)

    logger.info("annotating graph with profiling data")
    for batch_size in split_client.batch_sizes():
        device_file = output_dir / "annotations" / f"device_batchsize_{batch_size}.json"
        profiling_file = (
            output_dir / "annotations" / f"profiling_batchsize_{batch_size}.json"
        )

        with open(profiling_file, "r") as f:
            profiling_data = json.load(f)

        with open(device_file, "r") as f:
            device_data = json.load(f)

        torch_graph.annotate_with_profiling_data(batch_size, profiling_data)

    logger.info("generating partitions")
    partition_provider = core.PartitionProvider(torch_graph)

    if program_args.dataflow:
        partition_provider.visualize_dataflow(output_dir / "visualizations", True)
        partition_provider.visualize_dominance(output_dir / "visualizations")

    # # run partitioning
    partition_provider.solve_partitioning_problem()

    # torch_graph_dict: dict[int, core.TorchGraph] = {}
    # device_dict: dict[int, str] = {}
    # profiling_dict: dict[int, str] = {}
