import argparse
import importlib
import json
import sys
from pathlib import Path

import torch_split.client as client
import torch_split.core as core
import torch_split.logging as logging
import torch_split.utils as utils

sys.path.insert(0, ".")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch Split CLI")
    parser.add_argument(
        "model", type=str, help="Path to TorchSplitClient interface instance e.g. './src/main.py:client'"
    )
    parser.add_argument("-v", action="store_true", help="enable verbose")
    parser.add_argument("-vv", action="store_true", help="enable more verbose")
    parser.add_argument("-vvv", action="store_true", help="enable most verbose")
    parser.add_argument(
        "-o", "--output", type=str, default="torchsplit_bin", help="output path for artifacts and cache"
    )
    parser.add_argument(
        "-d", "--dataflow", action="store_true", help="render visualization of dataflow graph in the output path"
    )
    args = parser.parse_args()

    # set logging level
    if args.vvv:
        args.log_level = "DEBUG"
    elif args.vv:
        args.log_level = "INFO"
    elif args.v:
        args.log_level = "WARNING"
    else:
        args.log_level = "ERROR"

    logging.set_level(args.log_level)
    logger = logging.get_logger(__name__)

    module_path, class_name = args.model.split(":")
    module_path = module_path.replace("/", ".").rstrip(".py").lstrip(".")
    logger.debug("loading '%s' from module: '%s'", class_name, module_path)

    # load module
    module = importlib.import_module(module_path)
    split_client: client.SplitClient = getattr(module, class_name)()
    model_name = split_client.get_model().__class__.__name__
    model_hash = utils.hash_model_architecture(split_client.get_model())

    # create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    architecture_hash_file = output_dir / "architecture.hash"
    architecture_hash_file.touch(exist_ok=True)

    with open(architecture_hash_file, "r") as f:
        previous_hash = f.readline().strip()
        require_update = previous_hash != model_hash

    logger.info("artifacts and cache will be stored in %s", output_dir.absolute())
    logger.info("model: %s", model_name)
    logger.info("model hash: %s", model_hash)

    gm = utils.capture_graph(split_client)

    # check to see if there's a need to run profiling
    if require_update:
        logger.info("model architecture has changed since last run, running profiling")
        instrumented_model = client.InstrumentedModule(gm)
        instrumented_model = split_client.setup_benchmark(instrumented_model)
        instrumented_model.export_to_file(output_dir / "profiling.json")
        logger.info("profiling data saved to %s", (output_dir / "profiling.json").absolute())
        with open(architecture_hash_file, "w") as f:
            f.write(model_hash + "\n")

    with open(output_dir / "profiling.json", "r") as f:
        profiling_data = json.load(f)

    # create annotated graph
    torch_graph = core.TorchGraph.from_fx_graph(gm)
    torch_graph.annotate_with_profiling_data(profiling_data)
    partition_provider = core.PartitionProvider(torch_graph)

    if args.dataflow:
        partition_provider.visualize_dataflow(output_dir / "visualizations")

    # run partitioning
    partition_provider.create_partition()