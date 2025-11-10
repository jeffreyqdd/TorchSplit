import hashlib
import io
import json
import traceback

import torch
import torch._dynamo as dynamo
import torch._dynamo.exc as dynamo_exc
import torch.fx as fx

from torch_split.client import SplitClient
from torch_split.profiling import annotators


def capture_graph(client: SplitClient, *args, **kwargs) -> fx.GraphModule:
    """
    Capture the computation graph of a PyTorch model using torch.fx.

    Args:
        client (SplitClient): The client providing the model and example inputs.

    Returns:
        fx.GraphModule: The captured computation graph as a GraphModule with shape and device information

    Raises:
        RuntimeError: If the model and example inputs are not on the same device
        ValueError: If the export does not return an ExportResult
    """
    try:
        # no need to move to specified device as we are only generating the graph IR
        # no performance measurement is done here
        model = client.get_model()
        dynamo.config.dynamic_shapes = False
        export_result = dynamo.export(model, tracing_mode="concrete")(*args, **kwargs)
        if isinstance(export_result, dynamo.eval_frame.ExportResult):
            dynamo.export(export_result.graph_module)
            return export_result.graph_module
        raise ValueError("Export did not return an ExportResult")
    except dynamo_exc.TorchRuntimeError:
        traceback.print_exc()
        traceback.print_stack()
        raise RuntimeError("ensure model and example inputs are on the same device")


def write_graph_to_buffer(gm: fx.GraphModule, buffer: io.BytesIO) -> None:
    """
    Write the FX graph or GraphModule to a bytes buffer.

    Args:
        gm (fx.GraphModule): The GraphModule to serialize.
        buffer (io.BytesIO): The buffer to write the serialized graph to.
    """
    torch.save(gm, buffer)


def read_graph_from_buffer(buffer: io.BytesIO) -> fx.GraphModule:
    """
    Read an FX graph from a bytes buffer.
    Args:
        buffer (io.BytesIO): The buffer to read the serialized graph from.
    Returns:
        fx.Graph: The deserialized FX graph.
    Raises:
        ValueError: If the deserialized object is not an fx.Graph.
    """

    buffer.seek(0)
    ret = torch.load(buffer, weights_only=False)

    if not isinstance(ret, fx.GraphModule):
        raise ValueError(f"Deserialized object is not an fx.Graph but {type(ret)}")

    return ret


def hash_model_architecture(model: torch.nn.Module) -> str:
    """Generate a hash representing the architecture of a PyTorch model."""
    modules = []
    for name, module in model.named_modules():
        if name == "":
            continue
        module_info = {
            "name": name,
            "class": module.__class__.__name__,
            "params": {k: tuple(v.shape) for k, v in module._parameters.items() if v is not None},
        }
        modules.append(module_info)

    arch_str = json.dumps({"class": model.__class__.__name__, "modules": modules}, sort_keys=True)
    return hashlib.sha256(arch_str.encode()).hexdigest()
