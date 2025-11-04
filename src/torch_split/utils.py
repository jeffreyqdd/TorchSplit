import hashlib
import io
import json
import traceback

import torch
import torch._dynamo as dynamo
import torch._dynamo.exc as dynamo_exc
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from torch_split.client import SplitClient
from torch_split.profiling import annotators


def capture_graph2(client: SplitClient) -> fx.GraphModule:
    module = client.get_model()
    module.to("meta")
    for p in module.parameters():
        p.requires_grad_(False)

    args, kwargs = client.get_example_inputs()

    with FakeTensorMode() as mode:
        # Convert module parameters and buffers
        for name, param in module.named_parameters(recurse=True):
            setattr(module, name, mode.from_tensor(param))
        for name, buf in module.named_buffers(recurse=True):
            setattr(module, name, mode.from_tensor(buf))

        # Convert all example inputs to fake tensors
        fake_args = tuple(mode.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args)
        fake_kwargs = {
            k: (mode.from_tensor(v) if isinstance(v, torch.Tensor) else v)
            for k, v in kwargs.items()
        }

        # Symbolic tracing under fake tensor mode
        gm = make_fx(module, tracing_mode="symbolic")(*fake_args, **fake_kwargs)


    gm.graph.lint()
    gm.recompile()
    return gm


def capture_graph(client: SplitClient) -> fx.GraphModule:
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
        args, kwargs = client.get_example_inputs()

        export_result = dynamo.export(model)(*args, **kwargs)
        if isinstance(export_result, dynamo.eval_frame.ExportResult):
            annotators.DeviceAnnotator(export_result.graph_module).run(*args, **kwargs)
            return export_result.graph_module
    except dynamo_exc.TorchRuntimeError:
        traceback.print_exc()
        traceback.print_stack()
        raise RuntimeError("ensure model and example inputs are on the same device")

    raise ValueError("Export did not return an ExportResult")


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
    arch = {"class": model.__class__.__name__, "modules": []}
    for name, module in model.named_modules():
        if name == "":
            continue
        module_info = {
            "name": name,
            "class": module.__class__.__name__,
            "params": {k: tuple(v.shape) for k, v in module._parameters.items() if v is not None},
        }
        arch["modules"].append(module_info)

    arch_str = json.dumps(arch, sort_keys=True)
    return hashlib.sha256(arch_str.encode()).hexdigest()
