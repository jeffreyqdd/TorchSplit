import io

import torch
import torch._dynamo as dynamo
import torch.fx as fx

from torch_split.client import TorchSplitClient


def capture_graph(client: TorchSplitClient) -> fx.GraphModule:
    """
    Capture the computation graph of a PyTorch model using torch.fx.

    Args:
        client (TorchSplitClient): The client providing the model and example inputs.

    Returns:
        fx.GraphModule: The captured computation graph as a GraphModule.

    Raises:
        ValueError: If the export does not return an ExportResult.
    """

    # no need to move to specified device as we are only generating the graph IR
    # no performance measurement is done here
    model = client.get_model()
    args, kwargs = client.get_example_inputs()

    export_result = dynamo.export(model)(*args, **kwargs)

    if isinstance(export_result, dynamo.eval_frame.ExportResult):
        # from torch.fx.passes.shape_prop import ShapeProp
        # ShapeProp(graph_module).propagate(example_input)
        return export_result.graph_module
        # annoate

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

    return ret
