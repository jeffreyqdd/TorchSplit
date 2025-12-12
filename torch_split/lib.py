from torch_split.interface import SplitClient
from torch_split.compiler.partition.provider import Partition, PartitionProvider
from torch_split.compiler.switchboard import Switchboard
from torch_split.compiler.ir import TorchGraph
import torch_split.compiler.utils as utils


def batch_compiler(
    client: SplitClient, partitions: list[Partition], root: PartitionProvider, batch_size: int
) -> Switchboard:
    model = client.get_model()
    model_name = model.__class__.__name__
    _a, _b, generator = client.get_benchmarks(batch_size)
    args, kwargs = next(generator)
    gm = utils.capture_graph(client.get_model())(*args, **kwargs)
    tg = TorchGraph.from_fx_graph(gm, label=model_name)
    return PartitionProvider(tg).create_switchboard(partitions, roots=root)
