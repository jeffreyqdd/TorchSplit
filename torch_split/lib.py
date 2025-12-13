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


def get_partion_and_roots(split_interface: SplitClient) -> tuple[list[Partition], PartitionProvider]:
    """Generate a  template that can be reused to create identical component allocation for different batch sizes."""
    _a, _b, generator = split_interface.get_benchmarks(32)
    args, kwargs = next(generator)
    model = split_interface.get_model()
    model_name = model.__class__.__name__
    gm = utils.capture_graph(model)(*args, **kwargs)
    tg = TorchGraph.from_fx_graph(gm, label=model_name)
    pp = PartitionProvider(tg)
    ap = sorted(pp.all_partitions(), key=lambda p: -sum(len(sg.enclosed_region) for sg in p.subgraphs))
    # sb = pp.create_switchboard([ap[0]])
    return [ap[0]], pp
