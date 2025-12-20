from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .partition import Partition, PartitionProvider
from .ir import TorchGraph
from .switchboard import Switchboard
from .utils import capture_graph, hash_model_architecture


class SplitClient(ABC):
    """Abstract base class for clients that provide models and data loaders for splitting."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the model."""
        raise NotImplementedError("SplitClient.get_model is not implemented")

    @abstractmethod
    def get_dataloader(self, batch_size: int) -> DataLoader:
        """Return a DataLoader that yields example inputs for benchmarking."""
        raise NotImplementedError("SplitClient.run_benchmark is not implemented")

    def get_name(self) -> str:
        """Return the name of the model."""
        model = self.get_model()
        model_hash = hash_model_architecture(model)
        return model.__class__.__name__ + "_" + model_hash[:8]

    def get_best_device(self) -> torch.device:
        """Return the best device to run the model on."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


@dataclass(frozen=True)
class PartitionTemplate:
    """Template used for replicatable partitioning across different batch sizes."""

    provider: PartitionProvider
    partitions: list[Partition]


def _get_torchgraph_from_client(client: SplitClient, batch_size: int) -> TorchGraph:
    model = client.get_model()
    model_name = model.__class__.__name__
    batch = next(iter(client.get_dataloader(batch_size)))

    if isinstance(batch, Mapping):
        gm = capture_graph(client.get_model())(**batch)
    else:
        gm = capture_graph(client.get_model())(*batch)

    tg = TorchGraph.from_fx_graph(gm, label=model_name)
    return tg


def batch_compiler(client: SplitClient, template: PartitionTemplate, batch_size: int) -> Switchboard:
    """Generate a switchboard for a given batch size based on the provided template. (Guarantees identical partitioning structure.)"""
    tg = _get_torchgraph_from_client(client, batch_size)
    return PartitionProvider(tg).create_switchboard(template.partitions, roots=template.provider)


def get_partition_template(client: SplitClient) -> PartitionTemplate:
    """Generate a template that can be reused to create identical component allocation for different batch sizes."""
    tg = _get_torchgraph_from_client(client, batch_size=1)
    pp = PartitionProvider(tg)
    ap = sorted(pp.all_partitions(), key=lambda p: -sum(len(sg.enclosed_region) for sg in p.subgraphs))
    return PartitionTemplate(provider=pp, partitions=[ap[0]])
