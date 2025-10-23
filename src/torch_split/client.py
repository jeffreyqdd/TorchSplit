from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.nn import Module

from torch_split.profiling.profiler import InstrumentedModel


class TorchSplitClient(ABC):
    @abstractmethod
    def get_example_inputs(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Return example inputs for the model."""
        pass

    @abstractmethod
    def get_model(self) -> Module:
        """Return the model."""
        raise NotImplementedError("SplitInterface.get_model is not implemented")

    @abstractmethod
    def benchmark_model(self, model: InstrumentedModel):
        pass

    @abstractmethod
    def target_device(self) -> torch.device:
        """Return the target device for the model."""
        pass
