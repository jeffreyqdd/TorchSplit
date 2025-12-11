from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, NoReturn

import torch
import torch.nn as nn


class SplitClient(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def batch_sizes(self) -> list[int]:
        """Return a list of batch sizes to benchmark."""
        raise NotImplementedError("SplitClient.batch_sizes is not implemented")

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the model."""
        raise NotImplementedError("SplitClient.get_model is not implemented")

    @abstractmethod
    def get_benchmarks(
        self, batch_size: int
    ) -> tuple[int, int, Generator[tuple[tuple[Any, ...], dict[str, Any]], Any, NoReturn]]:
        """Run the model with the given batch size and return the outputs.

        Returns a tuple of (warmup runs, [args, kwargs])
        """
        raise NotImplementedError("SplitClient.run_benchmark is not implemented")

    def get_best_device(self) -> torch.device:
        """Return the best device to run the model on."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
