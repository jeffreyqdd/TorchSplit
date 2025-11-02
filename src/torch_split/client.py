from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.nn import Module


class TorchSplitClient(ABC):
    def __init__(self):
        super().__init__()
        self.check_target_device()

    @abstractmethod
    def get_example_inputs(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Return example inputs for the model."""
        pass

    @abstractmethod
    def get_model(self) -> Module:
        """Return the model."""
        raise NotImplementedError("SplitInterface.get_model is not implemented")

    @abstractmethod
    def target_device(self) -> torch.device:
        """Return the target device for the model."""
        pass

    def check_target_device(self):
        """Check if the target device is available."""
        device = self.target_device()
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Target device is CUDA but CUDA is not available.")
        elif device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Target device is MPS but MPS is not available.")
        elif device.type == "cpu":
            pass  # CPU is always available
        else:
            raise RuntimeError(f"Unknown target device type: {device.type}")
