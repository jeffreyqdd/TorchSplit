import enum
import torch.nn as nn
import warnings


class HintType(enum.Enum):
    MIN_MEMORY = "torch_split_min_memory"
    """Indicates minimum memory requirement of component in bytes."""

    ATOMIC = "torch_split_atomic"
    """Indicates that the component should not be split."""


def with_hint(layer: nn.Module, hint: HintType, value=None) -> nn.Module:
    match hint:
        case HintType.MIN_MEMORY:
            if value is None:
                raise ValueError("MIN_MEMORY hint requires a value")
            if not isinstance(value, int) or value < 0:
                raise ValueError("MIN_MEMORY hint value must be a non-negative integer")
        case HintType.ATOMIC:
            if value is not None:
                warnings.warn("ATOMIC hint does not require a value; ignoring the provided value")
        case _:
            raise ValueError(f"Unknown hint type: {hint}")

    setattr(layer, hint.value, value)
    return layer
