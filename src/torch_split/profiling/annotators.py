import torch
import torch.fx as fx


def _get_device_type(obj) -> torch.device | None:
    """calculate the device type of a tensor or nested structure of tensors"""
    if torch.is_tensor(obj):
        return obj.device
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            device = _get_device_type(o)
            if device is not None:
                return device
    elif isinstance(obj, dict):
        for v in obj.values():
            device = _get_device_type(v)
            if device is not None:
                return device

    return None


class DeviceAnnotator(fx.Interpreter):
    """Propagate device information through the FX graph. The device of each node's output is stored in node.meta['torch_split_device']."""

    def __init__(self, gm: fx.GraphModule):
        super().__init__(gm)
        self._last_device = None

    def run_node(self, n):
        result = super().run_node(n)

        if new_device := _get_device_type(result):
            self._last_device = new_device

        if self._last_device is None:
            raise RuntimeError(f"Could not determine device for node {n}")

        n.meta["torch_split_device"] = self._last_device
        return result
