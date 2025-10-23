import torch
import torch.nn as nn

from torch_split.client import TorchSplitClient
from torch_split.profiling import profiler


def with_hint(x):
    setattr(x, "_hello_world", "foo bar")
    return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # for 32×32 input
        self.fc2 = nn.Linear(128, 10)  # 10 classes
        self.fc3 = with_hint(nn.Linear(10, 10))  # 10 classes
        # p_fc3_weight
        # p_fc_bias

    def forward(self, x):
        # nonlocal state
        x = self.pool(torch.relu(self.conv1(x)))  # (16, 16, 16)
        x = self.pool(torch.relu(self.conv2(x)))  # (32, 8, 8)
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x


# Google OR Tools
class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = SimpleCNN()
        self.cnn2 = SimpleCNN()

    def forward(self, x):
        # split = torch.tensor(True)
        # result = torch.cond(split, lambda x: self.cnn1(x), lambda x: self.cnn1(x), (x,))
        a = self.cnn1(x)
        b = self.cnn2(x)
        x = a + b
        return x


class TestInterface(TorchSplitClient):
    def __init__(self):
        super().__init__()
        self.model = Toy()

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_example_inputs(self) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        example_input = torch.randn(1, 3, 32, 32)
        return (example_input,), {}

    def benchmark_model(self, model: profiler.InstrumentedModel):
        raise NotImplementedError

    def target_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# export_path = Path("./.bin/toy")
# pp = partition.PartitionProvider(TestInterface())
# pp.visualize_dominance(export_path)
# pp.visualize_dataflow(export_path)
# pp.create_partition()

interpreter = profiler.InstrumentedModel(TestInterface())
interpreter.run(torch.randn(1, 3, 32, 32))
interpreter.run(torch.randn(1, 3, 32, 32))
interpreter.run(torch.randn(1, 3, 32, 32))
interpreter.run(torch.randn(1, 3, 32, 32))
