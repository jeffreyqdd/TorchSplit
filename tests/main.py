from collections.abc import Generator
from typing import Any, NoReturn

import torch
import torch.nn as nn

from torch_split.client import SplitClient


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

        self.seq = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            nn.ReLU(),
            self.pool,
            nn.Flatten(),
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3,
        )

    def forward(self, x):
        # nonlocal state
        x = self.pool(torch.relu(self.conv1(x)))  # (16, 16, 16)
        x = self.pool(torch.relu(self.conv2(x)))  # (32, 8, 8)
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x


class ToyExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = SimpleModel()
        self.model2 = SimpleModel()

    def forward(self, x):
        a = self.model1(x)
        b = self.model2(x)
        x = a + b
        return x


class TestInterface(SplitClient):
    def __init__(self):
        super().__init__()
        self.device = self.get_best_device()
        self.model = ToyExample()
        self.model.to(self.device)

    def get_model(self) -> torch.nn.Module:
        return self.model

    def batch_sizes(self) -> list[int]:
        return [1, 2, 4, 8, 16, 32, 64, 18, 256, 512]

    def get_example_inputs(
        self,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        example_input = torch.randn(1, 100)
        example_input.to(self.device)
        return (example_input,), {}

    def get_benchmarks(
        self, batch_size: int
    ) -> tuple[
        int, int, Generator[tuple[tuple[Any, ...], dict[str, Any]], Any, NoReturn]
    ]:
        def get_example_inputs(bs: int):
            while True:
                yield (torch.randn(bs, 100).to(self.device),), {}

        return 10, 30, get_example_inputs(batch_size)
