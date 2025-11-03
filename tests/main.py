from pathlib import Path

import torch
import torch.nn as nn

from torch_split.client import InstrumentedModule, SplitClient


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

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class Toy(nn.Module):
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
        self.model = Toy()
        self.model.to("cuda:0")
        self.model.eval()
        # self.model = Pipe(SimpleCNN, devices=[torch.device("cuda:0"), torch.device("cuda:1")], chunks=4)

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_example_inputs(
        self,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        example_input = torch.randn(1, 100).to("cuda:0")
        return (example_input,), {}

    def run_benchmark(self, module: InstrumentedModule):
        for batch_size in [1, 8, 16, 32, 64]:
            with module(batch_size) as m:
                for _ in range(30):
                    m.run(torch.randn(batch_size, 100).to("cuda:0"))


# gm = capture_graph(TestInterface())
# im = annotators.RuntimeAnnotator(gm)
# im.run(torch.randn(1, 100).to("cuda:0"))

# export_path = Path("./.bin/toy")
# pp = partition.PartitionProvider(TestInterface())
# pp.visualize_dominance(export_path)
# pp.visualize_dataflow(export_path)
# pp.create_partition()

# interpreter = profiler.InstrumentedModel(TestInterface())
# interpreter.run(torch.randn(1, 3, 32, 32))
# interpreter.run(torch.randn(1, 3, 32, 32))
# interpreter.run(torch.randn(1, 3, 32, 32))
# interpreter.run(torch.randn(1, 3, 32, 32))
