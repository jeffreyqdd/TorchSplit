from pathlib import Path

import torch
import torch.nn as nn

from torch_split.client import TorchSplitClient
from torch_split.core import partition


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # for 32×32 input
        self.fc2 = nn.Linear(128, 10)  # 10 classes
        self.fc3 = nn.Linear(10, 10)  # 10 classes
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
        split = torch.tensor(True)
        result = torch.cond(split, lambda x: self.cnn1(x), lambda x: self.cnn1(x), (x,))
        return result


class TestInterface(TorchSplitClient):
    def __init__(self):
        super().__init__()
        self.model = Toy()

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_example_inputs(self) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        example_input = torch.randn(1, 3, 32, 32)
        return (example_input,), {}


# x = TorchGraph.from_split_interface(TestInterface())
# print(x.to_bytes().getvalue())

# import time

# start = time.perf_counter()
# tgraph = ir.TorchGraph.from_split_interface(ti)
pp = partition.PartitionProvider(TestInterface())
# pp.visualize_dominance(Path("/Users/jeffreyqian/Downloads/dominance"))
pp.visualize_dataflow(Path("/Users/jeffreyqian/Downloads/dominance"), True)
# partition.DominanceInformation.from_torch_graph(tgraph)
# recomputed = tgraph.test()

# x = torch.randn(1, 3, 32, 32)
# print(recomputed(x))
# print(ti.get_model()(x))
# print(torch.all(recomputed(x) == ti.get_model()(x)))
# print(graph.code)

# start = time.perf_counter()
# buffer = io.BytesIO()
# ir._write_graph_to_buffer(graph, buffer)
# print("Serialized graph in", (time.perf_counter() - start) * 1000, "ms")

# start = time.perf_counter()
# new_graph = ir._read_graph_from_buffer(buffer)
# print("Deserialized graph in", (time.perf_counter() - start) * 1000, "ms")

# print(new_graph.code)
# print(new_graph(torch.randn(1, 3, 32, 32)))

# torch.save(graph.graph, "graph_only.pth")
# dot = graphviz.Digraph(name="ToyCnn")
# tgraph.render_graph(dot)
# dot.render("example_split_xyz")
