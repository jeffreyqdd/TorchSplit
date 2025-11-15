from torch_split.core.partition import PartitionProvider

Forest = tuple[
    "PartitionProvider",
    dict[
        "PartitionProvider.PartitionCandidate",
        set["PartitionProvider.PartitionCandidate"],
    ],
]


def solve(partition_providers: list[PartitionProvider]):
    forest: list[Forest] = [pp.generate_tree() for pp in partition_providers]
    # strategy: str = "branch_parallel",  # future: "hybrid", "pipeline"
    # top_k: int = 1,
    # alpha: float = 1.0,  # compute imbalance weight
    # beta: float = 1.0,  # communication weight
    # gamma: float = 1000.0,  # memory violation penalty
    # beta_tp: float = 10.0,  # TP disruption penalty (if used)
    pass
