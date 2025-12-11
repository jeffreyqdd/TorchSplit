# import uuid
# import torch
# from torch_split.lib.partition.provider import PartitionProvider

# Candidate = PartitionProvider.PartitionCandidate
# Forest = tuple[
#     "PartitionProvider",
#     dict[
#         "PartitionProvider.PartitionCandidate",
#         set["PartitionProvider.PartitionCandidate"],
#     ],
# ]


# def expand(node: Candidate, edges: dict[Candidate, set[Candidate]]):
#     print(
#         f"Expanding node that encloses {node.partition.cut.source} to {node.partition.cut.sink}"
#     )
#     for children in edges.get(node, []):
#         print(
#             f"  Child node that encloses {children.partition.cut.source} to {children.partition.cut.sink}"
#         )
#         expand(children, edges)


# def _log_tree(node: Candidate, edges: dict[Candidate, set[Candidate]], depth: int = 0):
#     print(
#         " " * depth,
#         f"({depth}) {node.partition.cut.source} -> {node.partition.cut.sink}",
#     )
#     for child in edges.get(node, []):
#         _log_tree(child, edges, depth + 1)


# def solve(partition_providers: list[PartitionProvider]):
#     pass

#     assert len(partition_providers) == 1, "only 1 allowed for now"
#     for pp in partition_providers:
#         pp, roots, tree = pp.generate_tree()
#         assert len(roots) == 1, "Expected a single root for the dominance tree"

#         # expand(roots[0], tree)
#         _log_tree(roots[0], tree)

#         # # Get successors of source (actual entry nodes) and predecessors of sink (actual exit nodes)
#         # entry_nodes = pp._dominance.successors_of(source)
#         # exit_nodes = pp._dominance.predecessors_of(sink)

#         # print(f"Exporting full model from {entry_nodes} to {exit_nodes}")
#         # gm = pp.export(entry_nodes, exit_nodes)

#         # # Save with metadata preservation - only keep picklable metadata
#         # # Filter out unpicklable items (weak references, etc.)
#         # def filter_metadata(meta_dict):
#         #     filtered = {}
#         #     for key, value in meta_dict.items():
#         #         try:
#         #             # Try to pickle the value to see if it's picklable
#         #             import pickle

#         #             pickle.dumps(value)
#         #             filtered[key] = value
#         #         except (TypeError, AttributeError, pickle.PicklingError):
#         #             # Skip unpicklable values
#         #             pass
#         #     return filtered

#         # metadata = {node.name: filter_metadata(node.meta) for node in gm.graph.nodes}
#         # save_dict = {"graph_module": gm, "metadata": metadata}
#         # torch.save(save_dict, "model_module.pt")
#         # return
