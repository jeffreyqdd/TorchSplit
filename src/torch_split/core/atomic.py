# class AtomicRegionDetector:
#     """
#     Detects indivisible subgraphs (atomic regions) from a Torch symbol graph (FX or torch.export).
#     An atomic region represents a group of ops that must remain contiguous—no cuts allowed inside.

#     Typical atomic regions:
#       - Fused ops (bias+gelu, layernorm, swish)
#       - Tensor-parallel collectives (all_gather, reduce_scatter, all_reduce)
#       - CUDA graph capture boundaries
#       - Stateful or fused nn modules
#     """

#     def __init__(self, torch_graph: TorchGraph):
#         self._graph = torch_graph
#         self._group: Dict[NodeId, int] = {}  # node_uuid -> atomic_group_id
#         self._next_gid = 0
#         self._analyze()

#     # ------------------------------------------------------------
#     # PUBLIC INTERFACE
#     # ------------------------------------------------------------
#     def atomic_group(self, node: NodeId) -> int:
#         """Return atomic group id for this node."""
#         return self._group.get(node, -1)

#     def is_same_atomic_region(self, a: NodeId, b: NodeId) -> bool:
#         """Return True if a and b belong to the same atomic group."""
#         return self.atomic_group(a) == self.atomic_group(b)

#     # ------------------------------------------------------------
#     # MAIN ANALYSIS (internal)
#     # ------------------------------------------------------------
#     def _analyze(self):
#         """
#         Walk the FX graph and assign atomic_group_id based on:
#           1. Operator fusion patterns (pattern matching)
#           2. Node metadata (nn_module_stack, fusion_group, etc.)
#           3. Tensor-parallel group detection (distributed collectives)
#         """
#         for node in self._graph.nodes_in_topological_order():
#             if self._is_collective_op(node):
#                 self._assign_new_group(node)

#             elif self._is_fused_pattern(node):
#                 self._assign_to_prev_group(node)  # e.g., GELU fused with MUL

#             elif self._is_nn_module(node):
#                 self._assign_module_scope(node)

#             else:
#                 self._assign_new_group(node)

#         self._compress_group_ids()

#     # ------------------------------------------------------------
#     # Heuristic checks (extendable!!)
#     # ------------------------------------------------------------
#     def _is_collective_op(self, node) -> bool:
#         """Detect dist.all_reduce, all_gather, reduce_scatter, etc."""
#         return node.target in (
#             torch.distributed.all_reduce,
#             torch.distributed.all_gather,
#             torch.distributed.reduce_scatter,
#         )

#     def _is_fused_pattern(self, node) -> bool:
#         """Detect ops commonly fused: bias+add, gelu, layernorm."""
#         return (
#             "gelu" in str(node.target)
#             or "layer_norm" in str(node.target)
#             or node.meta.get("fusion_group", None) is not None
#         )

#     def _is_nn_module(self, node) -> bool:
#         """Group via module scope if preserved."""
#         return "nn_module_stack" in node.meta

#     # ------------------------------------------------------------
#     # Assignment helpers
#     # ------------------------------------------------------------
#     def _assign_new_group(self, node):
#         self._group[node.uuid] = self._next_gid
#         self._next_gid += 1

#     def _assign_to_prev_group(self, node):
#         """Attach to the last group's id (fuse upward or downward)."""
#         prev = node.prev  # conceptual: previous op in topological order
#         if prev:
#             self._group[node.uuid] = self._group.get(prev.uuid, self._next_gid)
#         else:
#             self._assign_new_group(node)

#     def _assign_module_scope(self, node):
#         """Group all nodes with same module scope."""
#         key = tuple(node.meta.get("nn_module_stack", []))
#         gid = hash(key) % 1_000_000
#         self._group[node.uuid] = gid

#     def _compress_group_ids(self):
#         """Normalize group IDs from 0..N."""
#         mapping = {gid: i for i, gid in enumerate(sorted(set(self._group.values())))}
#         for k, gid in self._group.items():
#             self._group[k] = mapping[gid]
