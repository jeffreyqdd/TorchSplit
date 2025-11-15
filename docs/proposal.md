# TorchSplit: Automated Model Partitioning and MIG-Aware Multi-Model Scheduling

## Project selection

Design and implement an end-to-end system that automatically partitions PyTorch models into communication-aware regions and schedules multiple models on MIG-partitioned GPUs to maximize utilization while meeting per-model latency/throughput targets. The core contribution is a solver-backed planner that chooses: (1) partition boundaries, (2) MIG layouts per GPU, and (3) capacity shares per model per MIG slice, with explicit penalties or constraints to co-locate large intermediate tensors and respect memory budgets.

Why this project
- Practical: Consolidates multiple inference workloads on shared accelerators using NVIDIA MIG while controlling tail latency and network overhead.
- Novelty: Combines dominance/SESE region analysis with leximin fairness and co-location-aware ILP that supports multi-model sharing on MIG slices.
- Impact: Increases GPU utilization and reduces p50/p95 latency for multi-tenant inference while simplifying operator effort.

## Background and scope (your codebase today)

TorchSplit already provides key building blocks:

- Graph IR and dominance
  - `torch_split.core.ir.TorchGraph`: FX-based dataflow IR with inputs/outputs, parameter edges, and rendering.
  - `torch_split.core.dominance.DominanceInformation`: ENTRY/EXIT augmentation, dominance/post-dominance sets, trees, and region reachability utilities.

- Partition discovery and evaluation
  - `torch_split.core.partition.PartitionProvider`: Constructs SESE regions as candidate partitions, groups enclosed nodes into disjoint subgraphs, and estimates per-batch execution time by aggregating node-level profiles.

- Profiling & annotations
  - `torch_split.profiling.annotators.DeviceAnnotator`: Propagates device placement per node.
  - `torch_split.profiling.annotators.RuntimeAnnotator`: Per-node timing, output size, and peak memory stats; aggregates avg/std/min/max per batch size.

- Tracing and export
  - `src/torch_split/tracing/*`: Skeletons for OpenTelemetry exporters and schemas to persist traces for later analysis and visualization.

- MIG-aware solver (extended here)
  - `torch_split.core.solvers.ilp.solve_leximin`: Original leximin solver with exclusive assignment to MIG instances.
  - `torch_split.core.solvers.ilp.solve_leximin_shared`: New variant supporting fractional sharing across MIG slices and co-location preferences (hard and soft). Decision variables include:
    - `y[node, layout] ∈ {0,1}`: choose one MIG layout per GPU.
    - `x[model, node, c] ≥ 0`: (fractional) count of c-GB MIG instances on node assigned to model.
    - `a[model, node] ∈ {0,1}`: placement indicator linking any usage on a node.
    - Throughput `T_m = Σ_{n,c} x[m,n,c] * throughput[m][c]`.
    - Leximin objective `max Z` with `Z ≤ T_m` for unlocked models; optional penalty for separating large-traffic pairs.

Scope for the project
- Build a full pipeline: trace → profile → partition candidates → solver → deploy-ready plan.
- Support K GPUs with heterogeneous MIG layouts and N models, each with per-batch throughput, latency, and memory profiles.
- Optimize for fairness (leximin) or weighted objectives while respecting capacity, memory, and co-location.

## Implementation plan

Phase 1 — Profiling and data plumbing
- Complete `RuntimeAnnotator` integration with `SplitClient` workloads to collect per-node metrics across a user-provided batch size grid.
- Extend summaries with per-node peak memory and output tensor size; propagate to `TorchGraph` via `annotate_with_profiling_data`.
- Add a simple artifact writer (JSON/msgpack) keyed by model architecture hash (`utils.hash_model_architecture`).

Deliverables
- CLI: `torchsplit trace` → produce FX + profiling blobs.
- Docs: how to implement `SplitClient` and run profiling.

Phase 2 — Partition candidate generation and ranking
- In `PartitionProvider`:
  - Finish candidate generation using dominance (SESE) enumeration already scaffolded.
  - Rank by compute-share and prune by intersection/subsumption (laminar family), leveraging existing helpers.
  - Augment partition cost with: (a) estimated network traffic (`network_traffic`), (b) peak memory, and (c) sensitivity to batch size.

Deliverables
- API: `solve_partitioning_problem` returns top-K candidates with metrics.
- Graphviz outputs for dataflow and dominance sets for debugging.

Phase 3 — MIG-aware multi-model allocation
- Finalize `solve_leximin_shared` with the following constraints:
  - Capacity: `Σ_m x[m,n,c] ≤ count_c(layout_n)` for each (node, c-GB slice).
  - Memory: For each model m on (n,c), ensure peak mem per instance ≤ c-GB (or add a violation penalty `γ`).
  - Latency/Throughput SLAs: per-model min throughput or max latency using profiled Tput/latency vs batch size curves.
  - Co-location: (hard) require certain pairs on the same GPU; (soft) penalize separation proportional to expected activation size.
  - Integer vs fractional: toggle sharing via vtype; rounding and pack/repair step to obtain deployable integer allocations.
- Objective: leximin on per-model throughput (default) with secondary penalties for separation and memory violations.

Deliverables
- Solver API with inputs: `{models, nodes, valid_layouts, throughput, mem, latency_targets, colocation_pairs}`.
- Unit tests on small toy instances (provided in `tests/`).

Phase 4 — Plan materialization and deployment hooks
- Emit per-node MIG layout, per-model instance counts per config, and placement hints.
- Optional: emit `nvidia-smi mig -cgi` command stubs or orchestrator JSON for cluster integration.

Deliverables
- `torchsplit plan` CLI subcommand: reads profiles and produces a deployment plan file.

Phase 5 — Evaluation
- Benchmarks: multi-model mixes (e.g., CLIP, BERT-large, FastPitch, OCR), varied batch sizes, and two MIG regimes.
- Metrics: (a) aggregate throughput, (b) p50/p95 per-model latency, (c) network bytes across boundaries, (d) GPU utilization.
- Ablations: exclusive vs shared MIG; co-location off/on; different fairness weights.

## Expected results
- Higher GPU utilization via shared MIG partitions and fractional allocation compared to exclusive placement.
- Improved fairness across heterogeneous models with leximin objective (no model starves).
- Reduced cross-GPU traffic and latency for models with large intermediate tensors when co-location is enabled.
- Deployable plans that respect per-slice memory and chosen MIG layouts with minimal fragmentation.

## Expected challenges
- Accurate profiling under variability (warmup, cache, CUDA graphs, stochastic layers).
- Generalizing profiles across batch sizes and avoiding overfitting (need smoothing/fit curves).
- Correct memory accounting vs MIG slice limits (peak vs steady-state, activation checkpointing).
- Solver rounding (fractional → integer) without violating SLAs; may require repair heuristics.
- MIG orchestration/permissioning in multi-user clusters; reconfiguration latency and fragmentation.
- Handling dynamic control flow (cond/higher-order ops) and shape polymorphism (dynamo export constraints).
- Gurobi licensing/limits; consider fallback MILP/CP-SAT alternatives if needed.

## Expected GPU/TPU hours per person
Estimates for a 2–3 person team over ~6–8 weeks:
- Profiling + data collection: 15–30 GPU hours/person (multiple batch sizes × 3–5 models × repeats).
- Solver development + unit tests: 5–10 GPU hours/person (toy models, small sanity runs; mostly CPU-bound).
- End-to-end experiments: 30–60 GPU hours/person (MIG reconfigs × model mixes × ablations).
- Optional TPU replication (if applicable): 10–20 TPU hours/person (feature parity subset).

These are conservative ranges; exact hours depend on the size of the model zoo and the number of ablations.

## Timeline (indicative)
- Week 1–2: Profiling pipeline and data artifacts; finish SESE enumeration and ranking.
- Week 3–4: MIG-aware solver with co-location and memory/latency constraints; CLI integration.
- Week 5–6: Evaluation on 1–4 GPUs with two MIG regimes; ablations and visualization.
- Week 7–8: Polish, docs, and optional TPU/alternative solver exploration.

## Risks and mitigations
- Solver feasibility: Keep integer problem sizes modest; enable sharing/continuous relaxations + repair.
- Profiling noise: Use medians/percentiles, warmups, and multiple seeds; cache artifacts by model hash.
- MIG fragmentation: Provide alternative layouts per node and allow solver to choose.
- Heterogeneous nodes: Parameterize per-node capabilities and configs; relax assumptions.

## References to code components
- IR: `torch_split.core.ir.TorchGraph`
- Dominance/regions: `torch_split.core.dominance.DominanceInformation`, `torch_split.core.partition.PartitionProvider`
- Profiling: `torch_split.profiling.annotators.DeviceAnnotator`, `RuntimeAnnotator`
- Solver: `torch_split.core.solvers.ilp.solve_leximin`, `solve_leximin_shared`
- Tracing/export: `src/torch_split/tracing/*`
