import gurobipy as gp
import numpy as np
from gurobipy import GRB


class Solver:
    def __init__(self, num_gpus: int, memory_slices: list[int], profile_result: dict):
        self.num_gpus = num_gpus
        self.memory_slices = memory_slices
        self.profile_result = profile_result

    def solve_leximin_for_batch(self, batch_size: int):
        throughput: dict = {}
        utilization: dict = {}
        for memory_restriction in self.memory_slices:
            if str(memory_restriction) not in self.profile_result:
                continue
            if str(batch_size) not in self.profile_result[str(memory_restriction)]:
                continue

            for name, data in self.profile_result[str(memory_restriction)][str(batch_size)].items():
                if name not in throughput:
                    throughput[name] = {}
                if name not in utilization:
                    utilization[name] = {}

                throughput[name][memory_restriction] = 1000.0 / np.mean(data["elapsed_time_ms"]) * batch_size
                utilization[name][memory_restriction] = max(
                    10, int(data["utilization_pct"])
                )  # set floor at 10% bc nvml does not want to give me true utilization

        valid_layouts: list[list[int]] = self.multistep_combinations(self.memory_slices, max(self.memory_slices))

        models = list(throughput.keys())
        configs = self.memory_slices
        nodes = list(range(self.num_gpus))

        UTILIZATION_LIMIT = 100.0
        PENALTY_WEIGHT = 1000.0

        def solve_step(locked_lower_bounds):
            m = gp.Model("Leximin_Level")
            x, y = {}, {}
            T_m = {}
            Z = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z")

            over_util = {}
            for node in nodes:
                over_util[node] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"over_util_{node}")

            for model in models:
                for node in nodes:
                    for c in configs:
                        if c in throughput.get(model, {}):
                            x[model, node, c] = m.addVar(vtype=GRB.INTEGER, name=f"x_{model}_{node}_{c}")

            for node in nodes:
                for lid, layout in enumerate(valid_layouts):
                    y[node, lid] = m.addVar(vtype=GRB.BINARY, name=f"y_{node}_{lid}")

            for model in models:
                T_m[model] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"T_{model}")
                m.addConstr(
                    T_m[model]
                    == gp.quicksum(
                        x[model, node, c] * throughput[model][c]
                        for node in nodes
                        for c in configs
                        if (model, node, c) in x
                    )
                )

                if model in locked_lower_bounds:
                    m.addConstr(T_m[model] >= locked_lower_bounds[model])
                else:
                    m.addConstr(Z <= T_m[model])

            for model in models:
                m.addConstr(
                    gp.quicksum(x[model, node, c] for node in nodes for c in configs if (model, node, c) in x) >= 1
                )

            for node in nodes:
                m.addConstr(gp.quicksum(y[node, lid] for lid in range(len(valid_layouts))) == 1)

            for node in nodes:
                for c in configs:
                    m.addConstr(
                        gp.quicksum(x[model, node, c] for model in models if (model, node, c) in x)
                        <= gp.quicksum(y[node, lid] * layout.count(c) for lid, layout in enumerate(valid_layouts))
                    )

            for node in nodes:
                node_utilization_expr = gp.quicksum(
                    x[model, node, c] * utilization.get(model, {}).get(c, 0)
                    for model in models
                    for c in configs
                    if (model, node, c) in x
                )
                m.addConstr(node_utilization_expr <= UTILIZATION_LIMIT + over_util[node])

            m.setObjective(Z - PENALTY_WEIGHT * gp.quicksum(over_util[node] for node in nodes), GRB.MAXIMIZE)
            m.setParam("OutputFlag", 0)
            m.optimize()

            if m.Status != GRB.OPTIMAL:
                return {}, {}

            throughput_vals = {model: T_m[model].X for model in models}
            assignment = {
                (model, node, c): int(x[model, node, c].X)
                for model in models
                for node in nodes
                for c in configs
                if (model, node, c) in x and x[model, node, c].X > 0.5
            }
            return throughput_vals, assignment

        locked = {}
        final_assignment = {}
        for _ in range(len(models)):
            T_vals, assignment = solve_step(locked)
            if not T_vals:
                break
            final_assignment = assignment
            unlocked = [m for m in models if m not in locked]
            if not unlocked:
                break
            min_model = min(unlocked, key=lambda m: T_vals[m])
            locked[min_model] = T_vals[min_model]

        return locked, final_assignment, utilization

    def multistep_combinations(self, items, K):
        """return unique multiset combinations that sum to less than or equal to K"""
        items = sorted(items)
        results = []

        def backtrack(start, current, current_sum):
            if current_sum <= K and current:
                results.append(current.copy())

            for i in range(start, len(items)):
                val = items[i]
                if current_sum + val > K:
                    break

                current.append(val)
                backtrack(i, current, current_sum + val)  # reuse allowed
                current.pop()

        backtrack(0, [], 0)
        return results
