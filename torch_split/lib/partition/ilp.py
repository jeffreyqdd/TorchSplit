import gurobipy as gp
from gurobipy import GRB

# # Define models, configs, and nodes
# # ================ Pipeline 1 ================

# models = ['AC', 'B', 'D']
# configs = [6, 12, 24]
# # nodes = ['GPU0', 'GPU1', 'GPU2', 'GPU3', 'GPU4', 'GPU5', 'GPU6', 'GPU7']

# # Throughput dictionary
# throughput = {
#     'AC': {6: 200, 12: 240, 24: 270},
#     'B': {24: 45},
#     'D': {6: 55, 12: 55, 24: 70},
# }

# ================ Pipeline 2 ================
# A: audio recognition
# B: encoder + search doc
# C: text check  (bart-larger)
# D: language detection  (roberta-large)
# E: text to speech (fastpitch)
models = ["A", "B", "C", "E"]
configs = [6, 12, 24]
nodes = ["GPU0", "GPU1", "GPU2", "GPU3"]  # , 'GPU4', 'GPU5', 'GPU6']

# Throughput dictionary
throughput = {
    "A": {12: 71, 24: 125},
    "B": {6: 5333, 12: 6083, 24: 7555},
    "C": {6: 26, 12: 45, 24: 92},
    # 'D': {6: 95, 12: 172, 24: 332},
    "E": {12: 3.9, 24: 4.82},
}

valid_layouts = [
    [24],
    [12, 12],
    [12, 6, 6],
    [6, 6, 6, 6],
]


def solve_leximin(locked_lower_bounds):
    m = gp.Model("Leximin_Level")
    x, y = {}, {}
    T_m = {}
    Z = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z")

    for model in models:
        for node in nodes:
            for c in configs:
                if c in throughput[model]:
                    x[model, node, c] = m.addVar(
                        vtype=GRB.INTEGER, name=f"x_{model}_{node}_{c}"
                    )

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

        m.addConstr(
            gp.quicksum(
                x[model, node, c]
                for node in nodes
                for c in configs
                if (model, node, c) in x
            )
            >= 1
        )

    for node in nodes:
        m.addConstr(gp.quicksum(y[node, lid] for lid in range(len(valid_layouts))) == 1)
        for c in configs:
            m.addConstr(
                gp.quicksum(
                    x[model, node, c] for model in models if (model, node, c) in x
                )
                <= gp.quicksum(
                    y[node, lid] * layout.count(c)
                    for lid, layout in enumerate(valid_layouts)
                )
            )

    m.setObjective(Z, GRB.MAXIMIZE)
    m.setParam("OutputFlag", 0)
    m.optimize()

    throughput_vals = {model: T_m[model].X for model in models}
    assignment = {
        (model, node, c): int(x[model, node, c].X)
        for model in models
        for node in nodes
        for c in configs
        if (model, node, c) in x and x[model, node, c].X > 0.5
    }
    return throughput_vals, assignment


# Leximin loop
locked = {}
for _ in range(len(models)):
    T_vals, assignment = solve_leximin(locked)
    unlocked = [m for m in models if m not in locked]
    if not unlocked:
        break
    min_model = min(unlocked, key=lambda m: T_vals[m])
    locked[min_model] = T_vals[min_model]

# Print results
print(f"\nPipeline Throughput: {min(locked.values()):.2f}")
print(" Leximin Model Throughputs:")
for model in models:
    print(f" {model}: {locked[model]:.2f}")


print("\nAssignments:")
for (model, node, c), count in assignment.items():
    print(f" - Model {model} assigned {count}x to {node} with MIG {c}GB")
