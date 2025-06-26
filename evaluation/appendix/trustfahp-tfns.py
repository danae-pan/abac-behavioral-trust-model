import numpy as np
from pyfdm.TFN import TFN
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------
# 1. Helper Functions
# ----------------------
def compute_synthetic_extents(matrix):
    row_sums = [sum(matrix[i, :], start=TFN(0, 0, 0)) for i in range(matrix.shape[0])]
    total_sum = sum(row_sums, start=TFN(0, 0, 0))
    return [r / total_sum for r in row_sums]

def possibility_degree(m2: TFN, m1: TFN) -> float:
    if m2.b >= m1.b:
        return 1.0
    elif m1.a >= m2.c:
        return 0.0
    else:
        return (m1.a - m2.c) / ((m2.b - m2.c) - (m1.b - m1.a))

def compute_final_weights(synthetic_extents):
    n = len(synthetic_extents)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                V[i][j] = possibility_degree(synthetic_extents[i], synthetic_extents[j])
    d_prime = np.min(V + np.eye(n), axis=1)
    return d_prime / np.sum(d_prime)

def create_uniform_matrix(size):
    return np.array([[TFN(1, 1, 1) for _ in range(size)] for _ in range(size)])

# ----------------------
# 2. TFN Scenarios (Moderated)
# ----------------------
def create_criteria_matrix(scenario):
    if scenario == "performance":
        return np.array([
            [TFN(1, 1, 1), TFN(2, 3, 4), TFN(1, 2, 3)],
            [TFN(1/4, 1/3, 1/2), TFN(1, 1, 1), TFN(1, 2, 3)],
            [TFN(1/3, 1/2, 1), TFN(1/3, 1/2, 1), TFN(1, 1, 1)]
        ])
    elif scenario == "security":
        return np.array([
            [TFN(1, 1, 1), TFN(1/4, 1/3, 1/2), TFN(1/3, 1/2, 1)],
            [TFN(2, 3, 4), TFN(1, 1, 1), TFN(1/3, 1/2, 1)],
            [TFN(1, 2, 3), TFN(1, 2, 3), TFN(1, 1, 1)]
        ])
    else:
        return create_uniform_matrix(3)

# ----------------------
# 3. Execution
# ----------------------
scenarios = ["performance", "balanced", "security"]
category_indices = {
    "performance": list(range(0, 6)),
    "reliability": list(range(6, 10)),
    "security": list(range(10, 18))
}
global_weights_by_scenario = {}

for scenario in scenarios:
    perf = create_uniform_matrix(6)
    reli = create_uniform_matrix(4)
    sec = create_uniform_matrix(8)
    crit = create_criteria_matrix(scenario)

    crit_weights = compute_final_weights(compute_synthetic_extents(crit))
    print(f"Scenario: {scenario.capitalize()} — FAHP Criteria Weights: {np.round(crit_weights, 3)}")

    weights = np.concatenate([
        compute_final_weights(compute_synthetic_extents(perf)) * crit_weights[0],
        compute_final_weights(compute_synthetic_extents(reli)) * crit_weights[1],
        compute_final_weights(compute_synthetic_extents(sec)) * crit_weights[2],
    ])

    grouped = {
        "performance": sum(weights[category_indices["performance"]]),
        "reliability": sum(weights[category_indices["reliability"]]),
        "security": sum(weights[category_indices["security"]])
    }
    global_weights_by_scenario[scenario] = grouped

# ----------------------
# 4. Plotting
# ----------------------
labels = ["performance", "reliability", "security"]
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
for i, scenario in enumerate(scenarios):
    values = [global_weights_by_scenario[scenario][cat] for cat in labels]
    ax.bar(x + i * width, values, width, label=scenario.capitalize())

ax.set_ylabel("Weight Sum per Category")
ax.set_title("Global Category Weight Comparison Across Criteria Prioritization")
ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.show()
