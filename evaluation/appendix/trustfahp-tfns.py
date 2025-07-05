import numpy as np
from pyfdm.TFN import TFN
import matplotlib.pyplot as plt
from pathlib import Path

"""
trustfahp-tfns.py

Compares the impact of different outer-layer FAHP criteria matrices on the final 
global category weights (performance, reliability, security). Each scenario (performance-
dominant, security-dominant, balanced) affects how much weight each evidence category 
receives in the final trust computation.

Outputs:
- plots/global_category_weight_comparison.png: bar plot comparing category weights across scenarios
"""

# ----------------------
# 1. FAHP Utilities
# ----------------------
def compute_synthetic_extents(matrix):
    """
    Computes fuzzy synthetic extent values from a fuzzy pairwise comparison matrix.
    """
    row_sums = [sum(matrix[i, :], start=TFN(0, 0, 0)) for i in range(matrix.shape[0])]
    total_sum = sum(row_sums, start=TFN(0, 0, 0))
    return [r / total_sum for r in row_sums]

def possibility_degree(m2: TFN, m1: TFN) -> float:
    """
    Computes the possibility degree that fuzzy number m2 ≥ m1.
    """
    if m2.b >= m1.b:
        return 1.0
    elif m1.a >= m2.c:
        return 0.0
    else:
        return (m1.a - m2.c) / ((m2.b - m2.c) - (m1.b - m1.a))

def compute_final_weights(synthetic_extents):
    """
    Computes crisp weights from fuzzy synthetic extents using the minimum possibility method.
    """
    n = len(synthetic_extents)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                V[i][j] = possibility_degree(synthetic_extents[i], synthetic_extents[j])
    d_prime = np.min(V + np.eye(n), axis=1)
    return d_prime / np.sum(d_prime)

def create_uniform_matrix(size):
    """
    Creates a size×size identity fuzzy matrix filled with TFN(1,1,1).
    Used to generate equal weights when no specific pairwise comparisons are provided.
    """
    return np.array([[TFN(1, 1, 1) for _ in range(size)] for _ in range(size)])

# ----------------------
# 2. FAHP Criteria Scenarios
# ----------------------
def create_criteria_matrix(scenario):
    """
    Define outer-layer FAHP criteria matrices for each scenario.
    Returns a 3x3 TFN matrix prioritizing one category.
    """
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
# 3. Weight Computation
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

ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
plot_path = PLOTS_DIR / "global_category_weight_comparison.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to: {plot_path}")
