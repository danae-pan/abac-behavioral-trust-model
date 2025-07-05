import json
import numpy as np
import matplotlib.pyplot as plt
import random
from pyfdm.TFN import TFN
from pathlib import Path

"""
trustfahp-betadistro.py

Simulates trust evolution using FAHP with category-specific Beta distributions.
Evidence for performance, reliability, and security is generated using asymmetric Beta parameters
to model skewed behavioral patterns per category.

Inputs:
- datasets/processed/all_subjects.json (with initial trust values)

Outputs:
- trust_estimator/trust_records_fahp_evidence_category_model.json: trust values over time
- trust_estimator/trust_subject_evidence_category_params.json: per-subject Beta parameters
- plots/trust_evolution_100_subjects.png: trust trajectories for 100 subjects
- plots/evidence_per_category_over_time.png: average evidence per category (line plot)
- plots/evidence_distribution_per_category.png: category-specific histograms
"""


# ----------------------
# 1. Evidence Labels, Polarity and Category Skewness
# ----------------------
evidence_labels = [
    "Average user CPU Utilization", "Average user throughput", "Average IP packet transmission delay",
    "Average IP packet delay jitter time", "Average IP packet network bandwidth occupancy",
    "Average IP packet response time", "Average user transmission rate", "Average user IP packet loss rate",
    "Average connection establishment rate", "Average connection establishment delay",
    "Average number of illegal connection by user", "Average number of critical ports scanned by user",
    "Average number of user attempts to exceed authority", "Average number of user connection failures",
    "Average number of requests including a honey attribute", "Average number of interaction with a decoy resource",
    "Average time spent on the decoy resource", "Average number of attempt modifications on the decoy resource"
]

evidence_polarity = [
    False, True, False, False, False, False,
    False, False, True, False,
    False, False, False, False, False, False, False, False
]

category_skewness = {
    "performance": "right",
    "reliability": "left",
    "security": "symmetric"
}

# ----------------------
# 2. FAHP Utilities
# ----------------------
def compute_synthetic_extents(matrix):
    """
    Computes fuzzy synthetic extent values from a fuzzy pairwise comparison matrix.
    """
    row_sums = [sum(matrix[i, :], start=TFN(0, 0, 0)) for i in range(matrix.shape[0])]
    total_sum = sum(row_sums, start=TFN(0, 0, 0))
    return [r / total_sum for r in row_sums]

def possibility_degree(m2, m1):
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
# 3. Trust Simulation 
# ----------------------

def invert_polarity(evidence, polarity_flags):
    """
    Inverts evidence dimensions where higher values indicate better behavior,
    so that all dimensions align with the intuition: higher = more suspicious.
    """
    modified = evidence.copy()
    for idx, invert in enumerate(polarity_flags):
        if invert:
            max_val = np.max(modified[:, idx])
            min_val = np.min(modified[:, idx])
            modified[:, idx] = (max_val + min_val) - modified[:, idx]
    return modified

def generate_categorized_beta_evidence(subject_params):
    """
    Generates an 18-dimensional evidence vector using category-specific Beta distributions.

    Each category (performance, reliability, security) has its own (α, β) parameters,
    and evidence is sampled accordingly.
    """
    evidence = np.zeros(18)
    for category in subject_params:
        alpha, beta = subject_params[category]
        if category == "performance":
            evidence[0:6] = np.random.beta(alpha, beta, 6)
        elif category == "reliability":
            evidence[6:10] = np.random.beta(alpha, beta, 4)
        elif category == "security":
            evidence[10:18] = np.random.beta(alpha, beta, 8)
    return np.clip(evidence, 0, 1)

def update_trust(prev_trust, evidence, weights, gamma=0.85):
    """
    Updates trust using exponential smoothing on weighted evidence scores.
    T_new = 1 - dot(weights, evidence)
    """
    T_new = 1 - np.dot(weights, evidence)
    return gamma * prev_trust + (1 - gamma) * np.clip(T_new, 0, 1)

def simulate_trust(subjects, weights, initial_trust_map, months=12, gamma=0.85):
    """Simulates trust evolution over time using FAHP and skewed Beta evidence.

    Each subject is assigned category-specific (α, β) parameters. Evidence is sampled
    monthly and used to update trust. Category-wise evidence history is recorded.
    """
    trust_records = {}
    subject_params = {}

    for subj in subjects:
        sid = subj["subject_id"]
        trust_records[sid] = [initial_trust_map.get(sid)]
        subject_params[sid] = {}

        for category in category_skewness:
            if category_skewness[category] == "left":
                alpha = round(random.uniform(3, 6), 1)
                beta = round(random.uniform(1.5, 3), 1)
            elif category_skewness[category] == "right":
                alpha = round(random.uniform(1.5, 3), 1)
                beta = round(random.uniform(3, 6), 1)
            else:
                alpha = round(random.uniform(2, 4), 1)
                beta = round(random.uniform(2, 4), 1)
            subject_params[sid][category] = (alpha, beta)

    category_evidence_history = {
        "performance": [[] for _ in range(months)],
        "reliability": [[] for _ in range(months)],
        "security": [[] for _ in range(months)]
    }

    for month_idx in range(months):
        for subj in subjects:
            sid = subj["subject_id"]
            prev = trust_records[sid][-1]
            evidence = generate_categorized_beta_evidence(subject_params[sid])
            evidence = invert_polarity(np.array([evidence]), evidence_polarity)[0]
            category_evidence_history["performance"][month_idx].extend(evidence[0:6])
            category_evidence_history["reliability"][month_idx].extend(evidence[6:10])
            category_evidence_history["security"][month_idx].extend(evidence[10:18])

            trust = update_trust(prev, evidence, weights, gamma)
            trust_records[sid].append(trust)

    return trust_records, subject_params, category_evidence_history

# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    ROOT = Path(__file__).resolve().parents[2]
    INPUT_PATH = ROOT / "datasets" / "processed" / "all_subjects.json"
    OUTPUT_DIR = ROOT / "datasets" / "processed" / "trust_estimator"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    PLOTS_DIR = ROOT / "plots"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_PATH, "r") as f:
        subjects = json.load(f)

    initial_trust_map = {s["subject_id"]: s["trust_value"] for s in subjects}

    perf = create_uniform_matrix(6)
    reli = create_uniform_matrix(4)
    sec = create_uniform_matrix(8)
    crit = create_uniform_matrix(3)

    global_weights = np.concatenate([
        compute_final_weights(compute_synthetic_extents(perf)) * compute_final_weights(compute_synthetic_extents(crit))[0],
        compute_final_weights(compute_synthetic_extents(reli)) * compute_final_weights(compute_synthetic_extents(crit))[1],
        compute_final_weights(compute_synthetic_extents(sec)) * compute_final_weights(compute_synthetic_extents(crit))[2],
    ])

    trust_records, subject_params, category_evidence_history = simulate_trust(subjects, global_weights, initial_trust_map)

    with open(OUTPUT_DIR / "trust_records_fahp_evidence_category_model.json", "w") as f:
        json.dump(trust_records, f, indent=2)

    with open(OUTPUT_DIR / "trust_subject_evidence_category_params.json", "w") as f:
        json.dump(subject_params, f, indent=2)

    final_map = {sid: records[-1] for sid, records in trust_records.items()}
    print("Final Trust Summary:")
    print(f"  Total Subjects: {len(final_map)}")
    print(f"  Avg Trust Final: {np.mean(list(final_map.values())):.4f}")

    # Trust plot for 100 subjects
    plt.figure(figsize=(10, 6))
    for i, (_, history) in enumerate(trust_records.items()):
        if i >= 100:
            break
        plt.plot(history, alpha=0.2)
    plt.title("Trust Evolution for 100 Subjects (Category-specific Evidence Modeling)")
    plt.xlabel("Month")
    plt.ylabel("Trust Value")
    plt.ylim(0, 1)
    plt.xlim(0, 12)
    plt.tight_layout()
    plot_path = PLOTS_DIR / "trust_evolution_100_subjects.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {plot_path}")


    # Evidence average per category over time
    plt.figure(figsize=(10, 6))
    for category in ["performance", "reliability", "security"]:
        avg_per_month = [np.mean(month_vals) for month_vals in category_evidence_history[category]]
        plt.plot(range(1, len(avg_per_month) + 1), avg_per_month, label=category)

    plt.title("Average Evidence per Category Over Time")
    plt.xlabel("Month")
    plt.ylabel("Average Evidence")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_path = PLOTS_DIR / "evidence_per_category_over_time.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {plot_path}")


    # Evidence distribution histogram
    plt.figure(figsize=(10, 6))
    for category in ["performance", "reliability", "security"]:
        all_vals = [val for month_vals in category_evidence_history[category] for val in month_vals]
        plt.hist(all_vals, bins=20, alpha=0.5, label=category)

    plt.title("Evidence Distribution per Category")
    plt.xlabel("Evidence Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plot_path = PLOTS_DIR / "evidence_distribution_per_category.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {plot_path}")

