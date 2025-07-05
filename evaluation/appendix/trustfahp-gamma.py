from pyfdm.TFN import TFN
import numpy as np
import json
import matplotlib.pyplot as plt
import random
from pathlib import Path

"""
trustfahp-gamma.py

Simulates the impact of the γ (memory) parameter on trust evolution using FAHP.
Each subject receives behavior sampled from a Beta distribution.
The experiment compares average trust over time across different γ values.

Inputs:
- datasets/processed/all_subjects.json (with initial trust values)

Outputs:
- trust_estimator/trust_records_random_beta_gamma_*.json: trust timelines for each γ
- plots/trust_over_time_gamma_comparison.png: line plot of average trust per month
"""

# ----------------------
# 1. Evidence Labels and Polarity
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

# -------------------------------
# 2. FAHP Utilities
# -------------------------------
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

def create_uniform_matrix(size: int):
    """
    Creates a size×size identity fuzzy matrix filled with TFN(1,1,1).
    Used to generate equal weights when no specific pairwise comparisons are provided.
    """
    return np.array([[TFN(1, 1, 1) for _ in range(size)] for _ in range(size)])

# -------------------------------
# 3. Trust Simulation
# -------------------------------
def invert_polarity(evidence: np.ndarray, polarity_flags: list) -> np.ndarray:
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

def generate_beta_evidence_custom(alpha, beta):
    """
    Generates a vector of 18 behavioral evidence values using a Beta distribution.
    Values are clipped to [0,1].
    """
    return np.clip(np.random.beta(alpha, beta, 18), 0, 1)

def update_trust(prev_trust, evidence, weights, gamma=0.85):
    """
    Updates trust using exponential smoothing on weighted evidence scores.
    T_new = 1 - dot(weights, evidence)
    """
    T_new = 1 - np.dot(weights, evidence)
    return gamma * prev_trust + (1 - gamma) * np.clip(T_new, 0, 1)

def simulate_trust(subjects, global_weights, initial_trust_map, months=12, gamma=0.85):
    """
    Simulates monthly trust evolution for a list of subjects.
    Each subject is assigned random (α, β) parameters for evidence generation.
    """
    trust_records = {}
    subject_params = {}

    for subj in subjects:
        sid = subj["subject_id"]
        trust_records[sid] = [initial_trust_map.get(sid)]
        subject_params[sid] = (round(random.uniform(1.5, 6), 1), round(random.uniform(1.5, 6), 1))

    for _ in range(months):
        for subj in subjects:
            sid = subj["subject_id"]
            prev_trust = trust_records[sid][-1]
            alpha, beta = subject_params[sid]
            evidence = generate_beta_evidence_custom(alpha, beta)
            evidence = invert_polarity(np.array([evidence]), evidence_polarity)[0]
            trust = update_trust(prev_trust, evidence, global_weights, gamma)
            trust_records[sid].append(trust)

    return trust_records, subject_params

# -------------------- ENTRY POINT --------------------

if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    ROOT = Path(__file__).resolve().parents[2]
    INPUT_PATH = ROOT / "datasets" / "processed" / "all_subjects.json"
    OUTPUT_DIR = ROOT / "datasets" / "processed" / "trust_estimator"
    PLOTS_DIR = ROOT / "plots"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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

    gamma_values = [0.2, 0.8]
    months = 12
    all_results = {}

    plt.figure(figsize=(10, 6))

    for gamma in gamma_values:
        trust_records, subject_params = simulate_trust(subjects, global_weights, initial_trust_map, months, gamma)
        all_results[gamma] = trust_records

        out_path = OUTPUT_DIR / f"trust_records_random_beta_gamma_{str(gamma).replace('.', '_')}.json"
        with open(out_path, "w") as f:
            json.dump(trust_records, f, indent=2)

        avg_trust = np.mean([v for v in zip(*trust_records.values())], axis=1)
        plt.plot(avg_trust, label=f"γ = {gamma}")

    plt.title("Average Trust Over Time for Different γ")
    plt.xlabel("Month")
    plt.ylabel("Average Trust")
    plt.xlim(0, months)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / "trust_over_time_gamma_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[+] Saved trust evolution plot to: {plot_path}")
