from pyfdm.TFN import TFN
import numpy as np
import json
import matplotlib.pyplot as plt
import random
from pathlib import Path

# Labels and polarity flags for each behavioral evidence dimension
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

def create_dummy_fuzzy_matrix(size: int):
    """
    Creates a size×size identity fuzzy matrix filled with TFN(1,1,1).
    Used to generate equal weights when no specific pairwise comparisons are provided.
    """
    return np.array([[TFN(1, 1, 1) for _ in range(size)] for _ in range(size)])

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

    ROOT = Path(__file__).resolve().parents[1]
    SUBJECTS_PATH = ROOT / "datasets" / "processed" / "all_subjects.json"
    OUTPUT_DIR = ROOT / "datasets" / "processed" / "trust_estimator"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SUBJECTS_PATH, encoding="utf-8") as f:
        subjects = json.load(f)

    initial_trust_map = {s["subject_id"]: s["trust_value"] for s in subjects}

    # Create default TFN matrices
    perf = create_dummy_fuzzy_matrix(6)
    reli = create_dummy_fuzzy_matrix(4)
    sec = create_dummy_fuzzy_matrix(8)
    crit = create_dummy_fuzzy_matrix(3)

    global_weights = np.concatenate([
        compute_final_weights(compute_synthetic_extents(perf)) * compute_final_weights(compute_synthetic_extents(crit))[0],
        compute_final_weights(compute_synthetic_extents(reli)) * compute_final_weights(compute_synthetic_extents(crit))[1],
        compute_final_weights(compute_synthetic_extents(sec)) * compute_final_weights(compute_synthetic_extents(crit))[2],
    ])

    trust_records, subject_params = simulate_trust(subjects, global_weights, initial_trust_map)

    with open(OUTPUT_DIR / "trust_records_fahp_random_params.json", "w") as f:
        json.dump(trust_records, f, indent=2)

    with open(OUTPUT_DIR / "trust_subject_beta_params.json", "w") as f:
        json.dump(subject_params, f, indent=2)

    final_map = {sid: records[-1] for sid, records in trust_records.items()}
    increased = [v for sid, v in final_map.items() if v > initial_trust_map.get(sid)]
    decreased = [v for sid, v in final_map.items() if v < initial_trust_map.get(sid)]

    print("Final Trust Summary:")
    print(f"  Total Subjects:         {len(final_map)}")
    print(f"  Trust Increased:      {len(increased)}")
    print(f"  Trust Decreased:      {len(decreased)}")
    print(f"  Avg Final Trust (All):  {np.mean(list(final_map.values())):.4f}")
    print(f"  Avg Increased Trust:            {np.mean(increased) if increased else 0:.4f}")
    print(f"  Avg Decreased Trust:            {np.mean(decreased) if decreased else 0:.4f}")

    # Plot trust evolution visualisation - optional
    plt.figure(figsize=(10, 6))
    for i, (sid, history) in enumerate(trust_records.items()):
        if i >= 100:
            break
        plt.plot(history, alpha=0.2)
    plt.title("Trust Evolution for 100 Subjects (Random α, β between [1.5, 6])")
    plt.xlabel("Month")
    plt.ylabel("Trust Value")
    plt.ylim(0, 1)
    plt.xlim(0, 12)
    plt.tight_layout()
    plt.show()
