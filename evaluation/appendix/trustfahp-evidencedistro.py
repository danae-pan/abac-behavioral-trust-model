from pyfdm.TFN import TFN
import numpy as np
import json
import matplotlib.pyplot as plt
import random
from pathlib import Path

# -----------------------------------------
# Trust Simulation with Skewed Risk Profiles
# -----------------------------------------
# This script simulates dynamic trust evolution using FAHP weights, random risk profiles, 
# and evidence values drawn from Beta distributions. The profiles (benign, malicious, neutral)
# influence the alpha/beta parameters used to generate behavioral evidence, leading to 
# personalized and evolving trust scores over time.

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
    """Invert polarity for evidence dimensions where higher values are beneficial."""
    modified = evidence.copy()
    for idx, invert in enumerate(polarity_flags):
        if invert:
            max_val = np.max(modified[:, idx])
            min_val = np.min(modified[:, idx])
            modified[:, idx] = (max_val + min_val) - modified[:, idx]
    return modified

def compute_synthetic_extents(matrix):
    """Compute fuzzy synthetic extents from TFN comparison matrix."""
    row_sums = [sum(matrix[i, :], start=TFN(0, 0, 0)) for i in range(matrix.shape[0])]
    total_sum = sum(row_sums, start=TFN(0, 0, 0))
    return [r / total_sum for r in row_sums]

def possibility_degree(m2: TFN, m1: TFN) -> float:
    """Calculate the possibility degree between two TFNs."""
    if m2.b >= m1.b:
        return 1.0
    elif m1.a >= m2.c:
        return 0.0
    else:
        return (m1.a - m2.c) / ((m2.b - m2.c) - (m1.b - m1.a))

def compute_final_weights(synthetic_extents):
    """Calculate crisp weights from fuzzy extents using possibility degree matrix."""
    n = len(synthetic_extents)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                V[i][j] = possibility_degree(synthetic_extents[i], synthetic_extents[j])
    d_prime = np.min(V + np.eye(n), axis=1)
    return d_prime / np.sum(d_prime)

def create_dummy_fuzzy_matrix(size: int):
    """Create an identity fuzzy comparison matrix with TFN(1,1,1) values."""
    return np.array([[TFN(1, 1, 1) for _ in range(size)] for _ in range(size)])

def update_trust(prev_trust, evidence, weights, gamma=0.85):
    """Apply exponential trust update formula using FAHP weights and new evidence."""
    T_new = 1 - np.dot(weights, evidence)
    return gamma * prev_trust + (1 - gamma) * np.clip(T_new, 0, 1)

def assign_risk_profile():
    """Randomly assign a risk profile (benign, malicious, or neutral) to each subject."""
    roll = random.random()
    if roll < 0.3:
        return "malicious"  # 30%
    elif roll < 0.5:
        return "benign"     # 50%
    else:
        return "neutral"    # 20%

def sample_evidence_for_profile(profile):
    """Generate 18 evidence values based on subject profile and Beta distribution skew."""
    evidence = []
    for _ in range(18):
        if profile == "benign":
            alpha, beta = random.uniform(6, 9), random.uniform(1.1, 2)
        elif profile == "malicious":
            alpha, beta = random.uniform(1.1, 2), random.uniform(6, 9)
        else:
            alpha = beta = random.uniform(1.5, 6)
        val = np.clip(np.random.beta(alpha, beta), 0, 1)
        evidence.append(val)
    return evidence


def simulate_trust(subjects, global_weights, initial_trust_map, months=12, gamma=0.85):
    """
    Simulate trust evolution per subject over time with skewed evidence sampling based on risk profile.
    Returns trust trajectory and subject risk profile mapping.
    """
    trust_records = {}
    profile_map = {}

    for subj in subjects:
        sid = subj["subject_id"]
        trust_records[sid] = [initial_trust_map.get(sid)]
        profile_map[sid] = assign_risk_profile()

    for _ in range(months):
        for subj in subjects:
            sid = subj["subject_id"]
            prev_trust = trust_records[sid][-1]
            evidence = sample_evidence_for_profile(profile_map[sid])
            evidence = invert_polarity(np.array([evidence]), evidence_polarity)[0]
            trust = update_trust(prev_trust, evidence, global_weights, gamma)
            trust_records[sid].append(trust)

    return trust_records, profile_map

if __name__ == "__main__":
    # Reproducibility setup
    np.random.seed(0)
    random.seed(0)

    ROOT = Path(__file__).resolve().parents[2]
    SUBJECTS_PATH = ROOT / "datasets" / "processed" / "all_subjects.json"
    OUTPUT_DIR = ROOT / "datasets" / "processed" / "trust_estimator"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SUBJECTS_PATH, encoding="utf-8") as f:
        subjects = json.load(f)

    initial_trust_map = {s["subject_id"]: s["trust_value"] for s in subjects}

    # Use dummy FAHP weights for three main criteria: performance, reliability, security
    perf = create_dummy_fuzzy_matrix(6)
    reli = create_dummy_fuzzy_matrix(4)
    sec = create_dummy_fuzzy_matrix(8)
    crit = create_dummy_fuzzy_matrix(3)

    global_weights = np.concatenate([
        compute_final_weights(compute_synthetic_extents(perf)) * compute_final_weights(compute_synthetic_extents(crit))[0],
        compute_final_weights(compute_synthetic_extents(reli)) * compute_final_weights(compute_synthetic_extents(crit))[1],
        compute_final_weights(compute_synthetic_extents(sec)) * compute_final_weights(compute_synthetic_extents(crit))[2],
    ])

    # Run simulation and save results
    trust_records, profile_map = simulate_trust(subjects, global_weights, initial_trust_map)

    with open(OUTPUT_DIR / "trust_records_fahp_risk_profiles.json", "w") as f:
        json.dump(trust_records, f, indent=2)

    with open(OUTPUT_DIR / "subject_risk_profiles.json", "w") as f:
        json.dump(profile_map, f, indent=2)

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

    # Visualize sample of subject trust trajectories
    plt.figure(figsize=(10, 6))
    for i, (sid, history) in enumerate(trust_records.items()):
        if i >= 100:
            break
        plt.plot(history, alpha=0.2)
    plt.title("Trust Evolution for 100 Subjects (Skewed Risk Profiles)")
    plt.xlabel("Month")
    plt.ylabel("Trust Value")
    plt.ylim(0, 1)
    plt.xlim(0, 12)
    plt.tight_layout()
    plt.show()
