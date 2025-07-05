import json
import numpy as np
from pathlib import Path
import random
from pyfdm.TFN import TFN
from py_abac.storage.memory import MemoryStorage
from py_abac import Policy, PDP, Request 
import matplotlib.pyplot as plt

"""
trustfahp-insiderattack.py

Simulates trust evolution for a single insider using the CERT IP theft scenario.
Implements a two-phase FAHP weighting scheme, where the first month is
performance-dominant and the second month is security-dominant. Evidence is
skewed using CERT-aligned beta distributions, and polarity inversion ensures
all dimensions reflect increasing suspicion.

Usage:
- Expects processed subject and object datasets with ABAC policy rules.
- Selects one student with valid access before trust drop.
- Applies time-stepped trust updates across 8 weeks using evolving weights.
- Outputs weekly trust values and a line plot showing trust evolution.

Outputs:
- datasets/processed/trust_estimator/trust_ip_theft_insider.json
- plots/trust_cert_simulation.png

"""

# ----------------------
# 1. Evidence Labels and Polarity
# ----------------------

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

# -------------------------------
# 2. FAHP Utilities
# -------------------------------
def compute_synthetic_extents(matrix):
    """
    Computes fuzzy synthetic extent values from a fuzzy pairwise comparison matrix.
    """
    row_sums = [sum(matrix[i], start=TFN(0, 0, 0)) for i in range(matrix.shape[0])]
    total = sum(row_sums, start=TFN(0, 0, 0))
    return [r / total for r in row_sums]

def possibility_degree(m2, m1):
    """
    Computes the possibility degree that fuzzy number m2 ≥ m1.
    """
    if m2.b >= m1.b:
        return 1.0
    elif m1.a >= m2.c:
        return 0.0
    return (m1.a - m2.c) / ((m2.b - m2.c) - (m1.b - m1.a))

def compute_final_weights(extents):
    """
    Computes crisp weights from fuzzy synthetic extents using the minimum possibility method.
    """
    n = len(extents)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                V[i][j] = possibility_degree(extents[i], extents[j])
    d = np.min(V + np.eye(n), axis=1)
    return d / np.sum(d)

# -------------------------------
# 3. FAHP Outer Matrices
# -------------------------------

# Month 1: Performance-Dominant
outer_m1 = np.array([
    [TFN(1, 1, 1), TFN(2, 3, 4), TFN(6, 7, 8)],
    [TFN(1/4, 1/3, 1/2), TFN(1, 1, 1), TFN(5, 6, 7)],
    [TFN(1/8, 1/7, 1/6), TFN(1/7, 1/6, 1/5), TFN(1, 1, 1)]
])
outer_weights_m1 = compute_final_weights(compute_synthetic_extents(outer_m1))

# Month 2: Security-Dominant
outer_m2 = np.array([
    [TFN(1, 1, 1), TFN(1/7, 1/6, 1/5), TFN(1/6, 1/5, 1/4)],
    [TFN(5, 6, 7), TFN(1, 1, 1), TFN(1/4, 1/3, 1/2)],
    [TFN(4, 5, 6), TFN(1, 2, 3), TFN(1, 1, 1)]
])
outer_weights_m2 = compute_final_weights(compute_synthetic_extents(outer_m2))

# ----------------------------------------
# 3A. FAHP Inner Matrices – First Month
# ----------------------------------------

# Performance (6 indicators) 
perf = np.array([
    [TFN(1,1,1), TFN(3,4,5), TFN(5,6,7), TFN(4,5,6), TFN(3,4,5), TFN(6,7,8)],
    [TFN(1/5,1/4,1/3), TFN(1,1,1), TFN(3,4,5), TFN(5,6,7), TFN(4,5,6), TFN(7,8,9)],
    [TFN(1/7,1/6,1/5), TFN(1/5,1/4,1/3), TFN(1,1,1), TFN(4,5,6), TFN(3,4,5), TFN(5,6,7)],
    [TFN(1/6,1/5,1/4), TFN(1/7,1/6,1/5), TFN(1/6,1/5,1/4), TFN(1,1,1), TFN(3,4,5), TFN(5,6,7)],
    [TFN(1/5,1/4,1/3), TFN(1/6,1/5,1/4), TFN(1/5,1/4,1/3), TFN(1/5,1/4,1/3), TFN(1,1,1), TFN(6,7,8)],
    [TFN(1/8,1/7,1/6), TFN(1/9,1/8,1/7), TFN(1/7,1/6,1/5), TFN(1/7,1/6,1/5), TFN(1/8,1/7,1/6), TFN(1,1,1)]
])

w_perf = compute_final_weights(compute_synthetic_extents(perf)) * outer_weights_m1[0]

# Reliability (4 indicators) 
reli = np.array([
    [TFN(1,1,1), TFN(2,3,4), TFN(1,2,3), TFN(2,3,4)],
    [TFN(1/4,1/3,1/2), TFN(1,1,1), TFN(1,2,3), TFN(1,2,3)],
    [TFN(1/3,1/2,1), TFN(1/3,1/2,1), TFN(1,1,1), TFN(1,2,3)],
    [TFN(1/4,1/3,1/2), TFN(1/3,1/2,1), TFN(1/3,1/2,1), TFN(1,1,1)]
])
w_reli = compute_final_weights(compute_synthetic_extents(reli)) * outer_weights_m1[1]

# Security (8 indicators) 
secu = np.array([
    # 10  11   12   13   14   15   16   17
    [TFN(1,1,1), TFN(3,4,5), TFN(1,2,3), TFN(1,2,3), TFN(3,4,5), TFN(3,4,5), TFN(3,4,5), TFN(3,4,5)],  # 10
    [TFN(1/5,1/4,1/3), TFN(1,1,1), TFN(1/3,1/2,1), TFN(1/3,1/2,1), TFN(1,2,3), TFN(1,2,3), TFN(1,2,3), TFN(1,2,3)],  # 11
    [TFN(1/3,1/2,1), TFN(1,2,3), TFN(1,1,1), TFN(1,2,3), TFN(2,3,4), TFN(2,3,4), TFN(2,3,4), TFN(2,3,4)],  # 12
    [TFN(1/3,1/2,1), TFN(1,2,3), TFN(1/3,1/2,1), TFN(1,1,1), TFN(2,3,4), TFN(2,3,4), TFN(2,3,4), TFN(2,3,4)],  # 13
    [TFN(1/5,1/4,1/3), TFN(1/3,1/2,1), TFN(1/4,1/3,1/2), TFN(1/4,1/3,1/2), TFN(1,1,1), TFN(1,2,3), TFN(1,2,3), TFN(1,2,3)],  # 14
    [TFN(1/5,1/4,1/3), TFN(1/3,1/2,1), TFN(1/4,1/3,1/2), TFN(1/4,1/3,1/2), TFN(1/3,1/2,1), TFN(1,1,1), TFN(1,2,3), TFN(1,2,3)],  # 15
    [TFN(1/5,1/4,1/3), TFN(1/3,1/2,1), TFN(1/4,1/3,1/2), TFN(1/4,1/3,1/2), TFN(1/3,1/2,1), TFN(1/3,1/2,1), TFN(1,1,1), TFN(1,2,3)],  # 16
    [TFN(1/5,1/4,1/3), TFN(1/3,1/2,1), TFN(1/4,1/3,1/2), TFN(1/4,1/3,1/2), TFN(1/3,1/2,1), TFN(1/3,1/2,1), TFN(1/3,1/2,1), TFN(1,1,1)]   # 17
])
w_secu = compute_final_weights(compute_synthetic_extents(secu)) * outer_weights_m1[2]

global_weights = np.concatenate([w_perf, w_reli, w_secu])

# ----------------------------------------
# 3B. FAHP Inner Matrices – Second Month
# ----------------------------------------

# Performance (6 indicators: 6–11) 
perf_2 = np.array([
    [TFN(1,1,1), TFN(3,4,5), TFN(3,4,5), TFN(1,2,3), TFN(3,4,5), TFN(1,2,3)],
    [TFN(1/5,1/4,1/3), TFN(1,1,1), TFN(3,4,5), TFN(1,2,3), TFN(3,4,5), TFN(1,2,3)],
    [TFN(1/5,1/4,1/3), TFN(1/5,1/4,1/3), TFN(1,1,1), TFN(1,2,3), TFN(3,4,5), TFN(1,2,3)],
    [TFN(1/3,1/2,1), TFN(1/3,1/2,1), TFN(1/3,1/2,1), TFN(1,1,1), TFN(1,2,3), TFN(1,2,3)],
    [TFN(1/5,1/4,1/3), TFN(1/5,1/4,1/3), TFN(1/5,1/4,1/3), TFN(1/3,1/2,1), TFN(1,1,1), TFN(1,2,3)],
    [TFN(1/3,1/2,1), TFN(1/3,1/2,1), TFN(1/3,1/2,1), TFN(1/3,1/2,1), TFN(1/3,1/2,1), TFN(1,1,1)]
])
w_perf_2 = compute_final_weights(compute_synthetic_extents(perf_2)) * outer_weights_m2[0]

# Reliability (4 indicators: 10–13) 
reli_2 = np.array([
    [TFN(1,1,1), TFN(1,2,3), TFN(1,2,3), TFN(1,2,3)],
    [TFN(1/3,1/2,1), TFN(1,1,1), TFN(1,2,3), TFN(1,2,3)],
    [TFN(1/3,1/2,1), TFN(1/3,1/2,1), TFN(1,1,1), TFN(1,2,3)],
    [TFN(1/3,1/2,1), TFN(1/3,1/2,1), TFN(1/3,1/2,1), TFN(1,1,1)]
])
w_reli_2 = compute_final_weights(compute_synthetic_extents(reli_2)) * outer_weights_m2[1]

# Security (8 indicators: 10–17) 
secu_2 = np.array([
    [TFN(1,1,1), TFN(2,3,4), TFN(2,3,4), TFN(3,4,5), TFN(6,7,8), TFN(6,7,8), TFN(6,7,8), TFN(6,7,8)],
    [TFN(1/4,1/3,1/2), TFN(1,1,1), TFN(1,2,3), TFN(1,2,3), TFN(4,5,6), TFN(4,5,6), TFN(4,5,6), TFN(4,5,6)],
    [TFN(1/4,1/3,1/2), TFN(1/3,1/2,1), TFN(1,1,1), TFN(3,4,5), TFN(4,5,6), TFN(4,5,6), TFN(4,5,6), TFN(4,5,6)],
    [TFN(1/5,1/4,1/3), TFN(1/3,1/2,1), TFN(1/5,1/4,1/3), TFN(1,1,1), TFN(6,7,8), TFN(6,7,8), TFN(6,7,8), TFN(6,7,8)],
    [TFN(1/8,1/7,1/6), TFN(1/6,1/5,1/4), TFN(1/6,1/5,1/4), TFN(1/8,1/7,1/6), TFN(1,1,1), TFN(1/4,1/3,1/2), TFN(1/3,1/2,1), TFN(1/3,1/2,1)],
    [TFN(1/8,1/7,1/6), TFN(1/6,1/5,1/4), TFN(1/6,1/5,1/4), TFN(1/8,1/7,1/6), TFN(2,3,4), TFN(1,1,1), TFN(1/3,1/2,1), TFN(1/4,1/3,1/2)],
    [TFN(1/8,1/7,1/6), TFN(1/6,1/5,1/4), TFN(1/6,1/5,1/4), TFN(1/8,1/7,1/6), TFN(1,2,3), TFN(1,2,3), TFN(1,1,1), TFN(1/3,1/2,1)],
    [TFN(1/8,1/7,1/6), TFN(1/6,1/5,1/4), TFN(1/6,1/5,1/4), TFN(1/8,1/7,1/6), TFN(1,2,3), TFN(2,3,4), TFN(1,2,3), TFN(1,1,1)]
])


w_secu_2 = compute_final_weights(compute_synthetic_extents(secu_2)) * outer_weights_m2[2]

global_weights_month2 = np.concatenate([w_perf_2, w_reli_2, w_secu_2])

# -------------------------------
# 4. Trust Simulation
# -------------------------------
def generate_evidence_two_months(week):
    """
    Generate 18 evidence values using CERT-relevant Beta distributions,
    differentiating between first and second month behavior.
    """
    first_month_map = {
        0: "moderate", 1: "strong",    2: "moderate", 3: "weak",     4: "strong", 5: "weak",
        6: "strong",   7: "moderate",  8: "strong",   9: "weak",
        10: None,      11: None,       12: "moderate", 13: None,
        14: None, 15: None, 16: None, 17: None
    }

    second_month_map = {
        0: None, 1: None, 2: None, 3: None, 4: None, 5: None,
        6: "strong", 7: "moderate", 8: "strong", 9: "moderate",
        10: "moderate", 11: "weak", 12: "moderate", 13: "moderate",
        14: "moderate", 15: "moderate", 16: "weak", 17: "weak"
    }

    beta_params_month1 = {
        "strong":   (2, 6),    
        "moderate": (1.5, 7),   
        "weak":     (1.2, 8)    
    }

    beta_params_month2 = {
        "strong":   (3, 4),     
        "moderate": (2.5, 5),   
        "weak":     (2, 6)      
    }
    
    beta_params = beta_params_month1 if week < 4 else beta_params_month2

    relevance_map = first_month_map if week < 4 else second_month_map

    evidence = []
    for i in range(18):
        category = relevance_map.get(i)
        if category is None:
            val = np.random.uniform(0.05, 0.1)  
            evidence.append(val)
        else:
            alpha, beta = beta_params[category]
            val = np.clip(np.random.beta(alpha, beta), 0, 1)
            evidence.append(val)
    return evidence

def invert_polarity(evidence, polarity_flags):
    """
    Invert specific evidence values where higher is benign (True), so we flip it.
    Result: all values aligned such that higher = more suspicious.
    """
    evidence = evidence.copy()
    for i, invert in enumerate(polarity_flags):
        if invert:
            evidence[i] = 1.0 - evidence[i]
    return evidence

def update_trust(prev_trust, evidence, weights, gamma):
    """
    Updates trust using exponential smoothing on weighted evidence scores.
    T_new = 1 - dot(weights, evidence)
    """
    T_new = 1 - np.dot(weights, evidence)
    return gamma * prev_trust + (1 - gamma) * np.clip(T_new, 0, 1)


# -------------------- ENTRY POINT --------------------

NUM_WEEKS = 8
GAMMA = 0.4
random.seed(42)
np.random.seed(42)

BASE = Path(__file__).resolve().parents[1]
POLICY_PATH = BASE / "request_builder" / "policy.json"
SUBJECTS_PATH = BASE / "datasets" / "processed" / "all_attack_subjects.json"
OUTPUT_DIR = BASE / "datasets" / "processed" / "trust_estimator"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load ABAC policies
with open(POLICY_PATH) as f:
    policies = json.load(f)["policies"]
    
# Load PDP
storage = MemoryStorage()
for p in policies:
    storage.add(Policy.from_json(p))
pdp = PDP(storage)


with open(SUBJECTS_PATH) as f:
    subjects = json.load(f)

# Load objects (to validate course_code/school_code)
OBJECTS_PATH = BASE / "datasets" / "processed" / "all_attack_objects.json"
with open(OBJECTS_PATH) as f:
    objects = json.load(f)

# Build course assignment lookup
course_assignments = [
    o for o in objects if o["obj_type"] == "course assignment"
]

# Select a student with at least 1 course assignment they can access initially
eligible = []
for s in subjects:
    if s["role"] != "Student" or s["trust_value"] < 0.7:
        continue
    s_school = s.get("school_code", [])[0]
    s_courses = s.get("course_code", [])
    matching_objs = [
        o for o in objects
        if o["obj_type"] == "course assignment"
        and o.get("school_code") == s_school
        and o.get("course_code") in s_courses
    ]
    for obj in matching_objs:
        req = {
            "subject": {
                "id": str(s["subject_id"]),
                "attributes": {**s, "trust_value": s["trust_value"]}
            },
            "resource": {
                "id": str(obj["object_id"]),
                "attributes": obj
            },
            "action": {
                "id": "submit",
                "attributes": {
                    "method": "submit"
                }
            },
            "context": {}
        }
        if pdp.is_allowed(Request.from_json(req)):
            eligible.append(s)
            break

if not eligible:
    raise ValueError("No eligible students found with access before trust drop.")

selected = random.choice(eligible)
sid = selected["subject_id"]


trust_values = [selected["trust_value"]]
for week in range(NUM_WEEKS):
    ev = generate_evidence_two_months(week)
    ev = invert_polarity(ev, evidence_polarity)

    #Use different weights depending on the week
    if week < 4:
        weights = global_weights
        GAMMA = 0.4
    else:
        weights = global_weights_month2
        GAMMA = 0.2
    

    trust = update_trust(trust_values[-1], ev, weights, GAMMA)
    trust_values.append(trust)
    
with open(OUTPUT_DIR / "trust_ip_theft_insider.json", "w") as f:
    json.dump({sid: trust_values}, f, indent=2)

print(f"Simulated trust for insider {sid} saved to {OUTPUT_DIR / 'trust_ip_theft_insider.json'}")

plt.figure(figsize=(8, 4))
plt.plot(range(len(trust_values)), trust_values, marker='o', linestyle='-', color='navy')
plt.axvline(4.0, linestyle='--', color='gray', label='Resignation point')
plt.title("Insider Trust Evolution Over 8 Weeks")
plt.xlabel("Week")
plt.ylabel("Trust Value")
plt.xticks(range(len(trust_values)))
plt.grid(True)
plt.legend()

FIGS_DIR = BASE / "plots"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(FIGS_DIR / "trust_cert_simulation.png", dpi=300)
print(f"Trust evolution plot saved to {FIGS_DIR / 'trust_cert_simulation.png'}")
