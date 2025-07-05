"""
exp2_risk_evaluation.py

Performs a simple risk scoring per attribute based on:
- Likelihood: how often the attribute appears in access requests
- Impact: severity of the actions it enables, mapped via trust thresholds

Usage:
- Requires prior sensitivity diagnostics (`experiment1_attr_summary.csv`)
- Uses access requests and policy rules to determine attribute usage

Outputs:
- Printed ranked risk scores
- Saves results to: datasets/processed/sensitivity_results/exp2_risk_scores.csv
"""

import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

# ----------------------
# Configuration
# ----------------------
BASE = Path(__file__).resolve().parents[2]
REQ_PATH = BASE / "datasets" / "processed" / "access_requests.json"
POLICY_PATH = BASE / "request_builder" / "policy.json"
SENS_FLAT_PATH = BASE / "datasets" / "processed" / "sensitivity_results" / "sensitivity_flat.csv"
EXP1_SUMMARY_PATH = BASE / "datasets" / "processed" / "sensitivity_results" / "diagnostics" / "experiment1_attr_summary.csv"
OUTPUT_PATH = BASE / "datasets" / "processed" / "sensitivity_results" / "exp2_risk_scores.csv"

# ----------------------
# Load data
# ----------------------
with open(REQ_PATH) as f:
    access_requests = json.load(f)

flat_df = pd.read_csv(SENS_FLAT_PATH)
exp1_df = pd.read_csv(EXP1_SUMMARY_PATH)
exp1_df.rename(columns={"request_frequency": "request_count", "unique_object_count": "object_count"}, inplace=True)

with open(POLICY_PATH) as f:
    policies = json.load(f)["policies"]

# ----------------------
# Build policy matrix
# ----------------------
policy_matrix = defaultdict(lambda: defaultdict(set))
for p in policies:
    rules = p["rules"]
    obj_type = rules["resource"].get("$.obj_type", {}).get("value")
    methods = rules["action"].get("$.method", {}).get("values", []) or [rules["action"]["$.method"]["value"]]
    for attr_path in rules["resource"]:
        if attr_path != "$.obj_type":
            attr = attr_path.replace("$.", "")
            for method in methods:
                policy_matrix[obj_type][method].add(attr)

# -------------------------------------
# Count attribute/action occurrences
# -------------------------------------
attr_action_counts = defaultdict(lambda: defaultdict(int))
for req in access_requests:
    action = req["action"]["attributes"]["method"]
    res_attrs = req["resource"]["attributes"]
    obj_type = res_attrs.get("obj_type")
    valid_attrs = policy_matrix.get(obj_type, {}).get(action, set())
    for attr in valid_attrs:
        if res_attrs.get(attr) is not None:
            attr_action_counts[attr][action] += 1

# -------------------------------------
# Trust threshold → severity mapping
# -------------------------------------
"""
This dictionary maps each action to a corresponding minimum trust threshold,
which is then used to infer the severity of the action:

- Trust ≥ 0.9 → severity 3 (high)
- Trust ≥ 0.7 → severity 2 (medium)
- Else         → severity 1 (low)

Severity is then combined with attribute usage frequency to calculate an overall risk score.
"""

action_trust_map = {
    "view": 0.5,
    "access": 0.7,
    "enroll": 0.5,
    "submit": 0.7,
    "edit": 0.7,
    "delete": 0.9
}

# ----------------------
# Risk scoring
# ----------------------
results = []
for attr in attr_action_counts:
    row = exp1_df[exp1_df["attribute"] == attr]
    if row.empty:
        continue
    request_count = row["request_count"].values[0]
    object_count = row["object_count"].values[0]
    likelihood = request_count / object_count if object_count > 0 else 0

    total_weighted_severity = 0
    total_requests = 0
    for action, count in attr_action_counts[attr].items():
        trust = action_trust_map.get(action, 0.5)
        severity = 3 if trust >= 0.9 else 2 if trust >= 0.7 else 1
        total_weighted_severity += severity * count
        total_requests += count

    impact = total_weighted_severity / total_requests if total_requests > 0 else 0
    risk = likelihood * impact

    results.append({
        "attribute": attr,
        "likelihood": round(likelihood, 4),
        "impact": round(impact, 4),
        "risk": round(risk, 4)
    })

# ----------------------
# Output
# ----------------------
risk_df = pd.DataFrame(results).sort_values("risk", ascending=False)
risk_df.to_csv(OUTPUT_PATH, index=False)

print("\nRisk Evaluation Summary")
print(risk_df.to_string(index=False))
print(f"\n Saved to: {OUTPUT_PATH}")
