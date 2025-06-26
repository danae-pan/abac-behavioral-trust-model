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

# ----------------------
# Load data
# ----------------------
with open(REQ_PATH) as f:
    access_requests = json.load(f)

flat_df = pd.read_csv(SENS_FLAT_PATH)
exp1_df = pd.read_csv(EXP1_SUMMARY_PATH)

exp1_df.rename(columns={
    "request_frequency": "request_count",
    "unique_object_count": "object_count"
}, inplace=True)

# ----------------------
# Load policy set and extract attribute usage matrix
# ----------------------
with open(POLICY_PATH) as f:
    policies = json.load(f)["policies"]

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

# ----------------------
# Count attribute usage across actions
# ----------------------
attr_action_counts = defaultdict(lambda: defaultdict(int))
attr_total_counts = defaultdict(int)

for req in access_requests:
    action = req["action"]["attributes"]["method"]
    res_attrs = req["resource"]["attributes"]
    obj_type = res_attrs.get("obj_type")

    valid_attrs = policy_matrix.get(obj_type, {}).get(action, set())
    for attr in valid_attrs:
        if res_attrs.get(attr) is not None:
            attr_action_counts[attr][action] += 1
            attr_total_counts[attr] += 1

# ----------------------
# Count object appearances per attribute
# ----------------------
attr_object_counts = flat_df.groupby("attribute")["object_id"].nunique().to_dict()

# ----------------------
# Define trust thresholds → severity levels
# ----------------------
action_trust_map = {
    "view": 0.5,
    "access": 0.7,
    "enroll": 0.5,
    "submit": 0.7,
    "edit": 0.7,
    "delete": 0.9
}

# ----------------------
# Risk Calculation
# ----------------------
results = []

for attr in attr_action_counts:
    request_count = exp1_df.loc[exp1_df["attribute"] == attr, "request_count"].values[0]
    object_count = exp1_df.loc[exp1_df["attribute"] == attr, "object_count"].values[0]

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
print("\n[+] Risk Evaluation Summary")
print(risk_df.to_string(index=False))
