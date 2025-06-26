from pathlib import Path
import json
import matplotlib.pyplot as plt
from sensitivity_estimator.sensitivity import compute_object_sensitivity
from py_abac import Policy
from py_abac.storage.memory import MemoryStorage

# -------------------------------
# Experiment 4: Action Threshold Sensitivity Analysis
# -------------------------------

# Define base paths
ROOT = Path(__file__).resolve().parents[2]
REQUEST_PATH = ROOT / "datasets" / "processed" / "access_requests.json"

# Policy files with different trust thresholds for 'view'
POLICY_PATHS = {
    "view_0.3": ROOT / "request_builder" / "policy_view_0.3.json",   # Relaxed
    "view_0.5": ROOT / "request_builder" / "policy.json",            # Baseline
    "view_0.7": ROOT / "request_builder" / "policy_view_0.7.json",   # Strict
}

# Load access requests
with open(REQUEST_PATH, encoding="utf-8") as f:
    access_requests = json.load(f)

def extract_policy_resource_attributes(policy_dicts):
    """
    Extract resource attributes used in policy conditions.
    """
    used_attrs = set()
    for pol in policy_dicts:
        resource_rules = pol.get("rules", {}).get("resource", {})
        for attr in resource_rules:
            if attr.startswith("$."):
                used_attrs.add(attr[2:])
    return used_attrs

# Run sensitivity computation for each policy set
results = {}
for label, path in POLICY_PATHS.items():
    if not path.exists():
        print(f"[!] Skipping missing policy file: {path.name}")
        continue

    with open(path, encoding="utf-8") as f:
        policies = json.load(f)["policies"]

    storage = MemoryStorage()
    for p in policies:
        storage.add(Policy.from_json(p))

    used_attrs = extract_policy_resource_attributes(policies)
    threshold, avg_sens, *_ = compute_object_sensitivity(access_requests, storage, used_attrs)
    results[label] = avg_sens
    print(f"[{label}] → Average Sensitivity: {avg_sens:.4f}")

# Plot sensitivity comparison
label_map = {
    "view_0.3": "Relaxed (0.3)",
    "view_0.5": "Baseline (0.5)",
    "view_0.7": "Strict (0.7)"
}

raw_keys = list(results.keys())  
x_labels = [label_map[k] for k in raw_keys]
y_values = [results[k] for k in raw_keys]

plt.bar(x_labels, y_values, color=["#55A868", "#4C72B0", "#C44E52"][:len(x_labels)])
plt.ylabel("Average Object Sensitivity")
plt.title("Impact of Trust Threshold Strictness on Sensitivity")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
