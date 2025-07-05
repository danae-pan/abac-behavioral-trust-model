from pathlib import Path
import json
import matplotlib.pyplot as plt
from sensitivity_estimator.sensitivity import compute_object_sensitivity
from py_abac import Policy
from py_abac.storage.memory import MemoryStorage

"""
exp4_thresholds.py

Analyzes the impact of varying trust thresholds on object sensitivity.
Compares average sensitivity across three policy configurations with different
trust levels for the 'view' action (relaxed, baseline, strict).

Usage:
- Requires a shared access request set and 3 policy files with varying 'view' thresholds.

Outputs:
- Console summary of average sensitivities per policy.
- plots/threshold_sensitivity_comparison.png: bar chart visualization.
"""

# -------------------------------
# Experiment 4: Action Threshold Sensitivity Analysis
# -------------------------------


BASE = Path(__file__).resolve().parents[2]
REQUEST_PATH = BASE / "datasets" / "processed" / "access_requests.json"
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Policy files with different trust thresholds 
POLICY_PATHS = {
    "view_0.3": BASE / "request_builder" / "policy_view_0.3.json",   # Relaxed
    "view_0.5": BASE / "request_builder" / "policy.json",            # Baseline
    "view_0.7": BASE / "request_builder" / "policy_view_0.7.json",   # Strict
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

plt.figure(figsize=(8, 5))
plt.bar(x_labels, y_values, color=["#55A868", "#4C72B0", "#C44E52"][:len(x_labels)])
plt.ylabel("Average Object Sensitivity")
plt.title("Impact of Trust Threshold Strictness on Sensitivity")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

# Save plot
plot_path = PLOTS_DIR / "threshold_sensitivity_comparison.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Saved threshold sensitivity comparison plot to {plot_path}")
