import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from adjustText import adjust_text

"""
exp1_attribute_diagnostics.py

Performs attribute-level diagnostics to explore correlations between ABAC policy sensitivity
and metadata such as request frequency, policy usage, object coverage, and denial rates.

Usage:
- Expects prior sensitivity analysis results in `sensitivity_results/` directory.
- Requires processed access requests and policy definitions.

Outputs:
- plots/sens_vs_attr_freq_annotated.png
- plots/sens_vs_policy_count_annotated.png
- plots/sens_vs_objectcount_annotated.png
- plots/sens_vs_denial_rate_annotated.png
"""

# ------------------------------
# Experiment 1: Attribute Sensitivity Diagnostics
# ------------------------------


BASE_DIR = Path(__file__).resolve().parents[2]
REQUEST_BUILDER_DIR = BASE_DIR / "request_builder"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
SENS_DIR = PROCESSED_DIR / "sensitivity_results"
(SENS_DIR / "diagnostics").mkdir(parents=True, exist_ok=True)
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)



# Load input data
flat_df = pd.read_csv(SENS_DIR / "sensitivity_flat.csv")
summary_df = pd.read_csv(SENS_DIR / "sensitivity_attribute_summary.csv")

with open(PROCESSED_DIR / "access_requests.json") as f:
    access_requests = json.load(f)

with open(REQUEST_BUILDER_DIR / "policy.json") as f:
    policy_set = json.load(f)["policies"]

def extract_policy_matrix(policies):
    """
    Builds a matrix mapping object types and actions to the resource attributes 
    they reference in ABAC policies.
    """
    matrix = defaultdict(lambda: defaultdict(set))
    for p in policies:
        rules = p["rules"]
        obj_type = rules["resource"].get("$.obj_type", {}).get("value")
        methods = rules["action"].get("$.method", {}).get("values", []) or [rules["action"]["$.method"]["value"]]
        if obj_type:
            for attr_path in rules["resource"]:
                if attr_path != "$.obj_type":
                    attr = attr_path.replace("$.", "")
                    for method in methods:
                        matrix[obj_type][method].add(attr)
    return matrix

policy_matrix = extract_policy_matrix(policy_set)

# Count attribute usage frequency in requests
attr_freq_counter = Counter()
for req in access_requests:
    res = req["resource"]["attributes"]
    obj_type = res.get("obj_type")
    action = req["action"]["attributes"]["method"]
    valid_attrs = policy_matrix.get(obj_type, {}).get(action, set())
    for attr in valid_attrs:
        if res.get(attr) is not None:
            attr_freq_counter[attr] += 1

obj_count_df = flat_df.groupby("attribute")["object_id"].nunique().reset_index()
obj_count_df.columns = ["attribute", "unique_object_count"]

policy_count = Counter()
for p in policy_set:
    for attr in p["rules"]["resource"]:
        if attr != "$.obj_type":
            policy_count[attr.replace("$.", "")] += 1
policy_count_df = pd.DataFrame(policy_count.items(), columns=["attribute", "policy_count"])

summary_df = summary_df[summary_df["attribute"] != "obj_type"]
freq_df = pd.DataFrame(attr_freq_counter.items(), columns=["attribute", "request_frequency"])

merged = summary_df.merge(freq_df, on="attribute", how="left")
merged = merged.merge(obj_count_df, on="attribute", how="left")
merged = merged.merge(policy_count_df, on="attribute", how="left")
merged = merged.fillna(0)

# Plot 1: Sensitivity vs. Attribute Request Frequency
plt.figure(figsize=(9, 6))
sns.scatterplot(data=merged, x="request_frequency", y="mean", hue="attribute", palette="tab10", s=100)
plt.title("Sensitivity vs. Attribute Request Frequency")
plt.xlabel("Attribute Frequency in Access Requests")
plt.ylabel("Mean Sensitivity")
texts = []
grouped = merged.groupby(["request_frequency", "mean"])
for _, group in grouped:
    for i, (_, row) in enumerate(group.iterrows()):
        jittered_y = row["mean"] + (i - len(group) / 2) * 0.3
        texts.append(plt.text(row["request_frequency"], jittered_y, row["attribute"], fontsize=8))
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
plt.tight_layout()
plt.savefig(PLOTS_DIR / "sens_vs_attr_freq_annotated.png", bbox_inches="tight", dpi=300)
plt.show()

# Plot 2: Sensitivity vs. Number of Policies
plt.figure(figsize=(9, 6))
sns.scatterplot(data=merged, x="policy_count", y="mean", hue="attribute", palette="tab10", s=100)
plt.title("Sensitivity vs. Number of Policies Using Each Attribute")
plt.xlabel("Number of Policies Using Attribute")
plt.ylabel("Mean Sensitivity")
texts = []
grouped = merged.groupby(["policy_count", "mean"])
for _, group in grouped:
    for i, (_, row) in enumerate(group.iterrows()):
        jittered_y = row["mean"] + (i - len(group) / 2) * 0.05
        texts.append(plt.text(row["policy_count"], jittered_y, row["attribute"], fontsize=8))
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
plt.tight_layout()
plt.savefig(PLOTS_DIR / "sens_vs_policy_count_annotated.png", bbox_inches="tight", dpi=300)
plt.show()

# Plot 3: Sensitivity vs. Unique Object Count
plt.figure(figsize=(9, 6))
sns.scatterplot(data=merged, x="unique_object_count", y="mean", hue="attribute", palette="tab10", s=100)
plt.title("Sensitivity vs. Number of Objects Containing Attribute")
plt.xlabel("Unique Object Count")
plt.ylabel("Mean Sensitivity")
texts = [plt.text(row["unique_object_count"], row["mean"], row["attribute"], fontsize=8) for _, row in merged.iterrows()]
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
plt.tight_layout()
plt.savefig(PLOTS_DIR / "sens_vs_objectcount_annotated.png", bbox_inches="tight", dpi=300)
plt.show()

# Plot 4: Sensitivity vs. Denial Rate
denial_df = pd.read_csv(SENS_DIR / "denial_rate_summary.csv")
denial_merged = summary_df.merge(denial_df, on="attribute", how="inner")
plt.figure(figsize=(9, 6))
sns.scatterplot(data=denial_merged, x="denial_rate", y="mean", hue="attribute", s=100, palette="tab10")
plt.title("Sensitivity vs. Denial Rate per Attribute")
plt.xlabel("Denial Rate")
plt.ylabel("Mean Sensitivity")
texts = [plt.text(row["denial_rate"], row["mean"], row["attribute"], fontsize=8) for _, row in denial_merged.iterrows()]
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
plt.tight_layout()
plt.savefig(PLOTS_DIR / "sens_vs_denial_rate_annotated.png", bbox_inches="tight", dpi=300)
plt.show()

# Save summary with all metrics for Experiment 2
merged.to_csv(SENS_DIR / "diagnostics" / "experiment1_attr_summary.csv", index=False)
print(f"Saved merged attribute diagnostics to {SENS_DIR / 'diagnostics' / 'experiment1_attr_summary.csv'}")
