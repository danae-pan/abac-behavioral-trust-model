import json
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from py_abac import PDP, Policy, AccessRequest
from py_abac.storage.memory import MemoryStorage
from sensitivity_estimator.sensitivity import compute_object_sensitivity

"""
exp3_insider_attack.py

Evaluates object sensitivity before and after insider trust degradation.
Simulates insider IP theft scenario using same-school user requests on 10 target objects.

Usage:
- Expects trusted insider and target object IDs from prior attack simulation.
- Generates access requests before and after trust change.
- Computes object-level sensitivity in both phases.

Outputs:
- insider_target_object_sensitivity.json: sensitivity results pre/post trust drop.
- plots/insider_object_sensitivity_comparison.png: bar chart of object sensitivities.
"""

# -------------------------------
# Paths
# -------------------------------
BASE = Path(__file__).resolve().parents[2]
SUBJECTS_PATH = BASE / "datasets" / "processed" / "all_attack_subjects.json"
OBJECTS_PATH = BASE / "datasets" / "processed" / "all_attack_objects.json"
POLICY_PATH = BASE / "request_builder" / "policy.json"
TRUST_PATH = BASE / "datasets" / "processed" / "trust_estimator" / "trust_ip_theft_insider.json"
TARGET_OBJECTS_PATH = BASE / "datasets" / "processed" / "target_objects.json"

# -------------------------------
# Load Data
# -------------------------------
with open(SUBJECTS_PATH) as f:
    subjects = json.load(f)
with open(OBJECTS_PATH) as f:
    objects = json.load(f)
with open(POLICY_PATH) as f:
    policy_data = json.load(f)["policies"]
with open(TRUST_PATH) as f:
    trust_data = json.load(f)
with open(TARGET_OBJECTS_PATH) as f:
    target_object_ids = json.load(f)

# Load policies into PDP
storage = MemoryStorage()
for pol in policy_data:
    storage.add(Policy.from_json(pol))
pdp = PDP(storage)


# Get insider info
insider_id = list(trust_data.keys())[0]
insider_final_trust = trust_data[insider_id][-1]
insider = next((s for s in subjects if str(s["subject_id"]) == insider_id), None)
initial_trust = insider["trust_value"]
school_code = insider["school_code"]

# Get all same-school users
candidate_subjects = [s for s in subjects if s["school_code"] == school_code]

# Map objects by ID
obj_map = {o["object_id"]: o for o in objects if o["object_id"] in target_object_ids}
target_objects = list(obj_map.values())

# Confirm actual denial after trust drop
print("Confirming insider access denial to 10 objects...")
denied_confirmed = []
for obj in target_objects:
    req = {
        "subject": {
            "id": insider_id,
            "attributes": {**insider, "trust_value": insider_final_trust}
        },
        "resource": {
            "id": str(obj["object_id"]),
            "attributes": obj
        },
        "action": {
            "id": "submit",
            "attributes": {"method": "submit"}
        },
        "context": {}
    }
    if not pdp.is_allowed(AccessRequest.from_json(req)):
        denied_confirmed.append(obj["object_id"])

if len(denied_confirmed) < 10:
    print(f"Only {len(denied_confirmed)} of 10 objects are denied.")
    target_object_ids = denied_confirmed
    target_objects = [obj_map[obj_id] for obj_id in denied_confirmed]

# -------------------------------
# Generate Requests (After Trust Drop)
# -------------------------------
print("Generating access requests from same-school users (after trust)...")
access_requests_after = []
for subj in candidate_subjects:
    for obj in target_objects:
        req = {
            "subject": {
                "id": str(subj["subject_id"]),
                "attributes": {**subj}
            },
            "resource": {
                "id": str(obj["object_id"]),
                "attributes": obj
            },
            "action": {
                "id": "submit",
                "attributes": {"method": "submit"}
            },
            "context": {}
        }
        access_requests_after.append(req)

# Inject final trust value for insider in AFTER requests
for req in access_requests_after:
    if req["subject"]["id"] == insider_id:
        req["subject"]["attributes"]["trust_value"] = insider_final_trust

# -------------------------------
# Run Sensitivity (After Trust)
# -------------------------------
policy_attrs = {
    attr[2:]
    for p in policy_data
    for attr in p.get("rules", {}).get("resource", {})
    if attr.startswith("$.")
}

threshold_after, avg_after, sensitivities_after, sensitive_attrs_after, _ = compute_object_sensitivity(
    access_requests_after, storage, policy_attrs
)

# -------------------------------
# Clone Requests and Reinsert Initial Trust for Insider
# -------------------------------
access_requests_before = deepcopy(access_requests_after)
for req in access_requests_before:
    if req["subject"]["id"] == insider_id:
        req["subject"]["attributes"]["trust_value"] = initial_trust

# -------------------------------
# Run Sensitivity (Before Trust)
# -------------------------------
_, avg_before, sensitivities_before, _, _ = compute_object_sensitivity(
    access_requests_before, storage, policy_attrs
)

# -------------------------------
# Plot Comparison
# -------------------------------
before_map = {o["object_id"]: o["sensitivity"] for o in sensitivities_before}
after_map = {o["object_id"]: o["sensitivity"] for o in sensitivities_after}

object_ids = list(before_map.keys())
x = range(len(object_ids))
bar_width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x, [before_map[oid] for oid in object_ids], width=bar_width, label='Before Trust Drop')
plt.bar([i + bar_width for i in x], [after_map[oid] for oid in object_ids], width=bar_width, label='After Trust Drop')

plt.xlabel("Object ID")
plt.ylabel("Sensitivity Score")
plt.title("Object Sensitivity: Before vs After Trust Drop")
plt.xticks([i + bar_width/2 for i in x], object_ids, rotation=45)
plt.legend()
plt.tight_layout()
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
plot_path = PLOTS_DIR / "insider_object_sensitivity_comparison.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Saved sensitivity comparison plot to {plot_path}")

# -------------------------------
# Save Output
# -------------------------------
OUTDIR = BASE / "datasets" / "processed" / "sensitivity_results"
OUTDIR.mkdir(parents=True, exist_ok=True)

with open(OUTDIR / "insider_target_object_sensitivity.json", "w") as f:
    json.dump({
        "avg_sensitivity_before": float(avg_before),
        "avg_sensitivity_after": float(avg_after),
        "threshold_after": float(threshold_after),
        "sensitive_attributes_after": list(sensitive_attrs_after),
        "detailed_sensitivities_before": json.loads(json.dumps(sensitivities_before, default=lambda o: float(o))),
        "detailed_sensitivities_after": json.loads(json.dumps(sensitivities_after, default=lambda o: float(o))),

    }, f, indent=2)

print("Saved comparison sensitivity results.")
