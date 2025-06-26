import json
import copy
import matplotlib.pyplot as plt
from py_abac.storage.memory import MemoryStorage
from py_abac import Policy, Request, PDP
from sensitivity_estimator.sensitivity import compute_object_sensitivity
from pathlib import Path

# ------------------------------
# Experiment 3: Trust Variation Impact on Sensitivity
# ------------------------------

BASE = Path(__file__).resolve().parents[2]

POLICY_PATH = BASE / "request_builder" / "policy.json"
SUBJECTS_PATH = BASE / "datasets" / "processed" / "all_subjects.json"
TRUST_RECORDS_PATH = BASE / "datasets" / "processed" / "trust_estimator" / "trust_records_fahp_random_params.json"
REQUESTS_PATH = BASE / "datasets" / "processed" / "access_requests.json"

# Load ABAC policies
with open(POLICY_PATH) as f:
    policy_data = json.load(f)["policies"]

storage = MemoryStorage()
for p in policy_data:
    storage.add(Policy.from_json(p))

# Extract relevant attributes from policy rules
policy_attrs = {
    attr[2:]
    for p in policy_data
    for attr in p.get("rules", {}).get("resource", {})
    if attr.startswith("$.")
}

# Load initial and final trust values
with open(SUBJECTS_PATH) as f:
    subjects = json.load(f)
initial_trust_map = {s["subject_id"]: s["trust_value"] for s in subjects}

with open(TRUST_RECORDS_PATH) as f:
    trust_records = json.load(f)
final_trust_map = {sid: rec[-1] for sid, rec in trust_records.items()}

# Load access requests
with open(REQUESTS_PATH) as f:
    all_requests = json.load(f)

# Group subject IDs by trust direction
increased_ids = [sid for sid in final_trust_map if final_trust_map[sid] > initial_trust_map.get(sid)]
decreased_ids = [sid for sid in final_trust_map if final_trust_map[sid] < initial_trust_map.get(sid)]

# Split requests by trust group
reqs_increase = [r for r in all_requests if r["subject"]["id"] in increased_ids]
reqs_decrease = [r for r in all_requests if r["subject"]["id"] in decreased_ids]

def patch_trust(requests, trust_map):
    patched = copy.deepcopy(requests)
    for r in patched:
        sid = r["subject"]["id"]
        r["subject"]["attributes"]["trust_value"] = trust_map.get(sid)
    return patched

reqs_inc_initial = patch_trust(reqs_increase, initial_trust_map)
reqs_inc_updated = patch_trust(reqs_increase, final_trust_map)
reqs_dec_initial = patch_trust(reqs_decrease, initial_trust_map)
reqs_dec_updated = patch_trust(reqs_decrease, final_trust_map)

# Evaluate sensitivity and denial rate for each group
def get_avg_sensitivity(reqs):
    threshold, avg_sens, *_ = compute_object_sensitivity(reqs, storage, policy_attrs)
    return avg_sens

def get_denial_rate(reqs):
    pdp = PDP(storage)
    denied = sum(1 for r in reqs if not pdp.is_allowed(Request.from_json(r)))
    return denied / len(reqs) if reqs else 0.0

print(f"\n🔍 Request Counts:")
print(f"  Trust ↑ Requests: {len(reqs_inc_initial)}")
print(f"  Trust ↓ Requests: {len(reqs_dec_initial)}")

print(f"\n🧪 Subject Counts:")
print(f"  Trust ↑ Subjects: {len(increased_ids)}")
print(f"  Trust ↓ Subjects: {len(decreased_ids)}")

sens_inc_init = get_avg_sensitivity(reqs_inc_initial)
sens_inc_upd = get_avg_sensitivity(reqs_inc_updated)
sens_dec_init = get_avg_sensitivity(reqs_dec_initial)
sens_dec_upd = get_avg_sensitivity(reqs_dec_updated)

denial_inc_init = get_denial_rate(reqs_inc_initial)
denial_inc_upd = get_denial_rate(reqs_inc_updated)
denial_dec_init = get_denial_rate(reqs_dec_initial)
denial_dec_upd = get_denial_rate(reqs_dec_updated)

print("Denial Rates:")
print(f"  Trust Increased Initial: {denial_inc_init:.2%} → Updated: {denial_inc_upd:.2%}")
print(f"  Trust Decreased  Initial: {denial_dec_init:.2%} → Updated: {denial_dec_upd:.2%}")

print("Sensitivity:")
print(f"  Trust Increased Initial: {sens_inc_init:.4f} → Updated: {sens_inc_upd:.4f}")
print(f"  Trust Decreased Initial: {sens_dec_init:.4f} → Updated: {sens_dec_upd:.4f}")

# ------------------------------
# Plot results
# ------------------------------
labels = ["Trust Increased", "Trust Decreased"]
x = range(len(labels))
bar_width = 0.35

initial_vals = [sens_inc_init, sens_dec_init]
updated_vals = [sens_inc_upd, sens_dec_upd]

plt.bar([i - bar_width/2 for i in x], initial_vals, width=bar_width, label="Initial Trust")
plt.bar([i + bar_width/2 for i in x], updated_vals, width=bar_width, label="Updated Trust")

plt.xticks(x, labels)
plt.ylabel("Average Object Sensitivity")
plt.title("Sensitivity Change by Trust Evolution")
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
