import json
from pathlib import Path
from py_abac import PDP, Policy, AccessRequest
from py_abac.storage.memory import MemoryStorage

"""
exp3_target_object.py


Identifies target objects for insider attack simulation.
Selects objects that were initially accessible but later denied due to trust degradation.

Steps:
1. Loads insider trust trajectory and policy rules.
2. Filters candidate objects of type "course assignment" that match the insider's school and course codes.
3. Evaluates access permission before and after the trust drop using ABAC policies.
4. Selects only those objects that were allowed initially but denied after trust decreased.

Usage:
- Requires:
    - Processed subject and object datasets (`all_attack_subjects.json`, `all_attack_objects.json`)
    - ABAC policy rules (`policy.json`)
    - Insider trust history (`trust_ip_theft_insider.json`)
- Output:
    - Saves list of object IDs denied only after trust drop to `datasets/processed/target_objects.json`.

"""

# -------------------------------
# Paths
# -------------------------------
BASE = Path(__file__).resolve().parents[2]
SUBJECTS_PATH = BASE / "datasets" / "processed" / "all_attack_subjects.json"
OBJECTS_PATH = BASE / "datasets" / "processed" / "all_attack_objects.json"
POLICY_PATH = BASE / "request_builder" / "policy.json"
TRUST_PATH = BASE / "datasets" / "processed" / "trust_estimator" / "trust_ip_theft_insider.json"
OUTPUT_PATH = BASE / "datasets" / "processed" / "target_objects.json"

# -------------------------------
# Load data
# -------------------------------
with open(SUBJECTS_PATH) as f:
    subjects = json.load(f)
with open(OBJECTS_PATH) as f:
    objects = json.load(f)
with open(POLICY_PATH) as f:
    policy_data = json.load(f)["policies"]
with open(TRUST_PATH) as f:
    trust_data = json.load(f)

# Load PDP
storage = MemoryStorage()
for p in policy_data:
    storage.add(Policy.from_json(p))
pdp = PDP(storage)

# Get insider info
insider_id = list(trust_data.keys())[0]
initial_subject = next(s for s in subjects if str(s["subject_id"]) == insider_id)
initial_trust = initial_subject["trust_value"]
final_trust = trust_data[insider_id][-1]

# Filter objects matching school and course (handle list values properly)
school_codes = set(initial_subject.get("school_code", []))
course_codes = set(initial_subject.get("course_code", []))

candidate_objects = [
    o for o in objects
    if o["obj_type"] == "course assignment"
    and o.get("school_code") in school_codes
    and o.get("course_code") in course_codes
    and "file_created" in o
]

# Evaluate access pre/post trust degradation
denied_due_to_trust = []
for obj in candidate_objects:
    base_req = {
        "subject": {
            "id": str(insider_id),
            "attributes": {**initial_subject}
        },
        "resource": {
            "id": obj["object_id"],
            "attributes": obj
        },
        "action": {
            "id": "submit",
            "attributes": {"method": "submit"}
        },
        "context": {}
    }

    # Evaluate with initial trust
    base_req["subject"]["attributes"]["trust_value"] = initial_trust
    allowed_before = pdp.is_allowed(AccessRequest.from_json(base_req))

    # Evaluate with final trust
    base_req["subject"]["attributes"]["trust_value"] = final_trust
    denied_after = not pdp.is_allowed(AccessRequest.from_json(base_req))

    if allowed_before and denied_after:
        denied_due_to_trust.append(obj["object_id"])

print(f"Found {len(denied_due_to_trust)} target objects denied due to trust")

with open(OUTPUT_PATH, "w") as f:
    json.dump(denied_due_to_trust, f, indent=2)

