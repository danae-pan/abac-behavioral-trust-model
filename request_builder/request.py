import json
import random
from collections import defaultdict, Counter
from pathlib import Path

# ----------------------
# Configuration
# ----------------------
ROOT = Path(__file__).resolve().parents[1]
SUBJECTS_PATH = ROOT / "datasets" / "processed" / "all_subjects.json"
OBJECTS_PATH = ROOT / "datasets" / "processed" / "all_objects.json"
OUTPUT_PATH = ROOT / "datasets" / "processed" / "access_requests.json"
MAX_REQUESTS = 100000
INVALID_RATIO = 0.1

# ----------------------
# Load subject and object data
# ----------------------
with open(SUBJECTS_PATH, encoding="utf-8") as f:
    subjects = json.load(f)

with open(OBJECTS_PATH, encoding="utf-8") as f:
    objects = json.load(f)

# ----------------------
# Group data by district
# ----------------------
subjects_by_districts = defaultdict(list)
for subj in subjects:
    districts = subj.get("district_name") or []
    if isinstance(districts, str):
        districts = [districts]
    for district in districts:
        if district:
            subjects_by_districts[district].append(subj)

objects_by_districts = defaultdict(list)
for obj in objects:
    districts = obj.get("district_name") or []
    if isinstance(districts, str):
        districts = [districts]
    for district in districts:
        if district:
            objects_by_districts[district].append(obj)

# ----------------------
# Maps defining valid actions and object types per role
# ----------------------
action_map = {
    "student enrollment": ["view", "edit"],
    "course enrollment": ["view", "edit", "delete", "enroll"],
    "student records": ["view"],
    "course material": ["access"],
    "course assignment": ["submit"]
}

role_object_scope = {
    "Student": ["course assignment", "course enrollment", "course material"],
    "Teacher": ["student records"],
    "Admin": ["student enrollment", "course enrollment"]
}

# ----------------------
# Request Generation Function
# ----------------------
def generate_requests_same_district(max_requests=MAX_REQUESTS, invalid_ratio=INVALID_RATIO):
    requests = []
    attempts = 0
    max_attempts = max_requests * 5

    while len(requests) < max_requests and attempts < max_attempts:
        attempts += 1
        district = random.choice(list(subjects_by_districts.keys()))
        subj = random.choice(subjects_by_districts[district])
        subj_role = subj["role"]
        allowed_obj_types = role_object_scope.get(subj_role, [])

        candidate_types = (
            [t for t in action_map if t not in allowed_obj_types]
            if random.random() < invalid_ratio else
            allowed_obj_types
        )
        if not candidate_types:
            continue

        obj_type = random.choice(candidate_types)
        matching_objs = [o for o in objects_by_districts[district] if o["obj_type"] == obj_type]
        if not matching_objs:
            continue

        obj = random.choice(matching_objs)
        action = random.choice(action_map.get(obj_type, ["view"]))

        subject_attrs = {
            "role": subj_role,
            "trust_value": subj["trust_value"],
            "position": subj.get("position"),
            "district_id": subj.get("district_id", []),
            "county_code": subj.get("county_number", []),
            "school_code": subj.get("school_code", []),
            "class_id": subj.get("class_id", []),
            "course_code": subj.get("course_code", [])
        }

        resource_attrs = {
            "obj_type": obj["obj_type"],
            "school_code": obj.get("school_code"),
            "class_id": obj.get("class_id"),
            "course_code": obj.get("course_code"),
            "district_id": obj.get("district_id"),
            "county_code": obj.get("county_number"),
            "file_created": obj.get("file_created"),
            "remaining_seats": obj.get("remaining_seats")
        }

        request = {
            "subject": {"id": subj["subject_id"], "attributes": subject_attrs},
            "resource": {"id": obj["object_id"], "attributes": resource_attrs},
            "action": {"id": f"{action}_action", "attributes": {"method": action}},
            "context": {}
        }

        requests.append(request)

    return requests

# ----------------------
# Generate and Save
# ----------------------
if __name__ == "__main__":
    all_requests = generate_requests_same_district()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_requests, f, indent=2, ensure_ascii=False)

    distribution = Counter(
        (r["subject"]["attributes"]["role"], r["resource"]["attributes"]["obj_type"])
        for r in all_requests
    )
    print("\nRequest Distribution by Role and Object Type:")
    for (role, obj_type), count in sorted(distribution.items()):
        print(f"- {role:<8} → {obj_type:<20}: {count}")
    print(f"\nSaved {len(all_requests)} requests to: {OUTPUT_PATH}")
