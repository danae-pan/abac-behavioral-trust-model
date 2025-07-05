import json
import random
from pathlib import Path
from faker import Faker

"""
generate_attack_subjects.py

Generates a synthetic subject pool for the insider threat simulation experiment.
- Includes one insider student with access to a shared course (MATH101)
- Adds 4 valid peers with the same course access and similar trust levels
- Adds 8 irrelevant students enrolled in different courses

All subjects share the same school, district, and county attributes to ensure
attribute alignment in ABAC policy evaluation. The generated dataset is saved to:
  datasets/processed/all_attack_subjects.json
"""

fake = Faker()
random.seed(42)

BASE = Path(__file__).resolve().parents[1]
OUTPUT = BASE / "datasets" / "processed" / "all_attack_subjects.json"

# Hardcoded shared info
school_code = "SCH001"
school_name = "Stellar High"
district_name = "North Ridge"
district_id = "NR-101"
county_name = "Kings County"
county_number = "KC-9"
state = "California"
shared_course_code = "MATH101"

students = []

# Insider
insider = {
    "subject_id": str(random.randint(7000000, 7999999)),
    "role": "Student",
    "school_code": [school_code],
    "school_name": [school_name],
    "district_name": [district_name],
    "district_id": [district_id],
    "county_name": [county_name],
    "county_number": [county_number],
    "state": state,
    "course_code": [shared_course_code],
    "trust_value": round(random.uniform(0.7, 0.9), 2),
}
students.append(insider)

# 4 valid peers with access to MATH101
for _ in range(4):
    peer = insider.copy()
    peer["subject_id"] = str(random.randint(7000000, 7999999))
    peer["trust_value"] = round(random.uniform(0.7, 0.95), 2)
    students.append(peer)

# 8 irrelevant students with different course access
for _ in range(8):
    s = insider.copy()
    s["subject_id"] = str(random.randint(7000000, 7999999))
    s["course_code"] = [fake.bothify("???###")]
    s["trust_value"] = round(random.uniform(0.5, 0.95), 2)
    students.append(s)

with open(OUTPUT, "w") as f:
    json.dump(students, f, indent=2)

print(f"Saved {len(students)} targeted subjects to {OUTPUT}")
