import json
import random
import time
from pathlib import Path
from faker import Faker

"""
generate_attack_objects.py

Generates a set of course assignment objects for the insider threat simulation.
Each object is aligned with the insider's course, school, district, and state attributes
to ensure realistic access request scenarios.

This script reads the insider subject from the processed subject file and uses their
attributes to produce ~10 course assignment objects. These objects are intended to be
used in sensitivity analysis before and after trust degradation.

Output:
- Saves the generated objects to 'datasets/processed/all_attack_objects.json'
"""

fake = Faker()
random.seed(42)

BASE = Path(__file__).resolve().parents[1]
SUBJECTS = BASE / "datasets" / "processed" / "all_attack_subjects.json"
OUTPUT = BASE / "datasets" / "processed" / "all_attack_objects.json"

with open(SUBJECTS) as f:
    subjects = json.load(f)

# Use insider's attributes for object alignment
insider = subjects[0]
school_code = insider["school_code"][0]
school_name = insider["school_name"][0]
district_name = insider["district_name"][0]
district_id = insider["district_id"][0]
county_name = insider["county_name"][0]
county_number = insider["county_number"][0]
state = insider["state"]
shared_course_code = insider["course_code"][0]

objects = []

# Generate ~10 course assignments matching the insider’s course
for _ in range(10):
    obj = {
        "object_id": str(random.randint(8000000, 8999999)),
        "obj_type": "course assignment",
        "school_code": school_code,
        "school_name": school_name,
        "district_name": district_name,
        "district_id": district_id,
        "county_name": county_name,
        "county_number": county_number,
        "state": state,
        "course_code": shared_course_code,
        "file_created": int(time.mktime(fake.date_time_this_year().timetuple())),
    }
    objects.append(obj)

with open(OUTPUT, "w") as f:
    json.dump(objects, f, indent=2)

print(f"Saved {len(objects)} course assignment objects to {OUTPUT}")
