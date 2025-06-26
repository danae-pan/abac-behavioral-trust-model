import json
import random
import time
from collections import defaultdict
from faker import Faker
from pathlib import Path

fake = Faker()

# Configuration
CLASS_IDS = [f"CLS{str(i).zfill(5)}" for i in range(1, 1001)]
COURSE_CODES = [str(1000 + i) for i in range(1000)]

type_weights = {
    "course material": 3,
    "course assignment": 3,
    "course enrollment": 2,
    "student enrollment": 1,
    "student records": 1
}
obj_types = list(type_weights.keys())

def build_object_entry(school_code, school_name, district_name, district_id,
                       county_name, county_number, state, obj_type):
    obj = {
        "object_id": str(random.randint(7750000, 9999999)),
        "obj_type": obj_type,
        "state": state,
        "school_code": school_code,
        "school_name": school_name,
        "district_name": district_name,
        "district_id": district_id,
        "county_name": county_name,
        "county_number": county_number,
    }

    if obj_type == "course enrollment":
        enroll_limit = random.randint(10, 50)
        enroll_total = random.randint(1, enroll_limit)
        obj.update({
            "enroll_total": enroll_total,
            "enroll_limit": enroll_limit,
            "remaining_seats": enroll_limit - enroll_total,
            "course_code": random.choice(COURSE_CODES),
        })

    elif obj_type == "course assignment":
        obj.update({
            "file_created": int(time.mktime(fake.date_time_this_year().timetuple())),
            "course_code": random.choice(COURSE_CODES),
        })

    elif obj_type == "course material":
        obj["course_code"] = random.choice(COURSE_CODES)

    elif obj_type == "student records":
        obj["class_id"] = random.choice(CLASS_IDS)

    return obj

def generate_objects_for_school(school_code, subjects, scale_factor=0.2):
    num_objects = max(5, int(len(subjects) * scale_factor))
    subject = subjects[0]

    district_name = subject.get("district_name", [None])[0]
    district_id = subject.get("district_id", [None])[0]
    county_name = subject.get("county_name", [None])[0]
    county_number = subject.get("county_number", [None])[0]
    school_name = subject.get("school_name", [None])[0]
    state = subject.get("state")

    objects = []

    for obj_type in obj_types:
        objects.append(build_object_entry(
            school_code, school_name, district_name, district_id,
            county_name, county_number, state, obj_type
        ))

    remaining = num_objects - len(objects)
    weighted_choices = random.choices(obj_types, weights=type_weights.values(), k=remaining)

    for obj_type in weighted_choices:
        objects.append(build_object_entry(
            school_code, school_name, district_name, district_id,
            county_name, county_number, state, obj_type
        ))

    return objects

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]  # Points to 'src'
    INPUT = ROOT / "datasets" / "processed" / "all_subjects.json"
    OUTPUT = ROOT / "datasets" / "processed" / "all_objects.json"

    if not INPUT.exists():
        print(f"Input file not found: {INPUT}")
        exit()

    with open(INPUT, encoding="utf-8") as f:
        all_subjects = json.load(f)

    subjects_by_school = defaultdict(list)
    for subj in all_subjects:
        key = subj.get("school_code")
        if isinstance(key, list):
            key = key[0] if key else None
        if key:
            subjects_by_school[key].append(subj)

    all_objects = []
    for school_code, subjects in subjects_by_school.items():
        all_objects.extend(generate_objects_for_school(school_code, subjects))

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_objects, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(all_objects)} objects across {len(subjects_by_school)} schools.")
    print(f"Output saved to: {OUTPUT}")
