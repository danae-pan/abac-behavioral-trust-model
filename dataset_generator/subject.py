import json
import random
from faker import Faker
from pathlib import Path

fake = Faker()

CLASS_IDS = [f"CLS{str(i).zfill(5)}" for i in range(1, 1001)]
COURSE_CODES = [str(1000 + i) for i in range(1000)]
ADMIN_POSITIONS = ["Academic", "Nominated"]
TEACHER_POSITIONS = ["Probationary", "Long Term", "Tenured"]

def random_sample(data, min_n=1, max_n=3):
    """
    Returns a random sample from the list between min_n and max_n elements.
    """
    if not data:
        return []
    if len(data) < min_n:
        return data
    return random.sample(data, random.randint(min_n, min(max_n, len(data))))


def generate_subjects_for_school(school, total_admins_state, total_students_state, county_data):
    """
    Generates Admins, Teachers, and Students for a specific school.
    Uses ratios from the state-level summary to scale subject counts.
    """
    try:
        num_students = int(school["num_students"])
    except:
        return []

    if num_students == 0:
        return []

    try:
        ratio = float(school["pupil_teacher_ratio"])
    except:
        ratio = 21.1  # fallback default

    num_teachers = int(round(num_students / ratio))
    num_admins = int(round((num_students / total_students_state) * total_admins_state))

    subjects = []

    school_code = school["school_id"]
    district_name = school["district_name"]
    county_name = school["county_name"]
    state = school["state_name_label"]
    school_name = school["school_name"]
    district_id = school["district_id"]
    county_number = school["county_number"]

    districts_by_county = {}
    district_id_to_name = {}
    school_code_to_name = {}

    for d_name, d_data in county_data["districts"].items():
        for sch in d_data["schools"]:
            d_id = sch["district_id"]
            s_id = sch["school_id"]
            s_name = sch["school_name"]

            districts_by_county.setdefault(d_id, []).append(s_id)
            district_id_to_name[d_id] = d_name
            school_code_to_name[s_id] = s_name

    # Admins
    for _ in range(num_admins):
        district_scope = random_sample(list(districts_by_county.keys()), min_n=2, max_n=4)
        school_scope = []
        for d_id in district_scope:
            school_scope.extend(random_sample(districts_by_county[d_id], min_n=1, max_n=3))

        district_name_scope = [district_id_to_name[d] for d in district_scope]
        school_name_scope = [school_code_to_name[s] for s in school_scope]

        subjects.append({
            "subject_id": str(random.randint(1000000, 3249999)),
            "role": "Admin",
            "position": random_sample(ADMIN_POSITIONS),
            "trust_value": round(random.uniform(0.5, 1.0), 2),
            "county_name": [county_name],
            "county_number": [county_number],
            "district_id": district_scope,
            "district_name": district_name_scope,
            "school_code": school_scope,
            "school_name": school_name_scope,
            "class_id": random_sample(CLASS_IDS),
            "course_code": random_sample(COURSE_CODES),
            "state": state
        })

    # Teachers
    for _ in range(num_teachers):
        subjects.append({
            "subject_id": str(random.randint(3250000, 5499999)),
            "role": "Teacher",
            "position": random.choice(TEACHER_POSITIONS),
            "trust_value": round(random.uniform(0.5, 1.0), 2),
            "county_name": [county_name],
            "county_number": [county_number],
            "district_name": [district_name],
            "district_id": [district_id],
            "school_code": [school_code],
            "school_name": [school_name],
            "class_id": random_sample(CLASS_IDS, min_n=2, max_n=4),
            "course_code": random_sample(COURSE_CODES, min_n=2, max_n=4),
            "state": state
        })

    # Students
    for _ in range(num_students):
        subjects.append({
            "subject_id": str(random.randint(5500000, 7749999)),
            "role": "Student",
            "position": None,
            "trust_value": round(random.uniform(0.5, 1.0), 2),
            "county_name": [county_name],
            "county_number": [county_number],
            "district_name": [district_name],
            "district_id": [district_id],
            "school_code": [school_code],
            "school_name": [school_name],
            "class_id": [random.choice(CLASS_IDS)],
            "course_code": random_sample(COURSE_CODES, min_n=2, max_n=5),
            "state": state
        })

    return subjects


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parents[1]

    input_file = ROOT_DIR / "datasets" / "processed" / "schools_by_state.json"
    summary_file = ROOT_DIR / "datasets" / "processed" / "state_admins_summary.json"
    output_file = ROOT_DIR / "datasets" / "processed" / "all_subjects.json"

    if not input_file.exists() or not summary_file.exists():
        print("Missing input file(s).")
        exit()

    with input_file.open(encoding="utf-8") as f:
        schools_by_state = json.load(f)

    with summary_file.open(encoding="utf-8") as f:
        state_admins_summary = json.load(f)

    admin_lookup = {entry["state"]: entry["total_admins"] for entry in state_admins_summary}
    all_subjects = []

    for state, data in schools_by_state.items():
        total_admins_state = admin_lookup.get(state.upper(), 0)
        counties = data.get("counties", {})

        all_schools = []
        for county_name, county_data in counties.items():
            for district_name, district_data in county_data["districts"].items():
                for school in district_data["schools"]:
                    school["county_name"] = county_name
                    school["district_name"] = district_name
                    school["state_name_label"] = state
                    all_schools.append((school, county_data))

        selected_schools = random.sample(all_schools, min(3, len(all_schools)))
        total_students_state = sum(
            int(school["num_students"]) for school, _ in all_schools
            if str(school["num_students"]).isdigit()
        )

        for school, county_data in selected_schools:
            all_subjects.extend(
                generate_subjects_for_school(school, total_admins_state, total_students_state, county_data)
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_subjects, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_subjects)} subjects to {output_file}")