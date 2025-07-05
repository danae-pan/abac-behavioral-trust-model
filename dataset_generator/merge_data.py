import json
from pathlib import Path

"""
merge_data.py

Merges flat school-level data with summarized state-level statistics to produce a
nested JSON structure organized by state, county, and district.

Each state entry includes:
- Total number of administrative staff (from NCES state summary)
- Pupil-teacher ratio
- Grouped data for counties, districts, and schools

Used as the basis for generating realistic ABAC subject attributes and for distributing
subjects across administrative units during simulation.

Inputs:
- datasets/processed/schools_data.json
- datasets/processed/state_admins_summary.json

Output:
- datasets/processed/schools_by_state.json
"""

def merge_school_and_state_data(school_data_path: str, state_summary_path: str, output_path: str):
    """
    Combines flat school-level data with state-level statistics to create a nested
    structure grouped by state, county, and district.
    """
    if not Path(school_data_path).exists() or not Path(state_summary_path).exists():
        print("Missing input file(s).")
        return

    with open(school_data_path, "r", encoding="utf-8") as f:
        flat_schools = json.load(f)

    with open(state_summary_path, "r", encoding="utf-8") as f:
        state_summary = {entry["state"]: entry for entry in json.load(f)}

    structured_data = {}

    for school in flat_schools:
        if not school or "state_name" not in school or not school["state_name"]:
            continue

        state = school.get("state_name_label", "").strip().upper()
        county = school.get("county_name", "Unknown County")
        district = school.get("district_name", "Unknown District")
        school_name = school.get("school_name", "Unnamed School")

        if state not in structured_data:
            structured_data[state] = {
                "total_admins": state_summary.get(state, {}).get("total_admins"),
                "pupil_teacher_ratio": state_summary.get(state, {}).get("pupil_teacher_ratio"),
                "counties": {}
            }

        counties = structured_data[state]["counties"]
        if county not in counties:
            counties[county] = {"districts": {}}

        districts = counties[county]["districts"]
        if district not in districts:
            districts[district] = {"schools": []}

        districts[district]["schools"].append({
            "school_name": school_name,
            "school_id": school.get("school_id"),
            "district_id": school.get("district_id"),
            "county_number": school.get("county_number"),
            "num_students": school.get("num_students"),
            "num_teachers": school.get("num_teachers"),
            "pupil_teacher_ratio": school.get("pupil_teacher_ratio")
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    print(f"Saved structured data to {output_path}")

# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parents[1]  # points to 'src/'

    INPUT_SCHOOLS = ROOT_DIR / "datasets" / "processed" / "schools_data.json"
    INPUT_STATE_SUMMARY = ROOT_DIR / "datasets" / "processed" / "state_admins_summary.json"
    OUTPUT = ROOT_DIR / "datasets" / "processed" / "schools_by_state.json"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merge_school_and_state_data(INPUT_SCHOOLS, INPUT_STATE_SUMMARY, OUTPUT)