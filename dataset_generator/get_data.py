import csv
import json
from pathlib import Path

"""
get_data.py

Parses a school-level CSV dataset downloaded from the NCES ELSI table generator and extracts
relevant fields such as school name, district, county, student/teacher count, and ratio.
Outputs a cleaned JSON file that can be used for synthetic subject generation in ABAC modeling.

Usage:
  - Expects input file in datasets/raw_csv/nces_schools_2023-2024.csv
  - Produces output JSON in datasets/processed/schools_data.json
"""

def extract_school_data(input_csv: str, output_json: str):
    """
    Extracts school-level data from a CSV file downloaded from the NCES ELSI table generator.
    Saves a structured list of school records to a JSON file.
    """
    schools = []

    if not Path(input_csv).exists():
        print(f"Input file not found: {input_csv}")
        return

    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for idx, row in enumerate(reader, 1):
            try:
                total_students_key = next(k for k in row if "Total Students" in k)
                total_teachers_key = next(k for k in row if "Total Teachers" in k)
                ratio_key = next(k for k in row if "Pupil/Teacher Ratio" in k)

                def safe_strip(val):
                    return val.strip() if isinstance(val, str) else val

                schools.append({
                    "school_name": safe_strip(row.get("School Name", "")),
                    "state_name": safe_strip(row.get("State Name", "")),
                    "state_name_label": safe_strip(row.get("State Name [Public School]", "")),
                    "school_id": safe_strip(row.get("School ID", "")),
                    "district_id": safe_strip(row.get("District ID", "")),
                    "district_name": safe_strip(row.get("District Name", "")),
                    "county_name": safe_strip(row.get("County Name", "")),
                    "county_number": safe_strip(row.get("County Number", "")),
                    "num_students": safe_strip(row.get(total_students_key, "")),
                    "num_teachers": safe_strip(row.get(total_teachers_key, "")),
                    "pupil_teacher_ratio": safe_strip(row.get(ratio_key, ""))
                })

            except Exception as e:
                print(f"[Row {idx}] Skipped due to error: {e}")
                continue

    with open(output_json, "w", encoding="utf-8") as jsonfile:
        json.dump(schools, jsonfile, indent=2, ensure_ascii=False)

    print(f"Saved {len(schools)} schools to {output_json}")

# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parents[1] 
    INPUT = ROOT_DIR / "datasets" / "raw_csv" / "nces_schools_2023-2024.csv"
    OUTPUT = ROOT_DIR / "datasets" / "processed" / "schools_data.json"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    extract_school_data(INPUT, OUTPUT)