import csv
import json
from pathlib import Path

"""
build_state_admins.py

Parses a state-level staff statistics CSV file from the NCES dataset to compute:
- Total administrator staff count per state (LEA + school level, including support staff)
- Pupil-teacher ratio per state

Outputs a JSON summary used for generating realistic administrator distributions
in educational ABAC simulations.

Expected input: datasets/raw_csv/nces_staff_statewise_2023_2024.csv
Output written to: datasets/processed/state_admins_summary.json
"""

def extract_admin_summary(input_csv: str, output_json: str):
    """
    Converts a state-level CSV file from NCES into a JSON summary
    that includes total administrator count and pupil-teacher ratio per state.
    """
    states_data = []

    if not Path(input_csv).exists():
        print(f"Input file not found: {input_csv}")
        return

    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            try:
                state_name = row["State Name"].strip().upper()
                lea_admins = float(row["LEA Administrators"])
                lea_support = float(row["LEA Administrative Support Staff"])
                school_admins = float(row["School Administrators"])
                school_support = float(row["School Administrative Support Staff"])
                ratio = float(row["Pupil/Teacher Ratio"])

                total_admins = round(lea_admins + lea_support + school_admins + school_support, 2)

                states_data.append({
                    "state": state_name,
                    "total_admins": total_admins,
                    "pupil_teacher_ratio": ratio
                })

            except (ValueError, KeyError) as e:
                print(f"Skipping row due to error: {e}")
                continue

    with open(output_json, "w", encoding="utf-8") as jsonfile:
        json.dump(states_data, jsonfile, indent=2, ensure_ascii=False)

    print(f"Saved {len(states_data)} states to {output_json}")


# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parents[1] 
    INPUT = ROOT_DIR / "datasets" / "raw_csv" / "nces_staff_statewise_2023_2024.csv"
    OUTPUT = ROOT_DIR / "datasets" / "processed" / "state_admins_summary.json"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)  
    extract_admin_summary(INPUT, OUTPUT)
