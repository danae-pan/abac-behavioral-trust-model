# Extending the Attribute Based Access Control (ABAC) with Honey-Attributes for Insider Threat Detection  
## A Trust Aware Framework

This repository contains the implementation of the thesis project:

**Panagiotopoulou, D. (2025). _Extending the Attribute Based Access Control (ABAC) with Honey-Attributes for Insider Threat Detection: A Trust Aware Framework_. Master's Thesis, Technical University of Denmark.**

---

## Overview

This project implements a trust-aware extension of ABAC for insider threat detection. The approach evaluates user behavior over time and recalibrates trust using **Fuzzy Analytic Hierarchy Process (FAHP)**. 

Rather than generating honey attributes, the framework assumes their existence and uses interactions with them as behavioral evidence. These interactions influence both user trust and attribute sensitivity, enabling more risk-aware and adaptive access decisions.


---

## Project Structure

```
src/
├── dataset_generator/          # Generates subject/object datasets
├── request_builder/            # Builds ABAC access requests from inputs
├── trust_estimator/           # Simulates trust evolution using FAHP
├── sensitivity_estimator/     # Computes object and attribute sensitivity
├── evaluation/
│   ├── main/                   # Core experiments (Exp1–Exp4)
│   └── appendix/               # Supplementary experiments for thesis appendix
├── datasets/
│   └── raw_csv/               # Sample NCES files to bootstrap pipeline
│   └── processed/             # Auto-generated data (subjects, objects, trust, sensitivity)
├── README.md
└── requirements.txt
```

---

## Environment Setup

1. **Python version:** 3.10.x  
2. **Install dependencies:**

    ```bash
    python3.10 -m venv abac-env310
    source abac-env310/bin/activate
    pip install -r requirements.txt
    ```

3. (Optional) Set local Python version:

    ```bash
    echo "abac-env310" > .python-version
    ```

---

## Running the Pipeline

Ensure you have NCES-style CSVs under `datasets/raw_csv/`, then run:

### 1. Dataset generation
```bash
python src/dataset_generator/build_state_admins.py
python src/dataset_generator/get_data.py
python src/dataset_generator/merge_data.py
python src/dataset_generator/subject.py
python src/dataset_generator/object.py
```

### 2. Request building
```bash
python src/request_builder/request.py
```

### 3. Trust simulation (default FAHP setup)
```bash
python src/trust_estimator/trustfahp-default.py
```

### 4. Sensitivity analysis
```bash
python src/sensitivity_estimator/sensitivity.py
```

---

## Running Experiments

### Main Experiments

- **Exp1 and Exp2** can be run directly:
    ```bash
    python src/evaluation/main/exp1_attribute_diagnostics.py
    python src/evaluation/main/exp2_risk_evaluation.py
    ```

- **Exp3 and Exp4** must be run using `-m` due to internal module imports:
    ```bash
    python -m evaluation.main.exp3_trust_variation
    python -m evaluation.main.exp4_action_thresholds
    ```

### Appendix Experiments

These are standalone and can be run directly:

```bash
python src/evaluation/appendix/trustfahp-gamma.py
python src/evaluation/appendix/trustfahp-betadistro.py
python src/evaluation/appendix/trustfahp-tfns.py
```

---

## Notes

- All intermediate and final outputs are saved in `datasets/processed/`
- This includes `all_subjects.json`, `all_objects.json`, access requests, trust records, and sensitivity values
- The `raw_csv` folder includes example CSVs to help you run the pipeline and observe sample outputs. For full-scale use, custom data should be generated, following the header convention from the sample files.
- Public educational data used for dataset generation can be retrieved via the [NCES Table Generator](https://nces.ed.gov/ccd/elsi/tableGenerator.aspx). This project used:
  - **State-level staffing data** (2023–2024): used to estimate administrative staff counts by role and region.
  - **School-level data** (2023–2024): including district, county, school name, and enrollment size.

---

## Acknowledgements

This project relies on the following open-source libraries:

- [`py-abac`](https://github.com/ketgo/py-abac) — Attribute-Based Access Control engine used for policy parsing and enforcement
- [`pyfdm`](https://github.com/jwieckowski/pyfdm) — Fuzzy decision-making framework used for fuzzy operations on the trust simulation

---

## Citation

Please cite this work as:

> Panagiotopoulou, D. (2025). _Extending the Attribute Based Access Control (ABAC) with Honey-Attributes for Insider Threat Detection: A Trust Aware Framework_. Master's Thesis, Technical University of Denmark.

---
