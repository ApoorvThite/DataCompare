# src/schemas/canonical_schema.py

"""
Canonical schema for the salaries dataset.

All formats (CSV, JSON, TOON) will be converted into this structure.

Fields
- job_id: unique identifier per row
- title: job title (from job_title)
- company_location: country or location of the company
- experience_level: junior, mid, senior, etc
- employment_type: full time, part time, contract, etc
- salary_min: lower bound of salary in USD
- salary_max: upper bound of salary in USD
- currency: currency code of the canonical salary (USD here)
- skills: list of strings, may be empty or derived later
- job_category: coarse category derived from title for ML tasks
"""

CANONICAL_SCHEMA = {
    "job_id": "string",
    "title": "string",
    "company_location": "string",
    "experience_level": "string",
    "employment_type": "string",
    "salary_min": "float",
    "salary_max": "float",
    "currency": "string",
    "skills": "list<string>",
    "job_category": "string",
}

CANONICAL_COLUMN_ORDER = [
    "job_id",
    "title",
    "company_location",
    "experience_level",
    "employment_type",
    "salary_min",
    "salary_max",
    "currency",
    "skills",
    "job_category",
]


def validate_canonical_df(df):
    """
    Lightweight check that the canonical DataFrame has required columns.
    Raises ValueError if something is missing.
    """
    missing = [c for c in CANONICAL_COLUMN_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Canonical DataFrame is missing columns: {missing}")
