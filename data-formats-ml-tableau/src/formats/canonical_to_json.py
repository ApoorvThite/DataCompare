# src/formats/canonical_to_json.py

from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.config import DATA_CANONICAL, DATA_FORMATTED
from src.schemas.canonical_schema import validate_canonical_df


CANONICAL_FILENAME = "salaries_canonical.parquet"
NESTED_JSON_FILENAME = "salaries_nested.json"


def load_canonical() -> pd.DataFrame:
    path = DATA_CANONICAL / CANONICAL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Canonical file not found at {path}")
    df = pd.read_parquet(path)
    validate_canonical_df(df)
    return df


def to_json_number(value):
    """
    Convert value to a JSON safe number or None.
    Handles numpy types and NaN.
    """
    # Handle missing
    try:
        if pd.isna(value):
            return None
    except TypeError:
        # Non scalar, ignore
        pass

    # Native Python number
    if isinstance(value, (int, float)):
        return value

    # Numpy number
    if isinstance(value, (np.integer, np.floating)):
        return value.item()

    # Try casting to float
    try:
        return float(value)
    except Exception:
        return None


def normalize_skills(value):
    """
    Ensure skills is a list of strings, safe for JSON.
    Handles list, tuple, numpy array, scalar, None.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]

    try:
        if pd.isna(value):
            return []
    except TypeError:
        pass

    # If it is a single string or other scalar, wrap in list
    return [str(value)]


def row_to_nested_json(rec: dict) -> dict:
    """
    Convert a canonical row dict into a nested JSON structure.

    Canonical fields:
      job_id, title, company_location, experience_level, employment_type,
      salary_min, salary_max, currency, skills, job_category
    """

    nested = {
        "job_id": str(rec["job_id"]),
        "title": str(rec["title"]),
        "company_location": str(rec["company_location"]),
        "experience_level": str(rec["experience_level"]),
        "employment_type": str(rec["employment_type"]),
        "salary": {
            "min": to_json_number(rec["salary_min"]),
            "max": to_json_number(rec["salary_max"]),
            "currency": str(rec["currency"]),
        },
        "skills": normalize_skills(rec.get("skills", [])),
        "job_category": str(rec["job_category"]),
    }
    return nested


def canonical_to_nested_json(df: pd.DataFrame) -> list[dict]:
    records = df.to_dict(orient="records")
    nested_records = [row_to_nested_json(rec) for rec in records]
    return nested_records


def save_nested_json(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, allow_nan=False)
    print(f"Saved nested JSON to {path}")


def main():
    print("Loading canonical dataset...")
    df_canonical = load_canonical()

    print("Converting canonical dataset to nested JSON representation...")
    nested_records = canonical_to_nested_json(df_canonical)

    json_path = DATA_FORMATTED / NESTED_JSON_FILENAME
    save_nested_json(nested_records, json_path)

    print("Preview of first nested JSON record:")
    if nested_records:
        print(json.dumps(nested_records[0], indent=2))


if __name__ == "__main__":
    main()