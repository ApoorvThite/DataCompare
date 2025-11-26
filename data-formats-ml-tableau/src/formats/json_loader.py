# JSON loader placeholder
# src/formats/json_loader.py

from pathlib import Path
import json
import pandas as pd

from src.config import DATA_CANONICAL, DATA_FORMATTED
from src.schemas.canonical_schema import (
    CANONICAL_COLUMN_ORDER,
    validate_canonical_df,
)


CANONICAL_FILENAME = "salaries_canonical.parquet"
NESTED_JSON_FILENAME = "salaries_nested.json"


def load_nested_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Nested JSON not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top level JSON array of records.")
    return data


def nested_to_canonical_record(rec: dict) -> dict:
    """
    Map one nested JSON record back into canonical fields.

    JSON structure:

    {
      "job_id": "...",
      "title": "...",
      "company_location": "...",
      "experience_level": "...",
      "employment_type": "...",
      "salary": {
        "min": ...,
        "max": ...,
        "currency": "USD"
      },
      "skills": [...],
      "job_category": "..."
    }
    """
    salary = rec.get("salary", {}) or {}
    skills = rec.get("skills", [])

    canonical = {
        "job_id": rec.get("job_id"),
        "title": rec.get("title"),
        "company_location": rec.get("company_location"),
        "experience_level": rec.get("experience_level"),
        "employment_type": rec.get("employment_type"),
        "salary_min": salary.get("min"),
        "salary_max": salary.get("max"),
        "currency": salary.get("currency"),
        "skills": skills if isinstance(skills, list) else [skills],
        "job_category": rec.get("job_category"),
    }

    return canonical


def json_to_canonical_df(path: Path) -> pd.DataFrame:
    nested_records = load_nested_json(path)
    canonical_records = [nested_to_canonical_record(r) for r in nested_records]
    df = pd.DataFrame(canonical_records)

    # Ensure column order and validation
    df = df[CANONICAL_COLUMN_ORDER]
    validate_canonical_df(df)
    return df


def compare_with_canonical(df_json: pd.DataFrame) -> None:
    """
    Optional helper to quickly compare with the canonical parquet file.
    Prints basic info.
    """
    canonical_path = DATA_CANONICAL / CANONICAL_FILENAME
    if not canonical_path.exists():
        print("Canonical parquet file not found for comparison.")
        return

    df_canonical = pd.read_parquet(canonical_path)

    print("JSON loaded shape:", df_json.shape)
    print("Canonical parquet shape:", df_canonical.shape)

    # Simple check on first few rows for sanity
    print("\nFirst rows from JSON loaded DataFrame:")
    print(df_json.head())

    print("\nFirst rows from canonical parquet DataFrame:")
    print(df_canonical.head())


def main():
    json_path = DATA_FORMATTED / NESTED_JSON_FILENAME
    print(f"Loading nested JSON from: {json_path}")
    df_json = json_to_canonical_df(json_path)

    print("Loaded JSON into canonical DataFrame.")
    compare_with_canonical(df_json)


if __name__ == "__main__":
    main()
