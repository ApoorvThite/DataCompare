# src/formats/csv_loader.py

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import DATA_RAW, DATA_CANONICAL, DATA_FORMATTED
from src.schemas.canonical_schema import (
    CANONICAL_COLUMN_ORDER,
    validate_canonical_df,
)


RAW_FILENAME = "global_ai_ml_data_salaries.csv"
CANONICAL_FILENAME = "salaries_canonical.parquet"
FLAT_CSV_FILENAME = "salaries_flat.csv"


def _ensure_paths():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_CANONICAL.mkdir(parents=True, exist_ok=True)


def _load_raw_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found at {path}")
    df = pd.read_csv(path)
    return df


def _generate_job_id(df: pd.DataFrame) -> pd.Series:
    """
    Create a stable job_id like JOB000001 based on row index.
    """
    return [f"JOB{idx:06d}" for idx in range(len(df))]


def _derive_job_category(title: str) -> str:
    """
    Very simple heuristic mapping from job_title to a coarse job_category.
    You can refine this later.
    """
    if not isinstance(title, str):
        return "Other"

    t = title.lower()

    if "scientist" in t:
        return "Data Scientist"
    if "machine learning" in t or "ml engineer" in t:
        return "ML Engineer"
    if "engineer" in t:
        return "Data Engineer"
    if "analyst" in t:
        return "Data Analyst"
    if "manager" in t:
        return "Manager"
    if "architect" in t:
        return "Architect"
    return "Other"


def raw_to_canonical(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map the original salary dataset into the canonical schema.

    Expected raw columns (from the Kaggle style dataset):
    - work_year
    - experience_level
    - employment_type
    - job_title
    - salary
    - salary_currency
    - salary_in_usd
    - employee_residence
    - remote_ratio
    - company_location
    - company_size
    """

    df = pd.DataFrame()

    df["job_id"] = _generate_job_id(df_raw)
    df["title"] = df_raw["job_title"].astype(str)
    df["company_location"] = df_raw["company_location"].astype(str)
    df["experience_level"] = df_raw["experience_level"].astype(str)
    df["employment_type"] = df_raw["employment_type"].astype(str)

    if "salary_in_usd" in df_raw.columns:
        salary_usd = df_raw["salary_in_usd"].astype(float)
    else:
        salary_usd = df_raw["salary"].astype(float)

    df["salary_min"] = salary_usd
    df["salary_max"] = salary_usd
    df["currency"] = "USD"

    df["skills"] = [[] for _ in range(len(df))]
    df["job_category"] = df["title"].apply(_derive_job_category)

    df = df[CANONICAL_COLUMN_ORDER]
    validate_canonical_df(df)

    return df


def save_canonical(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved canonical dataset to {path}")


# ---------- flat CSV helpers for formatted file ----------

def parse_skills_string(value):
    """
    Convert the semicolon separated skills string back into a list.

    Handles:
    - empty string or NaN -> []
    - "Python;SQL;Tableau" -> ["Python", "SQL", "Tableau"]
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(x) for x in value]

    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return []
        except TypeError:
            pass
        return [str(value)]

    s = value.strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(";") if p.strip()]
    return parts


def flat_csv_to_canonical_df(path: Path) -> pd.DataFrame:
    """
    Load the flat CSV (salaries_flat.csv) and ensure it matches the canonical schema.

    Assumes columns are already CANONICAL_COLUMN_ORDER but
    converts skills back to a list.
    """
    if not path.exists():
        raise FileNotFoundError(f"Flat CSV not found at {path}")

    df = pd.read_csv(path)

    # Convert skills string to list
    if "skills" in df.columns:
        df["skills"] = df["skills"].apply(parse_skills_string)
    else:
        df["skills"] = [[] for _ in range(len(df))]

    # Ensure all canonical columns exist
    missing = [c for c in CANONICAL_COLUMN_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Flat CSV DataFrame missing columns: {missing}")

    df = df[CANONICAL_COLUMN_ORDER]
    validate_canonical_df(df)
    return df


def main():
    """
    Entry point used to create the canonical parquet from the raw CSV.
    """
    _ensure_paths()

    raw_path = DATA_RAW / RAW_FILENAME
    canonical_path = DATA_CANONICAL / CANONICAL_FILENAME

    print(f"Loading raw CSV from: {raw_path}")
    df_raw = _load_raw_csv(raw_path)

    print("Transforming raw data to canonical schema...")
    df_canonical = raw_to_canonical(df_raw)

    print("Saving canonical dataset...")
    save_canonical(df_canonical, canonical_path)

    print("Preview of canonical data:")
    print(df_canonical.head())


if __name__ == "__main__":
    main()
