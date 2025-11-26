# canonical to CSV converter placeholder
# src/formats/canonical_to_csv.py

from pathlib import Path
import pandas as pd

from src.config import DATA_CANONICAL, DATA_FORMATTED
from src.schemas.canonical_schema import CANONICAL_COLUMN_ORDER, validate_canonical_df


CANONICAL_FILENAME = "salaries_canonical.parquet"
FLAT_CSV_FILENAME = "salaries_flat.csv"


def load_canonical() -> pd.DataFrame:
    path = DATA_CANONICAL / CANONICAL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Canonical file not found at {path}")
    df = pd.read_parquet(path)
    validate_canonical_df(df)
    return df


def skills_list_to_string(skills):
    """
    Convert list of skills into a semicolon separated string.
    If skills is already a string or missing, handle gracefully.
    """
    if isinstance(skills, list):
        return ";".join(map(str, skills))
    if pd.isna(skills):
        return ""
    return str(skills)


def canonical_to_flat_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten the canonical schema to a CSV friendly version.
    Only transformation needed for now is skills list -> string.
    """
    df_flat = df.copy()
    df_flat["skills"] = df_flat["skills"].apply(skills_list_to_string)
    # Ensure column order
    df_flat = df_flat[CANONICAL_COLUMN_ORDER]
    return df_flat


def save_flat_csv(df_flat: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df_flat.to_csv(path, index=False)
    print(f"Saved flat CSV to {path}")


def main():
    print("Loading canonical dataset...")
    df_canonical = load_canonical()

    print("Converting canonical dataset to flat CSV representation...")
    df_flat = canonical_to_flat_csv(df_canonical)

    flat_csv_path = DATA_FORMATTED / FLAT_CSV_FILENAME
    save_flat_csv(df_flat, flat_csv_path)

    print("Preview of flat CSV data:")
    print(df_flat.head())


if __name__ == "__main__":
    main()
