# TOON loader placeholder
# src/formats/toon_loader.py

from pathlib import Path
import pandas as pd

from src.config import DATA_FORMATTED
from src.schemas.canonical_schema import (
    CANONICAL_COLUMN_ORDER,
    validate_canonical_df,
)
from src.schemas.toon_parser import parse_toon_records


TOON_FILENAME = "salaries.toon"


def toon_to_canonical_df(path: Path) -> pd.DataFrame:
    """
    Load TOON file and map it into the canonical DataFrame.

    The TOON records are already written with the same field names
    as the canonical schema, so we mostly need to assemble a DataFrame
    and enforce column order and validation.
    """
    records = parse_toon_records(path)
    if not records:
        raise ValueError(f"No records found in TOON file at {path}")

    df = pd.DataFrame(records)

    # Ensure all canonical columns exist
    missing_cols = [c for c in CANONICAL_COLUMN_ORDER if c not in df.columns]
    if missing_cols:
        raise ValueError(f"TOON records missing expected columns: {missing_cols}")

    # Reorder columns to canonical order
    df = df[CANONICAL_COLUMN_ORDER]
    validate_canonical_df(df)
    return df


def main():
    toon_path = DATA_FORMATTED / TOON_FILENAME
    print(f"Loading TOON file from: {toon_path}")
    df_toon = toon_to_canonical_df(toon_path)

    print("Loaded TOON into canonical DataFrame.")
    print("Shape:", df_toon.shape)
    print("Preview:")
    print(df_toon.head())


if __name__ == "__main__":
    main()
