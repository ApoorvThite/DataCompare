# canonical to TOON converter placeholder
# src/formats/canonical_to_toon.py

from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.config import DATA_CANONICAL, DATA_FORMATTED, PROJECT_ROOT
from src.schemas.canonical_schema import CANONICAL_COLUMN_ORDER, validate_canonical_df


CANONICAL_FILENAME = "salaries_canonical.parquet"
TOON_SCHEMA_FILENAME = "src/schemas/toon_schema.toon"
TOON_OUTPUT_FILENAME = "salaries.toon"


def load_canonical() -> pd.DataFrame:
    path = DATA_CANONICAL / CANONICAL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Canonical file not found at {path}")
    df = pd.read_parquet(path)
    validate_canonical_df(df)
    return df


def load_toon_schema() -> str:
    schema_path = PROJECT_ROOT / TOON_SCHEMA_FILENAME
    if not schema_path.exists():
        raise FileNotFoundError(f"TOON schema file not found at {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        return f.read().rstrip()  # strip trailing newlines for clean concatenation


def toon_serialize_value(value):
    """
    Convert a Python value into a TOON friendly literal.

    Rules:
    - strings become "escaped"
    - numbers stay as is
    - lists become [ .. ] using JSON style for simplicity
    - None becomes null
    """
    # Handle missing
    try:
        if pd.isna(value):
            return "null"
    except TypeError:
        pass

    # Numpy arrays
    if isinstance(value, np.ndarray):
        value = value.tolist()

    # Lists or tuples
    if isinstance(value, (list, tuple)):
        # Use JSON style list
        return json.dumps(list(value))

    # Numpy numbers
    if isinstance(value, (np.integer, np.floating)):
        return str(value.item())

    # Plain numbers
    if isinstance(value, (int, float)):
        return str(value)

    # Everything else as string
    s = str(value)
    # Escape inner quotes
    s = s.replace('"', '\\"')
    return f'"{s}"'


def record_to_toon_block(rec: dict) -> str:
    """
    Convert a canonical row dict into a TOON record block.

    Example:

    record Job {
      job_id = "JOB000001"
      title = "Data Scientist"
      ...
    }
    """
    lines = ["record Job {"]

    for col in CANONICAL_COLUMN_ORDER:
        val = rec[col]
        literal = toon_serialize_value(val)
        lines.append(f"  {col} = {literal}")

    lines.append("}")
    return "\n".join(lines)


def canonical_to_toon_text(df: pd.DataFrame) -> str:
    """
    Combine schema header and all records into a single TOON text file.
    """
    records = df.to_dict(orient="records")
    record_blocks = [record_to_toon_block(rec) for rec in records]

    schema_text = load_toon_schema()
    parts = [schema_text, ""]  # blank line between schema and records
    parts.extend(record_blocks)
    return "\n\n".join(parts)


def save_toon(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved TOON file to {path}")


def main():
    print("Loading canonical dataset...")
    df_canonical = load_canonical()

    print("Converting canonical dataset to TOON representation...")
    toon_text = canonical_to_toon_text(df_canonical)

    toon_path = DATA_FORMATTED / TOON_OUTPUT_FILENAME
    save_toon(toon_text, toon_path)

    print("Preview of TOON file header and first record:")
    preview_lines = toon_text.splitlines()[:25]
    print("\n".join(preview_lines))


if __name__ == "__main__":
    main()
