# schema richness calculator placeholder
# src/metrics/schema_richness.py

from pathlib import Path
import pandas as pd

from src.config import DATA_FOR_TABLEAU


OUTPUT_FILENAME = "schema_metrics.csv"


def build_schema_metrics() -> pd.DataFrame:
    """
    Construct a small table that scores each format on schema richness.

    Fields:
      format
      has_explicit_schema      (0 or 1)
      has_type_annotations     (0 or 1)
      has_constraints          (0 or 1)  # primary keys, required fields, etc
      supports_nested_structs  (0 or 1)
      schema_richness_score    (0 to 1)
    """

    rows = []

    # CSV
    rows.append(
        {
            "format": "CSV",
            "has_explicit_schema": 0,
            "has_type_annotations": 0,
            "has_constraints": 0,
            "supports_nested_structs": 0,
        }
    )

    # JSON
    rows.append(
        {
            "format": "JSON",
            "has_explicit_schema": 0,
            "has_type_annotations": 0,  # types only implicit in values
            "has_constraints": 0,
            "supports_nested_structs": 1,
        }
    )

    # TOON
    rows.append(
        {
            "format": "TOON",
            "has_explicit_schema": 1,
            "has_type_annotations": 1,
            "has_constraints": 1,  # primary_key, required, etc in schema
            "supports_nested_structs": 1,  # you can extend TOON this way
        }
    )

    df = pd.DataFrame(rows)

    # Compute a simple richness score as average of the binary flags
    flag_cols = [
        "has_explicit_schema",
        "has_type_annotations",
        "has_constraints",
        "supports_nested_structs",
    ]
    df["schema_richness_score"] = df[flag_cols].mean(axis=1)

    return df


def main():
    DATA_FOR_TABLEAU.mkdir(parents=True, exist_ok=True)

    df = build_schema_metrics()

    output_path = DATA_FOR_TABLEAU / OUTPUT_FILENAME
    df.to_csv(output_path, index=False)

    print(f"Saved schema metrics to: {output_path}")
    print("\nSchema metrics preview:")
    print(df)


if __name__ == "__main__":
    main()
