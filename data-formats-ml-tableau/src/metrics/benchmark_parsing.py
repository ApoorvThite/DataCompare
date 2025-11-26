# parsing benchmark placeholder
# src/metrics/benchmark_parsing.py

import time
from pathlib import Path
import os

import pandas as pd

from src.config import DATA_FORMATTED, DATA_FOR_TABLEAU, DATA_CANONICAL
from src.formats.csv_loader import flat_csv_to_canonical_df
from src.formats.json_loader import json_to_canonical_df
from src.formats.toon_loader import toon_to_canonical_df


FLAT_CSV_FILENAME = "salaries_flat.csv"
NESTED_JSON_FILENAME = "salaries_nested.json"
TOON_FILENAME = "salaries.toon"
OUTPUT_METRICS_FILENAME = "format_metrics.csv"


def benchmark_format(name: str, path: Path, loader_fn):
    """
    Benchmark a single format:
    - file size
    - parse time into canonical DataFrame
    - number of records and fields
    """
    if not path.exists():
        raise FileNotFoundError(f"File for format {name} not found at {path}")

    size_bytes = os.path.getsize(path)

    t0 = time.perf_counter()
    df = loader_fn(path)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    num_records = len(df)
    num_fields = df.shape[1]

    return {
        "format": name,
        "file_path": str(path),
        "file_size_kb": round(size_bytes / 1024.0, 2),
        "parse_time_ms": round(elapsed_ms, 3),
        "num_records": num_records,
        "num_fields": num_fields,
    }


def main():
    DATA_FOR_TABLEAU.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_FORMATTED / FLAT_CSV_FILENAME
    json_path = DATA_FORMATTED / NESTED_JSON_FILENAME
    toon_path = DATA_FORMATTED / TOON_FILENAME

    print("Benchmarking CSV parsing...")
    csv_metrics = benchmark_format("CSV", csv_path, flat_csv_to_canonical_df)

    print("Benchmarking JSON parsing...")
    json_metrics = benchmark_format("JSON", json_path, json_to_canonical_df)

    print("Benchmarking TOON parsing...")
    toon_metrics = benchmark_format("TOON", toon_path, toon_to_canonical_df)

    metrics = [csv_metrics, json_metrics, toon_metrics]
    df_metrics = pd.DataFrame(metrics)

    output_path = DATA_FOR_TABLEAU / OUTPUT_METRICS_FILENAME
    df_metrics.to_csv(output_path, index=False)

    print(f"\nParsed metrics saved to: {output_path}")
    print("\nMetrics preview:")
    print(df_metrics)


if __name__ == "__main__":
    main()
