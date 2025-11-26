# src/metrics/format_structure_metrics.py

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Dict, Any, Set, List

import pandas as pd

from src.config import DATA_FORMATTED, DATA_FOR_TABLEAU
from src.schemas.toon_parser import parse_toon_records


FLAT_CSV_FILENAME = "salaries_flat.csv"
NESTED_JSON_FILENAME = "salaries_nested.json"
TOON_FILENAME = "salaries.toon"

OUTPUT_FILENAME = "format_structure_metrics.csv"


# -----------------------------
# Helpers: general file stats
# -----------------------------


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def compute_file_sizes(path: Path, num_records: int) -> Dict[str, float]:
    raw_bytes = path.stat().st_size

    with open(path, "rb") as f_in:
        compressed = gzip.compress(f_in.read())
    gzip_bytes = len(compressed)

    compression_ratio = gzip_bytes / raw_bytes if raw_bytes > 0 else 1.0
    avg_size_per_record = raw_bytes / num_records if num_records > 0 else 0.0

    return {
        "file_size_bytes": int(raw_bytes),
        "file_size_gzip_bytes": int(gzip_bytes),
        "compression_ratio": round(compression_ratio, 4),
        "avg_size_per_record_bytes": round(avg_size_per_record, 2),
    }


def count_tokens(text: str) -> int:
    """
    Very simple token count: count non space chunks.
    This will be much higher for JSON and TOON than CSV.
    """
    tokens = re.findall(r"\S+", text)
    return len(tokens)


# -----------------------------
# Helpers: JSON structure
# -----------------------------


def json_max_depth(obj: Any, current: int = 0) -> int:
    """
    Compute max nested level for JSON object.
    - Top level list/dict starts at depth 1.
    """
    if isinstance(obj, (dict, list)):
        if not obj:
            return current + 1
        if isinstance(obj, dict):
            return max(json_max_depth(v, current + 1) for v in obj.values())
        else:
            return max(json_max_depth(v, current + 1) for v in obj)
    else:
        return current


def json_key_paths(obj: Any, prefix: str = "") -> Set[str]:
    """
    Collect dotted key paths in a JSON object.
    Example: salary.min, salary.max, skills[0]
    """
    paths: Set[str] = set()

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            paths.add(new_prefix)
            paths.update(json_key_paths(v, new_prefix))
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            new_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            paths.add(new_prefix)
            paths.update(json_key_paths(v, new_prefix))

    return paths


# -----------------------------
# Feature engineering difficulty
# (heuristic, format-specific)
# -----------------------------


def feature_engineering_flags(format_name: str) -> Dict[str, int]:
    """
    Heuristic flags describing how hard feature engineering tends to be
    for each format.

    1 = yes / needed, 0 = no.

    - needs_flattening:
        JSON: 1 (nested objects)
        CSV: 0
        TOON: 0 (we design it flat)

    - needs_type_casts:
        CSV: 1 (all values arrive as strings)
        JSON: 0 (numbers already typed)
        TOON: 0 (types defined in schema)

    - needs_schema_lookup:
        TOON: 1 (you read the schema to know which fields to expect)
        CSV/JSON: 0 in this simple project.

    feature_engineering_difficulty_score =
        needs_flattening + needs_type_casts + needs_schema_lookup
    """
    format_name = format_name.upper()

    if format_name == "CSV":
        flags = {
            "needs_flattening": 0,
            "needs_type_casts": 1,
            "needs_schema_lookup": 0,
        }
    elif format_name == "JSON":
        flags = {
            "needs_flattening": 1,
            "needs_type_casts": 0,
            "needs_schema_lookup": 0,
        }
    elif format_name == "TOON":
        flags = {
            "needs_flattening": 0,
            "needs_type_casts": 0,
            "needs_schema_lookup": 1,
        }
    else:
        flags = {
            "needs_flattening": 0,
            "needs_type_casts": 0,
            "needs_schema_lookup": 0,
        }

    flags["feature_engineering_difficulty_score"] = (
        flags["needs_flattening"]
        + flags["needs_type_casts"]
        + flags["needs_schema_lookup"]
    )

    return flags


# -----------------------------
# Format specific metrics
# -----------------------------


def metrics_for_csv(path: Path) -> Dict[str, Any]:
    text = load_text(path)
    df = pd.read_csv(path)

    num_records = len(df)
    num_tokens = count_tokens(text)
    num_nested_levels = 0
    num_key_paths = len(df.columns)

    # Type inference: in CSV, everything starts as text, parser must
    # decide numeric vs string per cell.
    num_type_inference_ops = int(num_records * len(df.columns))

    file_sizes = compute_file_sizes(path, num_records)
    fe_flags = feature_engineering_flags("CSV")

    metrics: Dict[str, Any] = {
        "format": "CSV",
        "num_tokens": num_tokens,
        "num_nested_levels": num_nested_levels,
        "num_key_paths": num_key_paths,
        "num_type_inference_ops": num_type_inference_ops,
    }
    metrics.update(file_sizes)
    metrics.update(fe_flags)
    return metrics


def metrics_for_json(path: Path) -> Dict[str, Any]:
    text = load_text(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array of records.")

    num_records = len(data)
    num_tokens = count_tokens(text)
    max_depth = json_max_depth(data)
    key_paths = set()
    for rec in data:
        key_paths.update(json_key_paths(rec))

    num_key_paths = len(key_paths)

    # For JSON, values already carry types (numbers, strings, lists).
    # We treat type inference as zero explicit operations.
    num_type_inference_ops = 0

    file_sizes = compute_file_sizes(path, num_records)
    fe_flags = feature_engineering_flags("JSON")

    metrics: Dict[str, Any] = {
        "format": "JSON",
        "num_tokens": num_tokens,
        "num_nested_levels": max_depth,
        "num_key_paths": num_key_paths,
        "num_type_inference_ops": num_type_inference_ops,
    }
    metrics.update(file_sizes)
    metrics.update(fe_flags)
    return metrics


def metrics_for_toon(path: Path) -> Dict[str, Any]:
    text = load_text(path)
    records = parse_toon_records(path)
    num_records = len(records)

    if num_records == 0:
        raise ValueError("No records parsed from TOON file.")

    num_tokens = count_tokens(text)

    # We designed TOON as flat records with a schema section.
    # Treat data depth as 1.
    num_nested_levels = 1

    # Key paths for TOON are just the field names on the Job record.
    first_rec = records[0]
    num_key_paths = len(first_rec.keys())

    # TOON has explicit types in the schema, so the parser does not
    # need to guess types from raw strings.
    num_type_inference_ops = 0

    file_sizes = compute_file_sizes(path, num_records)
    fe_flags = feature_engineering_flags("TOON")

    metrics: Dict[str, Any] = {
        "format": "TOON",
        "num_tokens": num_tokens,
        "num_nested_levels": num_nested_levels,
        "num_key_paths": num_key_paths,
        "num_type_inference_ops": num_type_inference_ops,
    }
    metrics.update(file_sizes)
    metrics.update(fe_flags)
    return metrics


# -----------------------------
# Main
# -----------------------------


def main():
    DATA_FOR_TABLEAU.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_FORMATTED / FLAT_CSV_FILENAME
    json_path = DATA_FORMATTED / NESTED_JSON_FILENAME
    toon_path = DATA_FORMATTED / TOON_FILENAME

    csv_metrics = metrics_for_csv(csv_path)
    json_metrics = metrics_for_json(json_path)
    toon_metrics = metrics_for_toon(toon_path)

    rows: List[Dict[str, Any]] = [csv_metrics, json_metrics, toon_metrics]
    df = pd.DataFrame(rows)

    out_path = DATA_FOR_TABLEAU / OUTPUT_FILENAME
    df.to_csv(out_path, index=False)

    print(f"Saved format structure metrics to: {out_path}")
    print("\nPreview:")
    print(df)


if __name__ == "__main__":
    main()
