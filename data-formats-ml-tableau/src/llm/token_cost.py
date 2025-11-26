# src/llm/token_cost.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

import tiktoken

from src.config import DATA_FORMATTED, DATA_FOR_TABLEAU
from src.schemas.toon_parser import parse_toon_records


FLAT_CSV_FILENAME = "salaries_flat.csv"
NESTED_JSON_FILENAME = "salaries_nested.json"
TOON_FILENAME = "salaries.toon"

OUTPUT_FILENAME = "llm_token_cost.csv"

# Use a generic encoding compatible with many GPT style models
ENCODING_NAME = "cl100k_base"


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_encoding():
    try:
        return tiktoken.get_encoding(ENCODING_NAME)
    except KeyError:
        # Fallback if encoding name changes
        return tiktoken.encoding_for_model("gpt-4")


def tokenize_text(text: str) -> int:
    enc = get_encoding()
    tokens = enc.encode(text)
    return len(tokens)


def csv_token_cost(path: Path) -> Dict[str, Any]:
    text = load_text(path)
    df = pd.read_csv(path)

    num_records = len(df)
    num_fields = len(df.columns)

    total_tokens = tokenize_text(text)

    tokens_per_record = total_tokens / num_records if num_records > 0 else 0.0
    tokens_per_field = (
        total_tokens / (num_records * num_fields) if num_records * num_fields > 0 else 0.0
    )

    return {
        "format": "CSV",
        "llm_encoding": ENCODING_NAME,
        "llm_token_count": int(total_tokens),
        "tokens_per_record": round(tokens_per_record, 2),
        "tokens_per_field": round(tokens_per_field, 4),
    }


def json_token_cost(path: Path) -> Dict[str, Any]:
    text = load_text(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top level JSON array of records.")

    num_records = len(data)
    # Use the union of keys on the first record as field count
    first_rec = data[0] if num_records > 0 else {}
    num_fields = len(first_rec.keys())

    total_tokens = tokenize_text(text)

    tokens_per_record = total_tokens / num_records if num_records > 0 else 0.0
    tokens_per_field = (
        total_tokens / (num_records * num_fields) if num_records * num_fields > 0 else 0.0
    )

    return {
        "format": "JSON",
        "llm_encoding": ENCODING_NAME,
        "llm_token_count": int(total_tokens),
        "tokens_per_record": round(tokens_per_record, 2),
        "tokens_per_field": round(tokens_per_field, 4),
    }


def toon_token_cost(path: Path) -> Dict[str, Any]:
    text = load_text(path)
    records = parse_toon_records(path)

    num_records = len(records)
    if num_records == 0:
        raise ValueError("No records parsed from TOON file.")

    first_rec = records[0]
    num_fields = len(first_rec.keys())

    total_tokens = tokenize_text(text)

    tokens_per_record = total_tokens / num_records if num_records > 0 else 0.0
    tokens_per_field = (
        total_tokens / (num_records * num_fields) if num_records * num_fields > 0 else 0.0
    )

    return {
        "format": "TOON",
        "llm_encoding": ENCODING_NAME,
        "llm_token_count": int(total_tokens),
        "tokens_per_record": round(tokens_per_record, 2),
        "tokens_per_field": round(tokens_per_field, 4),
    }


def main():
    DATA_FOR_TABLEAU.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_FORMATTED / FLAT_CSV_FILENAME
    json_path = DATA_FORMATTED / NESTED_JSON_FILENAME
    toon_path = DATA_FORMATTED / TOON_FILENAME

    rows: List[Dict[str, Any]] = [
        csv_token_cost(csv_path),
        json_token_cost(json_path),
        toon_token_cost(toon_path),
    ]

    df = pd.DataFrame(rows)

    out_path = DATA_FOR_TABLEAU / OUTPUT_FILENAME
    df.to_csv(out_path, index=False)

    print(f"Saved LLM token cost metrics to: {out_path}")
    print("\nPreview:")
    print(df)


if __name__ == "__main__":
    main()
