# src/llm/density_metrics.py

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import tiktoken

from src.config import DATA_FORMATTED, DATA_FOR_TABLEAU
from src.schemas.toon_parser import parse_toon_records


FLAT_CSV_FILENAME = "salaries_flat.csv"
NESTED_JSON_FILENAME = "salaries_nested.json"
TOON_FILENAME = "salaries.toon"

OUTPUT_FILENAME = "llm_density_metrics.csv"
ENCODING_NAME = "cl100k_base"


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_encoding():
    try:
        return tiktoken.get_encoding(ENCODING_NAME)
    except KeyError:
        return tiktoken.encoding_for_model("gpt-4")


def tokenize_text(text: str) -> int:
    enc = get_encoding()
    tokens = enc.encode(text)
    return len(tokens)


def estimate_density(
    text: str, format_name: str
) -> Dict[str, Any]:
    """
    Estimate semantic density using a simple heuristic:

    1. Count characters that are mostly structural, not semantic.
    2. Convert that character count to an approximate token count.
    3. Define semantic_density = 1 - non_semantic_token_est / llm_token_count.
    """
    llm_token_count = tokenize_text(text)
    text_length_chars = len(text)

    if text_length_chars == 0 or llm_token_count == 0:
        return {
            "llm_token_count": 0,
            "text_length_chars": 0,
            "non_semantic_char_count": 0,
            "non_semantic_token_est": 0.0,
            "semantic_density": 0.0,
        }

    # Average chars per token
    avg_chars_per_token = text_length_chars / float(llm_token_count)

    fmt = format_name.upper()

    if fmt == "CSV":
        # Structural chars in CSV: commas and quotes
        non_semantic_chars = re.findall(r'[",]', text)
    else:
        # Structural chars in JSON and TOON: braces, brackets, colons, commas, quotes
        non_semantic_chars = re.findall(r'[{}\[\]:",]', text)

    non_semantic_char_count = len(non_semantic_chars)

    non_semantic_token_est = non_semantic_char_count / avg_chars_per_token

    # Clamp semantic density between 0 and 1
    semantic_density = 1.0 - (non_semantic_token_est / llm_token_count)
    semantic_density = max(0.0, min(1.0, semantic_density))

    return {
        "llm_token_count": int(llm_token_count),
        "text_length_chars": int(text_length_chars),
        "non_semantic_char_count": int(non_semantic_char_count),
        "non_semantic_token_est": round(non_semantic_token_est, 2),
        "semantic_density": round(semantic_density, 4),
    }


def csv_density(path: Path) -> Dict[str, Any]:
    text = load_text(path)
    density_stats = estimate_density(text, "CSV")

    return {
        "format": "CSV",
        "llm_encoding": ENCODING_NAME,
        **density_stats,
    }


def json_density(path: Path) -> Dict[str, Any]:
    text = load_text(path)
    # Validate JSON just to be safe
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _ = len(data)  # not used directly, but forces parse

    density_stats = estimate_density(text, "JSON")

    return {
        "format": "JSON",
        "llm_encoding": ENCODING_NAME,
        **density_stats,
    }


def toon_density(path: Path) -> Dict[str, Any]:
    text = load_text(path)
    # Validate TOON parsing
    records = parse_toon_records(path)
    if len(records) == 0:
        raise ValueError("No records parsed from TOON file.")

    density_stats = estimate_density(text, "TOON")

    return {
        "format": "TOON",
        "llm_encoding": ENCODING_NAME,
        **density_stats,
    }


def main():
    DATA_FOR_TABLEAU.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_FORMATTED / FLAT_CSV_FILENAME
    json_path = DATA_FORMATTED / NESTED_JSON_FILENAME
    toon_path = DATA_FORMATTED / TOON_FILENAME

    rows: List[Dict[str, Any]] = [
        csv_density(csv_path),
        json_density(json_path),
        toon_density(toon_path),
    ]

    df = pd.DataFrame(rows)

    out_path = DATA_FOR_TABLEAU / OUTPUT_FILENAME
    df.to_csv(out_path, index=False)

    print(f"Saved LLM density metrics to: {out_path}")
    print("\nPreview:")
    print(df)


if __name__ == "__main__":
    main()
