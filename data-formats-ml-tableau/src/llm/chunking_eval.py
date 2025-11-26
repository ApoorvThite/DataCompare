# src/llm/chunking_eval.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import tiktoken

from src.config import DATA_FORMATTED, DATA_FOR_TABLEAU
from src.schemas.toon_parser import parse_toon_records


FLAT_CSV_FILENAME = "salaries_flat.csv"
NESTED_JSON_FILENAME = "salaries_nested.json"
TOON_FILENAME = "salaries.toon"

OUTPUT_FILENAME = "llm_chunking_metrics.csv"
ENCODING_NAME = "cl100k_base"

# RAG style token budget per chunk
TOKEN_BUDGET = 512


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_encoding():
    try:
        return tiktoken.get_encoding(ENCODING_NAME)
    except KeyError:
        return tiktoken.encoding_for_model("gpt-4")


ENC = get_encoding()


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def make_line_chunks(text: str, token_budget: int) -> List[int]:
    """
    Simple line based chunking.

    - Split the text into lines.
    - Accumulate lines until adding another line would exceed the token budget.
    - Start a new chunk whenever the budget would be exceeded.
    - Return a list of token counts for each chunk.
    """
    lines = text.splitlines()
    chunks_token_counts: List[int] = []

    if not lines:
        return chunks_token_counts

    current_lines: List[str] = []
    current_tokens = 0

    for line in lines:
        if not current_lines:
            # starting a new chunk
            current_lines = [line]
            current_tokens = count_tokens(line)
            # if a single line exceeds budget, we still keep it as one chunk
            if current_tokens > token_budget:
                chunks_token_counts.append(current_tokens)
                current_lines = []
                current_tokens = 0
            continue

        candidate_text = "\n".join(current_lines + [line])
        candidate_tokens = count_tokens(candidate_text)

        if candidate_tokens <= token_budget:
            current_lines.append(line)
            current_tokens = candidate_tokens
        else:
            # finalize current chunk
            chunks_token_counts.append(current_tokens)
            # start new chunk with this line
            current_lines = [line]
            current_tokens = count_tokens(line)
            if current_tokens > token_budget:
                chunks_token_counts.append(current_tokens)
                current_lines = []
                current_tokens = 0

    if current_lines and current_tokens > 0:
        chunks_token_counts.append(current_tokens)

    return chunks_token_counts


def summarize_chunks(format_name: str, text: str, token_budget: int) -> Dict[str, Any]:
    """
    Produce chunking metrics for a given text and token budget.
    """
    token_counts = make_line_chunks(text, token_budget)

    if not token_counts:
        return {
            "format": format_name,
            "llm_encoding": ENCODING_NAME,
            "token_budget": token_budget,
            "num_chunks": 0,
            "avg_chunk_tokens": 0.0,
            "chunk_token_std": 0.0,
            "min_chunk_tokens": 0,
            "max_chunk_tokens": 0,
            "underutilized_chunk_fraction": 0.0,
        }

    token_array = np.array(token_counts, dtype=float)

    num_chunks = len(token_counts)
    avg_chunk_tokens = float(token_array.mean())
    chunk_token_std = float(token_array.std(ddof=0))
    min_chunk_tokens = int(token_array.min())
    max_chunk_tokens = int(token_array.max())

    # Underutilized chunks: those that use less than 50 percent of the budget
    underutilized_mask = token_array < (0.5 * token_budget)
    underutilized_fraction = float(underutilized_mask.mean())

    metrics: Dict[str, Any] = {
        "format": format_name,
        "llm_encoding": ENCODING_NAME,
        "token_budget": int(token_budget),
        "num_chunks": int(num_chunks),
        "avg_chunk_tokens": round(avg_chunk_tokens, 2),
        "chunk_token_std": round(chunk_token_std, 2),
        "min_chunk_tokens": min_chunk_tokens,
        "max_chunk_tokens": max_chunk_tokens,
        "underutilized_chunk_fraction": round(underutilized_fraction, 4),
    }
    return metrics


def csv_chunk_metrics(path: Path) -> Dict[str, Any]:
    text = load_text(path)
    return summarize_chunks("CSV", text, TOKEN_BUDGET)


def json_chunk_metrics(path: Path) -> Dict[str, Any]:
    text = load_text(path)
    return summarize_chunks("JSON", text, TOKEN_BUDGET)


def toon_chunk_metrics(path: Path) -> Dict[str, Any]:
    # Optionally validate TOON records can be parsed
    _ = parse_toon_records(path)
    text = load_text(path)
    return summarize_chunks("TOON", text, TOKEN_BUDGET)


def main():
    DATA_FOR_TABLEAU.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_FORMATTED / FLAT_CSV_FILENAME
    json_path = DATA_FORMATTED / NESTED_JSON_FILENAME
    toon_path = DATA_FORMATTED / TOON_FILENAME

    rows: List[Dict[str, Any]] = [
        csv_chunk_metrics(csv_path),
        json_chunk_metrics(json_path),
        toon_chunk_metrics(toon_path),
    ]

    df = pd.DataFrame(rows)

    out_path = DATA_FOR_TABLEAU / OUTPUT_FILENAME
    df.to_csv(out_path, index=False)

    print(f"Saved LLM chunking metrics to: {out_path}")
    print("\nPreview:")
    print(df)


if __name__ == "__main__":
    main()
