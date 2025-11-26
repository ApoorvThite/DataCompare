# src/llm/schema_inference_eval.py

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import tiktoken
from openai import OpenAI

from src.config import DATA_FORMATTED, DATA_FOR_TABLEAU
from src.schemas.toon_parser import parse_toon_records


# Formatted file names
FLAT_CSV_FILENAME = "salaries_flat.csv"
NESTED_JSON_FILENAME = "salaries_nested.json"
TOON_FILENAME = "salaries.toon"

OUTPUT_FILENAME = "llm_schema_eval.csv"
ENCODING_NAME = "cl100k_base"

# Model to use for schema inference
LLM_MODEL = "gpt-4o-mini"  # change if you want


# Ground truth schema for your dataset
# You can adjust types if needed
TRUE_SCHEMA: Dict[str, str] = {
    "job_id": "string",
    "title": "string",
    "company_location": "string",
    "experience_level": "string",
    "employment_type": "string",
    "salary_min": "number",
    "salary_max": "number",
    "currency": "string",
    "skills": "array",
    "job_category": "string",
}


def get_encoding():
    try:
        return tiktoken.get_encoding(ENCODING_NAME)
    except KeyError:
        return tiktoken.encoding_for_model("gpt-4")


ENC = get_encoding()


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def clean_field_name(name: str) -> str:
    """Normalize field names for comparison."""
    name = name.strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)
    return name


def normalize_type(t: str) -> str:
    """Map model type strings into coarse buckets."""
    t = t.strip().lower()
    if any(k in t for k in ["int", "float", "double", "decimal", "num"]):
        return "number"
    if "bool" in t:
        return "boolean"
    if any(k in t for k in ["array", "list", "sequence"]):
        return "array"
    if any(k in t for k in ["object", "dict", "map", "struct"]):
        return "object"
    # default
    return "string"


def strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` fences if the model uses them."""
    # typical pattern: ```json\n...\n```
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[0].strip()
    return text.strip()


def call_llm_for_schema(format_name: str, snippet: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Call the LLM with a snippet and ask for schema.
    Returns:
      - list of {name, type} dicts (may be empty if error)
      - prompt_tokens_est
      - response_tokens_est
    """
    client = OpenAI()

    instruction = (
        "You are a data schema expert. "
        "You will receive a snippet of a dataset. "
        "Your task is to infer the logical fields and their data types. "
        "Return only a JSON array where each element has keys 'name' and 'type'. "
        "Types should be one of: string, number, boolean, array, object. "
        "Do not include any extra keys or explanations."
    )

    user_prompt = f"Format: {format_name}\nHere is the dataset snippet:\n\n{snippet}\n"

    prompt_text = instruction + "\n\n" + user_prompt
    prompt_tokens_est = count_tokens(prompt_text)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content or ""
    response_tokens_est = count_tokens(content)

    cleaned = strip_code_fences(content)

    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON array.")
        return parsed, prompt_tokens_est, response_tokens_est
    except Exception:
        # If parsing fails, return empty result
        return [], prompt_tokens_est, response_tokens_est


def compute_schema_metrics(predicted: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare LLM predicted schema with TRUE_SCHEMA.
    Returns precision, recall, type accuracy, hallucinations.
    """
    true_fields = {clean_field_name(k): normalize_type(v) for k, v in TRUE_SCHEMA.items()}

    pred_field_names_raw: List[str] = []
    pred_types_raw: Dict[str, str] = {}

    for entry in predicted:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        t = entry.get("type")
        if not name or not isinstance(name, str):
            continue
        norm_name = clean_field_name(name)
        pred_field_names_raw.append(norm_name)
        if isinstance(t, str):
            pred_types_raw[norm_name] = normalize_type(t)

    pred_field_set = set(pred_field_names_raw)

    # True positives: fields that exist in ground truth
    hits = pred_field_set.intersection(true_fields.keys())

    recall = len(hits) / len(true_fields) if true_fields else 0.0
    precision = len(hits) / len(pred_field_set) if pred_field_set else 0.0

    # Type accuracy: among hits, how many have matching type
    correct_types = 0
    for name in hits:
        true_t = true_fields[name]
        pred_t = pred_types_raw.get(name)
        if pred_t is not None and pred_t == true_t:
            correct_types += 1

    type_accuracy = correct_types / len(hits) if hits else 0.0

    hallucinated_fields = len(pred_field_set.difference(true_fields.keys()))

    metrics = {
        "num_true_fields": len(true_fields),
        "num_predicted_fields": len(pred_field_set),
        "field_recall": round(recall, 4),
        "field_precision": round(precision, 4),
        "type_accuracy": round(type_accuracy, 4),
        "hallucinated_fields": int(hallucinated_fields),
    }
    return metrics


def build_snippet_csv(path: Path, max_lines: int = 20) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    return "\n".join(lines[:max_lines])


def build_snippet_json(path: Path, max_records: int = 5) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top level JSON array.")
    snippet_records = data[:max_records]
    return json.dumps(snippet_records, indent=2)


def build_snippet_toon(path: Path, max_lines: int = 200) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    return "\n".join(lines[:max_lines])


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

    DATA_FOR_TABLEAU.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_FORMATTED / FLAT_CSV_FILENAME
    json_path = DATA_FORMATTED / NESTED_JSON_FILENAME
    toon_path = DATA_FORMATTED / TOON_FILENAME

    rows: List[Dict[str, Any]] = []

    # CSV
    csv_snippet = build_snippet_csv(csv_path)
    csv_pred, csv_prompt_tokens, csv_resp_tokens = call_llm_for_schema("CSV", csv_snippet)
    csv_metrics = compute_schema_metrics(csv_pred)
    rows.append(
        {
            "format": "CSV",
            "llm_model": LLM_MODEL,
            "prompt_tokens_est": csv_prompt_tokens,
            "response_tokens_est": csv_resp_tokens,
            "parse_error_flag": int(len(csv_pred) == 0),
            **csv_metrics,
        }
    )

    # JSON
    json_snippet = build_snippet_json(json_path)
    json_pred, json_prompt_tokens, json_resp_tokens = call_llm_for_schema("JSON", json_snippet)
    json_metrics = compute_schema_metrics(json_pred)
    rows.append(
        {
            "format": "JSON",
            "llm_model": LLM_MODEL,
            "prompt_tokens_est": json_prompt_tokens,
            "response_tokens_est": json_resp_tokens,
            "parse_error_flag": int(len(json_pred) == 0),
            **json_metrics,
        }
    )

    # TOON
    toon_snippet = build_snippet_toon(toon_path)
    toon_pred, toon_prompt_tokens, toon_resp_tokens = call_llm_for_schema("TOON", toon_snippet)
    toon_metrics = compute_schema_metrics(toon_pred)
    rows.append(
        {
            "format": "TOON",
            "llm_model": LLM_MODEL,
            "prompt_tokens_est": toon_prompt_tokens,
            "response_tokens_est": toon_resp_tokens,
            "parse_error_flag": int(len(toon_pred) == 0),
            **toon_metrics,
        }
    )

    df = pd.DataFrame(rows)
    out_path = DATA_FOR_TABLEAU / OUTPUT_FILENAME
    df.to_csv(out_path, index=False)

    print(f"Saved LLM schema inference metrics to: {out_path}")
    print("\nPreview:")
    print(df)


if __name__ == "__main__":
    main()
