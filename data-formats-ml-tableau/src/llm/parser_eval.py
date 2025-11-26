# src/llm/parser_eval.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import tiktoken
from openai import OpenAI

from src.config import DATA_CANONICAL, DATA_FOR_TABLEAU

CANONICAL_FILENAME = "salaries_canonical.csv"
OUTPUT_FILENAME = "llm_parser_metrics.csv"

EMBEDDING_MODEL = "text-embedding-3-small"  # not strictly needed here, but kept for consistency
LLM_MODEL = "gpt-4o-mini"

ENCODING_NAME = "cl100k_base"

MAX_RECORDS = 40  # number of rows to test per format


# ---------------------------
# Tokenization helper
# ---------------------------

def get_encoding():
    try:
        return tiktoken.get_encoding(ENCODING_NAME)
    except KeyError:
        return tiktoken.encoding_for_model("gpt-4")


ENC = get_encoding()


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


# ---------------------------
# Load canonical data
# ---------------------------

def load_canonical() -> pd.DataFrame:
    """
    Load the raw canonical salaries dataset and adapt it into
    a unified schema that matches the rest of the project.

    Target columns:
      job_id, title, company_location, experience_level,
      employment_type, salary_min, salary_max, currency,
      skills, job_category
    """
    path = DATA_CANONICAL / CANONICAL_FILENAME
    df_raw = pd.read_csv(path)

    num_rows = len(df_raw)

    # job_id
    if "job_id" in df_raw.columns:
        job_id = df_raw["job_id"].astype(str)
    else:
        job_id = pd.Series(
            [f"JOB{str(i).zfill(6)}" for i in range(num_rows)],
            index=df_raw.index,
        )

    # title
    if "title" in df_raw.columns:
        title = df_raw["title"].astype(str)
    elif "job_title" in df_raw.columns:
        title = df_raw["job_title"].astype(str)
    else:
        string_cols = df_raw.select_dtypes(include=["object"]).columns
        if len(string_cols) > 0:
            title = df_raw[string_cols[0]].astype(str)
        else:
            title = pd.Series(["Unknown title"] * num_rows, index=df_raw.index)

    # company_location
    if "company_location" in df_raw.columns:
        company_location = df_raw["company_location"].astype(str)
    elif "employee_residence" in df_raw.columns:
        company_location = df_raw["employee_residence"].astype(str)
    else:
        company_location = pd.Series(["Unknown"] * num_rows, index=df_raw.index)

    # experience_level
    if "experience_level" in df_raw.columns:
        experience_level = df_raw["experience_level"].astype(str)
    else:
        experience_level = pd.Series(["Unknown"] * num_rows, index=df_raw.index)

    # employment_type
    if "employment_type" in df_raw.columns:
        employment_type = df_raw["employment_type"].astype(str)
    else:
        employment_type = pd.Series(["Unknown"] * num_rows, index=df_raw.index)

    # salary_min and salary_max
    if "salary_min" in df_raw.columns and "salary_max" in df_raw.columns:
        salary_min = pd.to_numeric(df_raw["salary_min"], errors="coerce")
        salary_max = pd.to_numeric(df_raw["salary_max"], errors="coerce")
    else:
        if "salary_in_usd" in df_raw.columns:
            base_salary = pd.to_numeric(df_raw["salary_in_usd"], errors="coerce")
        elif "salary" in df_raw.columns:
            base_salary = pd.to_numeric(df_raw["salary"], errors="coerce")
        else:
            base_salary = pd.Series([0.0] * num_rows, index=df_raw.index)

        salary_min = base_salary
        salary_max = base_salary

    # currency
    if "currency" in df_raw.columns:
        currency = df_raw["currency"].astype(str)
    elif "salary_currency" in df_raw.columns:
        currency = df_raw["salary_currency"].astype(str)
    else:
        currency = pd.Series(["USD"] * num_rows, index=df_raw.index)

    # skills
    if "skills" in df_raw.columns:
        skills_col = df_raw["skills"]
    else:
        skills_col = pd.Series(["[]"] * num_rows, index=df_raw.index)

    # job_category
    if "job_category" in df_raw.columns:
        job_category = df_raw["job_category"].astype(str)
    elif "job_title" in df_raw.columns:
        job_category = df_raw["job_title"].astype(str)
    else:
        job_category = title.copy()

    df = pd.DataFrame(
        {
            "job_id": job_id,
            "title": title,
            "company_location": company_location,
            "experience_level": experience_level,
            "employment_type": employment_type,
            "salary_min": salary_min,
            "salary_max": salary_max,
            "currency": currency,
            "skills": skills_col,
            "job_category": job_category,
        }
    )

    return df


# ---------------------------
# Record renderers
# ---------------------------

def render_csv_record(row: pd.Series) -> str:
    header = "job_id,title,company_location,experience_level,employment_type,salary_min,salary_max,currency,skills,job_category"
    values = [
        row["job_id"],
        row["title"],
        row["company_location"],
        row["experience_level"],
        row["employment_type"],
        str(row["salary_min"]),
        str(row["salary_max"]),
        row["currency"],
        str(row["skills"]),
        row["job_category"],
    ]
    line = ",".join(values)
    return header + "\n" + line


def render_json_record(row: pd.Series) -> str:
    record = {
        "job_id": row["job_id"],
        "title": row["title"],
        "company_location": row["company_location"],
        "experience_level": row["experience_level"],
        "employment_type": row["employment_type"],
        "salary": {
            "min": float(row["salary_min"]),
            "max": float(row["salary_max"]),
            "currency": row["currency"],
        },
        "skills": row["skills"] if isinstance(row["skills"], list) else [],
        "job_category": row["job_category"],
    }
    return json.dumps(record, ensure_ascii=False, indent=2)


def render_toon_record(row: pd.Series) -> str:
    skills_val = row["skills"] if isinstance(row["skills"], list) else []
    skills_str = ", ".join(map(str, skills_val))

    template = (
        "Job(\n"
        f"  job_id: string = \"{row['job_id']}\",\n"
        f"  title: string = \"{row['title']}\",\n"
        f"  company_location: string = \"{row['company_location']}\",\n"
        f"  experience_level: string = \"{row['experience_level']}\",\n"
        f"  employment_type: string = \"{row['employment_type']}\",\n"
        f"  salary_min: number = {row['salary_min']},\n"
        f"  salary_max: number = {row['salary_max']},\n"
        f"  currency: string = \"{row['currency']}\",\n"
        f"  skills: array[string] = [{skills_str}],\n"
        f"  job_category: string = \"{row['job_category']}\"\n"
        ")"
    )
    return template


# ---------------------------
# LLM helpers
# ---------------------------

TARGET_FIELDS = [
    "job_id",
    "title",
    "company_location",
    "experience_level",
    "employment_type",
    "salary_min",
    "salary_max",
    "currency",
    "job_category",
]


NUMERIC_FIELDS = {"salary_min", "salary_max"}
STRING_FIELDS = set(TARGET_FIELDS) - NUMERIC_FIELDS


def build_ground_truth(row: pd.Series) -> Dict[str, Any]:
    return {
        "job_id": str(row["job_id"]),
        "title": str(row["title"]),
        "company_location": str(row["company_location"]),
        "experience_level": str(row["experience_level"]),
        "employment_type": str(row["employment_type"]),
        "salary_min": float(row["salary_min"]),
        "salary_max": float(row["salary_max"]),
        "currency": str(row["currency"]),
        "job_category": str(row["job_category"]),
    }


def strip_code_fences(text: str) -> str:
    import re

    matches = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    return text.strip()


def call_llm_parse_record(format_name: str, snippet: str, client: OpenAI) -> Tuple[Dict[str, Any] | None, int, int]:
    """
    Ask the LLM to parse a single record in some format into a strict JSON object.
    Returns:
      parsed_dict or None if parse failed,
      prompt_tokens_est,
      response_tokens_est.
    """
    system_msg = (
        "You are a strict data extraction system. "
        "You will be given a single record in a specified format. "
        "Extract it as a JSON object with the following exact keys: "
        "job_id, title, company_location, experience_level, employment_type, "
        "salary_min, salary_max, currency, job_category. "
        "Ensure salary_min and salary_max are numbers, not strings. "
        "Return only the JSON object with no explanations."
    )

    user_msg = f"Format: {format_name}\nRecord:\n{snippet}\n"

    prompt_text = system_msg + "\n\n" + user_msg
    prompt_tokens = count_tokens(prompt_text)

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )

    content = resp.choices[0].message.content or ""
    response_tokens = count_tokens(content)

    cleaned = strip_code_fences(content)
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            return None, prompt_tokens, response_tokens
        return parsed, prompt_tokens, response_tokens
    except Exception:
        return None, prompt_tokens, response_tokens


# ---------------------------
# Metric computation per record and aggregation
# ---------------------------

def compare_record(pred: Dict[str, Any] | None, truth: Dict[str, Any]) -> Dict[str, float]:
    """
    Compare one predicted parsed dict with the ground truth.
    Returns per record metrics:
      field_match_rate,
      numeric_correct_ratio,
      string_correct_ratio,
      parse_failed_flag
    """
    if pred is None:
        return {
            "field_match_rate": 0.0,
            "numeric_correct_ratio": 0.0,
            "string_correct_ratio": 0.0,
            "parse_failed_flag": 1.0,
        }

    correct_fields = 0
    total_fields = len(TARGET_FIELDS)

    numeric_correct = 0
    numeric_total = len(NUMERIC_FIELDS)

    string_correct = 0
    string_total = len(STRING_FIELDS)

    for field in TARGET_FIELDS:
        if field not in pred:
            continue

        pred_val = pred[field]
        true_val = truth[field]

        if field in NUMERIC_FIELDS:
            try:
                pred_num = float(pred_val)
                true_num = float(true_val)
                if abs(pred_num - true_num) <= max(1e-6, 0.0001 * abs(true_num)):
                    numeric_correct += 1
                    correct_fields += 1
            except Exception:
                # numeric mismatch
                pass
        else:
            pred_str = str(pred_val).strip()
            true_str = str(true_val).strip()
            if pred_str == true_str:
                string_correct += 1
                correct_fields += 1

    field_match_rate = correct_fields / total_fields if total_fields > 0 else 0.0
    numeric_correct_ratio = numeric_correct / numeric_total if numeric_total > 0 else 0.0
    string_correct_ratio = string_correct / string_total if string_total > 0 else 0.0

    return {
        "field_match_rate": field_match_rate,
        "numeric_correct_ratio": numeric_correct_ratio,
        "string_correct_ratio": string_correct_ratio,
        "parse_failed_flag": 0.0,
    }


def evaluate_format(format_name: str, df: pd.DataFrame, renderer, client: OpenAI) -> Dict[str, Any]:
    """
    Evaluate LLM-as-parser on a sample of records for a given format.
    """
    df_sample = df.sample(
        n=min(MAX_RECORDS, len(df)),
        random_state=123,
    ).reset_index(drop=True)

    record_metrics: List[Dict[str, float]] = []
    prompt_tokens_list: List[int] = []
    response_tokens_list: List[int] = []

    for i in range(len(df_sample)):
        row = df_sample.loc[i]
        truth = build_ground_truth(row)
        snippet = renderer(row)

        pred, p_tokens, r_tokens = call_llm_parse_record(format_name, snippet, client)

        prompt_tokens_list.append(p_tokens)
        response_tokens_list.append(r_tokens)

        rec_metrics = compare_record(pred, truth)
        record_metrics.append(rec_metrics)

    # Aggregate
    field_match_rate = float(np.mean([m["field_match_rate"] for m in record_metrics]))
    numeric_accuracy = float(np.mean([m["numeric_correct_ratio"] for m in record_metrics]))
    string_accuracy = float(np.mean([m["string_correct_ratio"] for m in record_metrics]))
    parse_failure_rate = float(np.mean([m["parse_failed_flag"] for m in record_metrics]))

    avg_prompt_tokens = float(np.mean(prompt_tokens_list)) if prompt_tokens_list else 0.0
    avg_response_tokens = float(np.mean(response_tokens_list)) if response_tokens_list else 0.0

    return {
        "format": format_name,
        "llm_model": LLM_MODEL,
        "num_records_evaluated": int(len(df_sample)),
        "field_match_rate": round(field_match_rate, 4),
        "numeric_field_accuracy": round(numeric_accuracy, 4),
        "string_field_accuracy": round(string_accuracy, 4),
        "parse_failure_rate": round(parse_failure_rate, 4),
        "avg_prompt_tokens_est": round(avg_prompt_tokens, 1),
        "avg_response_tokens_est": round(avg_response_tokens, 1),
    }


# ---------------------------
# Main
# ---------------------------

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

    client = OpenAI()

    DATA_FOR_TABLEAU.mkdir(parents=True, exist_ok=True)

    df = load_canonical()

    rows: List[Dict[str, Any]] = []

    # CSV
    rows.append(
        evaluate_format("CSV", df, render_csv_record, client)
    )

    # JSON
    rows.append(
        evaluate_format("JSON", df, render_json_record, client)
    )

    # TOON
    rows.append(
        evaluate_format("TOON", df, render_toon_record, client)
    )

    out_df = pd.DataFrame(rows)
    out_path = DATA_FOR_TABLEAU / OUTPUT_FILENAME
    out_df.to_csv(out_path, index=False)

    print(f"Saved LLM parser metrics to: {out_path}")
    print("\nPreview:")
    print(out_df)


if __name__ == "__main__":
    main()
