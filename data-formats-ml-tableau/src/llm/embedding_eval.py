# src/llm/embedding_eval.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from openai import OpenAI

from src.config import DATA_CANONICAL, DATA_FOR_TABLEAU

# Canonical dataset file
CANONICAL_FILENAME = "salaries_canonical.csv"

OUTPUT_FILENAME = "llm_embedding_metrics.csv"

EMBEDDING_MODEL = "text-embedding-3-small"

# Limit number of records per format for API cost
MAX_RECORDS = 500


def load_canonical() -> pd.DataFrame:
    """
    Load the raw canonical salaries dataset and adapt it into
    the unified schema expected by the embedding code.

    Target columns we want to end up with:
      job_id, title, company_location, experience_level,
      employment_type, salary_min, salary_max, currency,
      skills, job_category
    """
    path = DATA_CANONICAL / CANONICAL_FILENAME
    df_raw = pd.read_csv(path)

    # Helper: safe get or default
    def col_or_default(name: str, default_value):
        return df_raw[name] if name in df_raw.columns else default_value

    num_rows = len(df_raw)

    # job_id: synthesize if not present
    if "job_id" in df_raw.columns:
        job_id = df_raw["job_id"].astype(str)
    else:
        job_id = pd.Series(
            [f"JOB{str(i).zfill(6)}" for i in range(num_rows)],
            index=df_raw.index,
        )

    # title: prefer 'title', then 'job_title', else first string column
    if "title" in df_raw.columns:
        title = df_raw["title"].astype(str)
    elif "job_title" in df_raw.columns:
        title = df_raw["job_title"].astype(str)
    else:
        # fallback to the first object dtype column or a constant
        string_cols = df_raw.select_dtypes(include=["object"]).columns
        if len(string_cols) > 0:
            title = df_raw[string_cols[0]].astype(str)
        else:
            title = pd.Series(["Unknown title"] * num_rows, index=df_raw.index)

    # company_location: prefer 'company_location', else 'employee_residence', else default
    if "company_location" in df_raw.columns:
        company_location = df_raw["company_location"].astype(str)
    elif "employee_residence" in df_raw.columns:
        company_location = df_raw["employee_residence"].astype(str)
    else:
        company_location = pd.Series(
            ["Unknown"] * num_rows, index=df_raw.index
        )

    # experience_level
    if "experience_level" in df_raw.columns:
        experience_level = df_raw["experience_level"].astype(str)
    else:
        experience_level = pd.Series(
            ["Unknown"] * num_rows, index=df_raw.index
        )

    # employment_type
    if "employment_type" in df_raw.columns:
        employment_type = df_raw["employment_type"].astype(str)
    else:
        employment_type = pd.Series(
            ["Unknown"] * num_rows, index=df_raw.index
        )

    # salary_min and salary_max: try 'salary_in_usd' or 'salary'
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

    # skills: if no skills column, use empty list per row
    if "skills" in df_raw.columns:
        skills_col = df_raw["skills"]
    else:
        # use string "[]" for now, the renderer will turn non list into []
        skills_col = pd.Series(["[]"] * num_rows, index=df_raw.index)

    # job_category: prefer 'job_category', else use 'job_title', else title
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


# ----------------------------------------
# Text renderers for each format style
# ----------------------------------------


def render_csv_style(row: pd.Series) -> str:
    """
    Render a record in a CSV like one line.
    This is not the full file, just a row style.
    """
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
    header = "job_id,title,company_location,experience_level,employment_type,salary_min,salary_max,currency,skills,job_category"
    line = ",".join(values)
    return header + "\n" + line


def render_json_style(row: pd.Series) -> str:
    """
    Render a record as a JSON object string.
    """
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
    import json

    return json.dumps(record, ensure_ascii=False)


def render_toon_style(row: pd.Series) -> str:
    """
    Render a record in a TOON inspired style.
    We keep it as a single logical "Job" entity with typed looking fields.
    """
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


# ----------------------------------------
# Embedding helpers
# ----------------------------------------


def get_embeddings_for_texts(
    texts: List[str], client: OpenAI
) -> np.ndarray:
    """
    Call OpenAI embeddings API in batches and return an array of shape (N, D).
    """
    embeddings: List[List[float]] = []

    # Simple batching to avoid very large requests
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        for item in resp.data:
            embeddings.append(item.embedding)

    return np.array(embeddings, dtype=float)


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix for rows of X.
    """
    # Normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms
    return X_norm @ X_norm.T


def compute_embedding_metrics(
    embeddings: np.ndarray, labels: np.ndarray
) -> Dict[str, Any]:
    """
    Compute cluster quality metrics from embeddings and class labels.
    """
    num_records = embeddings.shape[0]
    unique_labels = np.unique(labels)
    num_classes = unique_labels.shape[0]

    # Silhouette score with cosine distance
    # Handle edge case where silhouette cannot be computed
    if num_classes < 2 or num_records <= num_classes:
        sil = 0.0
    else:
        try:
            sil = silhouette_score(embeddings, labels, metric="cosine")
        except Exception:
            sil = 0.0

    # Compute intra class and inter class cosine similarity
    sim_matrix = cosine_similarity_matrix(embeddings)

    intra_sims: List[float] = []
    inter_sims: List[float] = []

    for i in range(num_records):
        for j in range(i + 1, num_records):
            if labels[i] == labels[j]:
                intra_sims.append(sim_matrix[i, j])
            else:
                inter_sims.append(sim_matrix[i, j])

    intra_mean = float(np.mean(intra_sims)) if intra_sims else 0.0
    inter_mean = float(np.mean(inter_sims)) if inter_sims else 0.0

    return {
        "num_records_used": int(num_records),
        "num_classes": int(num_classes),
        "silhouette_score": round(float(sil), 4),
        "intra_class_cosine_mean": round(intra_mean, 4),
        "inter_class_cosine_mean": round(inter_mean, 4),
    }


# ----------------------------------------
# Format specific pipelines
# ----------------------------------------


def format_embedding_metrics(
    format_name: str,
    df: pd.DataFrame,
    renderer,
    client: OpenAI,
) -> Dict[str, Any]:
    """
    Sample records, render text using the given renderer, embed, and compute metrics.
    """
    df_sample = df.sample(
        n=min(MAX_RECORDS, len(df)),
        random_state=42,
    ).reset_index(drop=True)

    texts = [renderer(df_sample.loc[i]) for i in range(len(df_sample))]
    labels = df_sample["job_category"].astype(str).values

    embeddings = get_embeddings_for_texts(texts, client)
    metrics = compute_embedding_metrics(embeddings, labels)

    result: Dict[str, Any] = {
        "format": format_name,
        "embedding_model": EMBEDDING_MODEL,
    }
    result.update(metrics)
    return result


def main():
    # Check API key
    client = OpenAI()

    DATA_FOR_TABLEAU.mkdir(parents=True, exist_ok=True)

    df = load_canonical()

    rows: List[Dict[str, Any]] = []

    # CSV style
    rows.append(
        format_embedding_metrics("CSV", df, render_csv_style, client)
    )

    # JSON style
    rows.append(
        format_embedding_metrics("JSON", df, render_json_style, client)
    )

    # TOON style
    rows.append(
        format_embedding_metrics("TOON", df, render_toon_style, client)
    )

    out_df = pd.DataFrame(rows)
    out_path = DATA_FOR_TABLEAU / OUTPUT_FILENAME
    out_df.to_csv(out_path, index=False)

    print(f"Saved LLM embedding metrics to: {out_path}")
    print("\nPreview:")
    print(out_df)


if __name__ == "__main__":
    main()
