# evaluate all pipelines placeholder
# src/ml/evaluate_pipeline.py

from pathlib import Path

import pandas as pd

from src.config import DATA_FOR_TABLEAU
from src.ml.pipeline_csv import run_csv_pipeline
from src.ml.pipeline_json import run_json_pipeline
from src.ml.pipeline_toon import run_toon_pipeline


OUTPUT_FILENAME = "ml_metrics.csv"


def main():
    DATA_FOR_TABLEAU.mkdir(parents=True, exist_ok=True)

    print("Running CSV pipeline...")
    csv_metrics = run_csv_pipeline()

    print("Running JSON pipeline...")
    json_metrics = run_json_pipeline()

    print("Running TOON pipeline...")
    toon_metrics = run_toon_pipeline()

    metrics_list = [csv_metrics, json_metrics, toon_metrics]
    df = pd.DataFrame(metrics_list)

    output_path = DATA_FOR_TABLEAU / OUTPUT_FILENAME
    df.to_csv(output_path, index=False)

    print(f"\nSaved ML metrics to: {output_path}")
    print("\nMetrics preview:")
    print(df)


if __name__ == "__main__":
    main()
