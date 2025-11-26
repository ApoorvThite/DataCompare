# src/ml/pipeline_csv.py

import time
from typing import Dict

import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import DATA_FORMATTED, RANDOM_STATE
from src.formats.csv_loader import flat_csv_to_canonical_df
from src.ml.preprocessing_common import (
    prepare_features_and_target,
    split_train_test,
    build_model_pipeline,
)
from src.ml.features_from_canonical import NUMERIC_FEATURES, CATEGORICAL_FEATURES


FLAT_CSV_FILENAME = "salaries_flat.csv"


def compute_encoded_stats(model, X_train):
    """
    Compute encoded feature space size and sparsity after fitting.

    Returns:
      encoded_feature_dim: int
      encoded_density: float in [0, 1]
      encoded_sparsity: float in [0, 1]
    """
    preprocess = model.named_steps["preprocess"]
    X_enc = preprocess.transform(X_train)

    n_samples, n_features = X_enc.shape

    if sparse.issparse(X_enc):
        non_zero = X_enc.nnz
    else:
        non_zero = np.count_nonzero(X_enc)

    total = n_samples * n_features if n_samples * n_features > 0 else 1
    density = non_zero / total
    sparsity = 1.0 - density

    return n_features, density, sparsity


def run_csv_pipeline() -> Dict[str, float]:
    """
    Train and evaluate a model using the flat CSV representation.
    Returns a metrics dict including timing breakdown, cross validation stats,
    and encoded feature space stats.
    """
    path = DATA_FORMATTED / FLAT_CSV_FILENAME
    df = flat_csv_to_canonical_df(path)

    # Features and target
    X, y = prepare_features_and_target(df)

    # Cross validation on full dataset
    model_for_cv = build_model_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_acc_scores = cross_val_score(model_for_cv, X, y, cv=cv, scoring="accuracy")
    cv_f1_scores = cross_val_score(model_for_cv, X, y, cv=cv, scoring="f1_macro")

    cv_accuracy_mean = cv_acc_scores.mean()
    cv_accuracy_std = cv_acc_scores.std()
    cv_f1_macro_mean = cv_f1_scores.mean()
    cv_f1_macro_std = cv_f1_scores.std()

    # Prep phase timing - feature selection, split, pipeline build
    t_prep0 = time.perf_counter()
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    model = build_model_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    t_prep1 = time.perf_counter()
    prep_time_s = t_prep1 - t_prep0

    # Train timing
    t_train0 = time.perf_counter()
    model.fit(X_train, y_train)
    t_train1 = time.perf_counter()
    train_time_s = t_train1 - t_train0

    # Encoded feature space stats (after fit)
    encoded_dim, encoded_density, encoded_sparsity = compute_encoded_stats(model, X_train)

    # Predict timing
    t_pred0 = time.perf_counter()
    y_pred = model.predict(X_test)
    t_pred1 = time.perf_counter()
    predict_time_s = t_pred1 - t_pred0

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    feature_count = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    pipeline_steps_count = len(model.steps)

    metrics = {
        "format": "CSV",
        "model_type": "LogisticRegression",
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "prep_time_s": round(prep_time_s, 4),
        "train_time_s": round(train_time_s, 4),
        "predict_time_s": round(predict_time_s, 4),
        "feature_count": feature_count,
        "pipeline_steps_count": pipeline_steps_count,
        "cv_accuracy_mean": round(cv_accuracy_mean, 4),
        "cv_accuracy_std": round(cv_accuracy_std, 4),
        "cv_f1_macro_mean": round(cv_f1_macro_mean, 4),
        "cv_f1_macro_std": round(cv_f1_macro_std, 4),
        "encoded_feature_dim": int(encoded_dim),
        "encoded_density": round(encoded_density, 4),
        "encoded_sparsity": round(encoded_sparsity, 4),
    }

    return metrics


def main():
    metrics = run_csv_pipeline()
    print("CSV pipeline metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
