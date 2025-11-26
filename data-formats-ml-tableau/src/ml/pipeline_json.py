# src/ml/pipeline_json.py

import time
from typing import Dict

import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import DATA_FORMATTED, RANDOM_STATE
from src.formats.json_loader import json_to_canonical_df
from src.ml.preprocessing_common import (
    prepare_features_and_target,
    split_train_test,
    build_model_pipeline,
)
from src.ml.features_from_canonical import NUMERIC_FEATURES, CATEGORICAL_FEATURES


NESTED_JSON_FILENAME = "salaries_nested.json"


def compute_encoded_stats(model, X_train):
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


def run_json_pipeline() -> Dict[str, float]:
    """
    Train and evaluate a model using the nested JSON representation.
    Returns a metrics dict including timing breakdown, cross validation stats,
    and encoded feature space stats.
    """
    path = DATA_FORMATTED / NESTED_JSON_FILENAME
    df = json_to_canonical_df(path)

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

    # Prep phase timing
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

    # Encoded feature space stats
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
        "format": "JSON",
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
    metrics = run_json_pipeline()
    print("JSON pipeline metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
