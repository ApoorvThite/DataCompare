# shared preprocessing placeholder
# src/ml/preprocessing_common.py

from typing import Tuple, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from src.config import TARGET_COL, RANDOM_STATE


FEATURE_COLS = [
    "title",
    "company_location",
    "experience_level",
    "employment_type",
    "salary_min",
    "salary_max",
]


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select feature columns and target column from canonical DataFrame.
    """
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(str)
    return X, y


def split_train_test(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train test split with fixed random state.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_model_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    """
    Build a scikit-learn pipeline:
    - ColumnTransformer with OneHotEncoder for categoricals
    - LogisticRegression classifier
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    clf = LogisticRegression(max_iter=2000, n_jobs=None)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    return model
