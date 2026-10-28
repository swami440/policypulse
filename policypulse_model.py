"""Shared data preparation for the PolicyPulse training and prediction scripts."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TEXT_COLUMNS = ["policy_name", "policy_description", "policy_objective"]
CATEGORICAL_COLUMNS = [
    "policy_domain",
    "policy_type",
    "target_population",
    "target_sector",
    "geographic_scope",
    "policy_duration",
]
NUMERIC_COLUMNS = [
    "policy_year",
    "mandatory",
    "financial_burden",
    "benefit_level",
    "implementation_cost",
    "policy_intensity",
    "coverage_level",
    "essential_service",
]
FEATURE_COLUMNS = TEXT_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
ACCEPTANCE_LABELS = {"Accepted", "Rejected"}


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a policy CSV, preserving empty cells as missing values."""
    return pd.read_csv(path)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select and normalize policy attributes without survey or target leakage."""
    features = frame.reindex(columns=FEATURE_COLUMNS).copy()

    text = frame.reindex(columns=TEXT_COLUMNS).astype("string").fillna("").astype(str)
    features["policy_text"] = text.agg(" ".join, axis=1).str.strip()
    features = features.drop(columns=TEXT_COLUMNS)

    for column in CATEGORICAL_COLUMNS:
        features[column] = (
            features[column].astype("string").fillna("unknown").astype(str)
        )
    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    return features


def make_preprocessor() -> ColumnTransformer:
    """Build a compact feature pipeline suitable for mixed policy data."""
    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=500,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
                "policy_text",
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_COLUMNS,
            ),
        ],
        remainder="drop",
    )
