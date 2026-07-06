"""
features.py — Feature engineering for the Telco Churn dataset.

CHANGED: Feature creation was entangled with preprocessing in the notebook.
Separating them means the preprocessing stage can be reused by the Glue ETL job,
while feature engineering remains a modelling-time concern. Using sklearn
ColumnTransformer ensures the same transformations are applied consistently
at training and inference time — preventing data leakage.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_pipeline.config import data as data_cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handcrafted features (business-driven)
# ---------------------------------------------------------------------------


def add_charges_per_month(df: pd.DataFrame) -> pd.DataFrame:
    """Derive average monthly charges over the customer's lifetime.

    For tenure > 0: TotalCharges / tenure.
    For tenure == 0 (brand-new customers): fallback to MonthlyCharges.

    This feature captures whether a customer's spending has been consistent
    with their current monthly rate — useful for spotting mid-contract
    plan changes that are sometimes associated with churn.

    Args:
        df: Cleaned DataFrame with TotalCharges and tenure present.

    Returns:
        DataFrame with an additional 'AvgMonthlyCharges' float column.
    """
    df = df.copy()
    df["AvgMonthlyCharges"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )
    return df


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket tenure (months) into categorical loyalty segments.

    Segments mirror typical telco contract lengths and are informative
    even after tenure is included as a raw numeric feature.

    Args:
        df: DataFrame with numeric tenure column.

    Returns:
        DataFrame with an additional 'TenureGroup' string column.
    """
    bins = [0, 12, 24, 48, 72]
    labels = ["0-1yr", "1-2yr", "2-4yr", "4+yr"]
    df = df.copy()
    df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True).astype(
        str
    )
    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count the total number of add-on services subscribed.

    Add-on services: OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies.

    Customers with more services have greater switching costs, so this
    aggregate feature can act as a proxy for 'stickiness'.

    Args:
        df: DataFrame with binary-encoded service columns.

    Returns:
        DataFrame with an additional 'ServiceCount' int column.
    """
    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    present = [c for c in service_cols if c in df.columns]
    df = df.copy()
    df["ServiceCount"] = df[present].sum(axis=1).astype("int8")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in the correct order.

    Args:
        df: Cleaned DataFrame from preprocess.run_cleaning_pipeline().

    Returns:
        DataFrame with additional derived features ready for the sklearn
        ColumnTransformer.
    """
    logger.info("Engineering features …")
    df = df.pipe(add_charges_per_month).pipe(add_tenure_group).pipe(add_service_count)
    logger.info("Feature engineering complete. Shape: %s.", df.shape)
    return df


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix from the target vector.

    Args:
        df: Engineered DataFrame including the target column.

    Returns:
        Tuple of (X: DataFrame, y: Series).
    """
    X = df.drop(columns=[data_cfg.target_col])
    y = df[data_cfg.target_col]
    return X, y


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split to preserve the churn class ratio (~26 % positive).

    Args:
        X: Feature matrix.
        y: Binary target vector.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_cfg.test_size,
        random_state=data_cfg.random_state,
        stratify=y,  # Critical for imbalanced datasets — do not remove
    )
    logger.info(
        "Train: %d rows (%.1f%% churn) | Test: %d rows (%.1f%% churn).",
        len(y_train),
        y_train.mean() * 100,
        len(y_test),
        y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Sklearn ColumnTransformer
# ---------------------------------------------------------------------------


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer fitted to the feature schema of X.

    The transformer handles:
    - Numeric columns: StandardScaler (zero mean, unit variance).
    - Categorical columns: OneHotEncoder with unknown category handling
      for robustness at inference time.

    Binary-encoded columns (already int8) pass through untouched.

    Args:
        X: Feature DataFrame (used only to infer column lists).

    Returns:
        Unfitted ColumnTransformer ready to be embedded in a Pipeline.
    """
    numeric_features = [c for c in data_cfg.numeric_cols + ["AvgMonthlyCharges"] if c in X.columns]
    categorical_features = [
        c for c in data_cfg.categorical_cols + ["TenureGroup"] if c in X.columns
    ]

    # Only register a transformer when its column list is non-empty.
    # Passing an empty list to ColumnTransformer raises ValueError in some
    # sklearn 1.x builds. remainder="passthrough" handles everything else.
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",  # Binary int8 columns pass through as-is
        verbose_feature_names_out=False,
    )
    logger.info(
        "ColumnTransformer: %d numeric, %d categorical columns.",
        len(numeric_features),
        len(categorical_features),
    )
    return preprocessor
