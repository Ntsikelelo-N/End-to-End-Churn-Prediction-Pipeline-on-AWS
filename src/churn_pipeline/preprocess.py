"""
preprocess.py — Data cleaning and type correction for the Telco Churn dataset.

CHANGED: The notebook mixed data loading, cleaning, and feature engineering in
the same cells. This module handles *only* cleaning — fixing dtypes, handling
nulls, and stripping identity columns. Feature engineering lives in features.py.
Pure functions make each step independently testable.
"""

import logging

import pandas as pd

from churn_pipeline.config import data as data_cfg

logger = logging.getLogger(__name__)


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce TotalCharges to float, imputing whitespace-only values.

    The IBM Telco dataset ships with 11 rows where TotalCharges is a blank
    string. These correspond to customers with tenure == 0 (brand-new customers
    who have not yet been billed). Imputing with MonthlyCharges is the most
    defensible business choice.

    Args:
        df: Raw DataFrame that still has TotalCharges as object dtype.

    Returns:
        DataFrame with TotalCharges as float64.
    """
    df = df.copy()

    # Replace whitespace-only strings with NaN before coercion
    df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", None, regex=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    n_missing = df["TotalCharges"].isna().sum()
    if n_missing > 0:
        # New customers: TotalCharges ≈ MonthlyCharges for tenure=0
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
        logger.info("Imputed %d missing TotalCharges values with MonthlyCharges.", n_missing)

    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Yes/No columns to 1/0 integers.

    Columns that contain 'No phone service' or 'No internet service' are
    treated as 'No' (0) because they represent the absence of the add-on
    service — semantically equivalent to not having opted in.

    Args:
        df: DataFrame after fix_total_charges().

    Returns:
        DataFrame with binary columns as int8.
    """
    df = df.copy()

    yes_no_map = {"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0}

    for col in data_cfg.binary_yes_no_cols:
        if col not in df.columns:
            logger.warning("Expected binary column '%s' not found; skipping.", col)
            continue
        df[col] = df[col].map(yes_no_map).astype("int8")

    # SeniorCitizen is already 0/1 in the source data
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("int8")

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map the Churn target column to a binary integer.

    Args:
        df: DataFrame with Churn as 'Yes'/'No'.

    Returns:
        DataFrame with Churn as int8 (1 = churned, 0 = retained).

    Note:
        We check is_integer_dtype rather than dtype == object because pandas ≥ 2.2
        may store string columns as pd.StringDtype() rather than dtype('O'). The
        is_integer_dtype guard handles both representations correctly: skip encoding
        only when the column is already a numeric integer type.
    """
    df = df.copy()
    target = df[data_cfg.target_col]
    if not pd.api.types.is_integer_dtype(target):
        df[data_cfg.target_col] = target.map({"Yes": 1, "No": 0}).astype("int8")
    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Encode gender as a binary feature (Male=1, Female=0).

    Args:
        df: DataFrame with gender as 'Male'/'Female'.

    Returns:
        DataFrame with gender as int8.
    """
    df = df.copy()
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).astype("int8")
    return df


def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the customerID column — it carries no predictive signal.

    Args:
        df: DataFrame with customerID present.

    Returns:
        DataFrame without the ID column.
    """
    return df.drop(columns=[data_cfg.id_col], errors="ignore")


def validate_no_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Assert that cleaning produced a null-free DataFrame.

    Raises:
        ValueError: If any nulls remain, with column-level detail.
    """
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    if not cols_with_nulls.empty:
        raise ValueError(f"Cleaning pipeline left nulls in columns:\n{cols_with_nulls.to_string()}")

    logger.info("Data validation passed: no nulls detected.")
    return df


def run_cleaning_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning sequence to a raw DataFrame.

    This is the single entry point for notebook usage and the Glue ETL job.

    Args:
        raw_df: DataFrame loaded directly from the raw CSV.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    logger.info("Starting cleaning pipeline on %d rows.", len(raw_df))

    df = (
        raw_df.pipe(drop_id_column)
        .pipe(fix_total_charges)
        .pipe(encode_binary_columns)
        .pipe(encode_target)
        .pipe(encode_gender)
        .pipe(validate_no_nulls)
    )

    logger.info("Cleaning complete. Output shape: %s.", df.shape)
    return df
