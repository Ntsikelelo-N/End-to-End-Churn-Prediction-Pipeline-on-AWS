"""
config.py — Single source of truth for all pipeline constants.

CHANGED: All magic numbers, bucket names, and model hyperparameters that were
scattered across the notebook are centralised here. Changing one value now
propagates everywhere — no more hunting through cells.
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# AWS / Storage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AWSConfig:
    region: str = "us-east-1"
    bucket_name: str = "churn-project-ntsikelelo"
    raw_prefix: str = "raw_data/"
    cleaned_prefix: str = "cleaned_data/"
    model_prefix: str = "models/"
    glue_database: str = "churn_db"
    glue_crawler_name: str = "churn-raw-crawler"
    glue_etl_job_name: str = "churn-etl-job"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    source_url: str = (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
        "/master/data/Telco-Customer-Churn.csv"
    )
    local_raw_path: str = "data/raw/Telco-Customer-Churn.csv"
    local_processed_path: str = "data/processed/churn_features.csv"

    # Column roles
    target_col: str = "Churn"
    id_col: str = "customerID"
    numeric_cols: List[str] = field(
        default_factory=lambda: ["tenure", "MonthlyCharges", "TotalCharges"]
    )
    binary_yes_no_cols: List[str] = field(
        default_factory=lambda: [
            "Partner",
            "Dependents",
            "PhoneService",
            "PaperlessBilling",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
    )
    categorical_cols: List[str] = field(
        default_factory=lambda: [
            "gender",
            "MultipleLines",
            "InternetService",
            "Contract",
            "PaymentMethod",
        ]
    )

    # Class imbalance: ~26 % positive rate in IBM Telco dataset
    positive_class_rate: float = 0.265
    test_size: float = 0.20
    random_state: int = 42


# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    # Logistic Regression
    lr_max_iter: int = 1000
    lr_C: float = 1.0

    # Random Forest
    rf_n_estimators: int = 200
    rf_max_depth: int = 8
    rf_min_samples_leaf: int = 10

    # XGBoost
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8

    # Cross-validation folds
    cv_folds: int = 5

    # Scoring metric — ROC-AUC is more appropriate than accuracy for imbalanced churn data
    scoring_metric: str = "roc_auc"


# ---------------------------------------------------------------------------
# Convenience accessor
# ---------------------------------------------------------------------------

aws = AWSConfig()
data = DataConfig()
model = ModelConfig()
