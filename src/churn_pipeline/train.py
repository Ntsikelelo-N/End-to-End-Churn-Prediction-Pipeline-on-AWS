"""
train.py — Model training with sklearn Pipelines and cross-validation.

CHANGED: Notebook had bare model.fit() calls with no cross-validation, no
class-imbalance handling, and hardcoded hyperparameters. This module wraps each
algorithm in a full sklearn Pipeline (preprocessor → classifier), handles the
~26% positive rate with class_weight='balanced', and returns CV results so the
notebook can compare candidates before committing to a final model.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from churn_pipeline.config import data as data_cfg
from churn_pipeline.config import model as model_cfg
from churn_pipeline.features import build_preprocessor

logger = logging.getLogger(__name__)

# XGBoost is optional — skip gracefully if not installed
try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("xgboost not installed; XGBClassifier will be skipped.")


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------


def build_logistic_regression_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Return a Pipeline: StandardScaler + OneHotEncoder → LogisticRegression.

    class_weight='balanced' up-weights the minority churn class to compensate
    for the ~74/26 class split without requiring over/under-sampling.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "classifier",
                LogisticRegression(
                    C=model_cfg.lr_C,
                    max_iter=model_cfg.lr_max_iter,
                    class_weight="balanced",
                    random_state=data_cfg.random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_random_forest_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Return a Pipeline: preprocessing → RandomForestClassifier."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=model_cfg.rf_n_estimators,
                    max_depth=model_cfg.rf_max_depth,
                    min_samples_leaf=model_cfg.rf_min_samples_leaf,
                    class_weight="balanced",
                    random_state=data_cfg.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Return a Pipeline: preprocessing → XGBClassifier.

    scale_pos_weight compensates for class imbalance by up-weighting
    positive examples: scale_pos_weight ≈ n_negative / n_positive.
    """
    if not _XGB_AVAILABLE:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")

    # ~74/26 ratio → scale_pos_weight ≈ 2.8  (positive_class_rate lives on DataConfig)
    scale_pos_weight = (1 - data_cfg.positive_class_rate) / data_cfg.positive_class_rate

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=model_cfg.xgb_n_estimators,
                    max_depth=model_cfg.xgb_max_depth,
                    learning_rate=model_cfg.xgb_learning_rate,
                    subsample=model_cfg.xgb_subsample,
                    scale_pos_weight=scale_pos_weight,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    label: str = "model",
) -> Dict[str, Any]:
    """Run stratified k-fold cross-validation and return a summary dict.

    Args:
        pipeline: Unfitted sklearn Pipeline.
        X: Feature matrix (training data only — never pass test set here).
        y: Target vector.
        label: Human-readable name for logging.

    Returns:
        Dict with keys: label, mean_roc_auc, std_roc_auc, mean_f1, std_f1.
    """
    cv = StratifiedKFold(n_splits=model_cfg.cv_folds, shuffle=True, random_state=42)
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=["roc_auc", "f1"],
        return_train_score=False,
        n_jobs=-1,
    )
    summary = {
        "label": label,
        "mean_roc_auc": float(np.mean(results["test_roc_auc"])),
        "std_roc_auc": float(np.std(results["test_roc_auc"])),
        "mean_f1": float(np.mean(results["test_f1"])),
        "std_f1": float(np.std(results["test_f1"])),
    }
    logger.info(
        "[%s] ROC-AUC: %.4f ± %.4f | F1: %.4f ± %.4f",
        label,
        summary["mean_roc_auc"],
        summary["std_roc_auc"],
        summary["mean_f1"],
        summary["std_f1"],
    )
    return summary


def compare_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Train and cross-validate all available classifiers, return a ranking.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        DataFrame of CV results sorted by mean ROC-AUC (descending).
    """
    candidates = {
        "LogisticRegression": build_logistic_regression_pipeline(X_train),
        "RandomForest": build_random_forest_pipeline(X_train),
    }
    if _XGB_AVAILABLE:
        candidates["XGBoost"] = build_xgboost_pipeline(X_train)

    records = []
    for name, pipeline in candidates.items():
        logger.info("Cross-validating %s …", name)
        record = cross_validate_pipeline(pipeline, X_train, y_train, label=name)
        records.append(record)

    results_df = (
        pd.DataFrame(records).sort_values("mean_roc_auc", ascending=False).reset_index(drop=True)
    )
    return results_df


# ---------------------------------------------------------------------------
# Final model fitting and persistence
# ---------------------------------------------------------------------------


def fit_final_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit the chosen pipeline on the full training set.

    This is called *after* model selection via cross-validation.
    The test set is never touched during this step.

    Args:
        pipeline: Unfitted pipeline for the selected algorithm.
        X_train: Full training feature matrix.
        y_train: Full training target vector.

    Returns:
        Fitted pipeline.
    """
    logger.info("Fitting final model on %d training samples …", len(y_train))
    pipeline.fit(X_train, y_train)
    logger.info("Model fitting complete.")
    return pipeline


def save_model(pipeline: Pipeline, path: str) -> Path:
    """Persist a fitted pipeline to disk as a pickle file.

    Args:
        pipeline: Fitted sklearn Pipeline to serialise.
        path: Destination file path (e.g. 'models/churn_rf.pkl').

    Returns:
        Path object pointing to the saved file.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Model saved to %s.", dest)
    return dest


def load_model(path: str) -> Pipeline:
    """Load a pickled pipeline from disk.

    Args:
        path: Path to the pickle file produced by save_model().

    Returns:
        Fitted sklearn Pipeline.

    Raises:
        FileNotFoundError: If the pickle does not exist at path.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"No model artifact found at {src}.")
    with src.open("rb") as f:
        pipeline = pickle.load(f)
    logger.info("Model loaded from %s.", src)
    return pipeline
