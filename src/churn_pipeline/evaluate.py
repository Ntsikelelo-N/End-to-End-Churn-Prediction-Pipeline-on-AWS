"""
evaluate.py — Evaluation metrics and reporting for the churn classifier.

CHANGED: The notebook had no evaluation at all (unfinished). This module
produces the metrics that a senior reviewer expects: ROC-AUC (correct for
imbalanced data), precision/recall at multiple thresholds, a confusion matrix,
and feature importance — all returned as clean DataFrames for notebook display.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def evaluate_on_test_set(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute a full suite of evaluation metrics on the held-out test set.

    Uses predicted probabilities rather than hard predictions so that
    threshold-sensitive metrics (precision, recall) can be explored.

    Args:
        pipeline: Fitted sklearn Pipeline with a predict_proba method.
        X_test: Test feature matrix.
        y_test: True binary labels for test samples.
        threshold: Decision threshold for converting probabilities to labels.
                   0.5 is the default; tune downward to increase recall on
                   the churn class at the cost of more false positives.

    Returns:
        Dict containing roc_auc, f1, precision, recall, confusion_matrix,
        classification_report, and the raw probability array.
    """
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "confusion_matrix": cm,
        "classification_report": report,
        "y_prob": y_prob,
        "threshold": threshold,
    }

    logger.info(
        "Test set | ROC-AUC: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
        roc_auc,
        f1,
        metrics["precision"],
        metrics["recall"],
    )
    return metrics


def threshold_sweep(
    y_test: pd.Series,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute precision, recall, and F1 across a range of decision thresholds.

    In churn prevention, the business cost of missing a churner (false negative)
    is usually much higher than the cost of a wasted retention offer (false
    positive). This table lets a stakeholder pick the threshold that fits their
    cost structure.

    Args:
        y_test: True binary labels.
        y_prob: Predicted probabilities for the positive class.
        thresholds: Array of threshold values to evaluate. Defaults to 0.1–0.9.

    Returns:
        DataFrame with columns: threshold, precision, recall, f1.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    records = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        records.append(
            {
                "threshold": round(float(t), 2),
                "precision": round(
                    float(
                        np.where(
                            y_pred.sum() > 0,
                            (y_pred & y_test.values).sum() / y_pred.sum(),
                            0.0,
                        )
                    ),
                    4,
                ),
                "recall": round(
                    float((y_pred & y_test.values).sum() / y_test.sum()), 4
                ),
                "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            }
        )
    return pd.DataFrame(records)


def get_roc_curve_data(
    y_test: pd.Series, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) arrays for plotting an ROC curve.

    Args:
        y_test: True binary labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Tuple of (false positive rates, true positive rates, thresholds).
    """
    return roc_curve(y_test, y_prob)


def get_precision_recall_curve_data(
    y_test: pd.Series, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision, recall, thresholds) for a precision-recall curve.

    The PR curve is more informative than ROC when the positive class is rare.

    Args:
        y_test: True binary labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Tuple of (precision, recall, thresholds).
    """
    return precision_recall_curve(y_test, y_prob)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def get_feature_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract feature importances from tree-based or linear models.

    Works with RandomForest (feature_importances_) and LogisticRegression
    (coef_). Returns a clean DataFrame sorted by absolute importance.

    Args:
        pipeline: Fitted sklearn Pipeline whose final step is a classifier.
        X: Feature matrix (used only to infer post-transform feature names).
        top_n: Number of top features to return.

    Returns:
        DataFrame with columns: feature, importance, sorted descending.
    """
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Recover feature names after ColumnTransformer
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        logger.warning(
            "Classifier %s has no feature_importances_ or coef_. "
            "Returning empty DataFrame.",
            type(classifier).__name__,
        )
        return pd.DataFrame(columns=["feature", "importance"])

    df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Baseline comparison (mandatory for credible ML work)
# ---------------------------------------------------------------------------


def dummy_roc_auc_baseline(y_test: pd.Series) -> float:
    """Return the ROC-AUC of a random classifier (always 0.5).

    Including this in the notebook makes it explicit that the model is
    better than chance — something that is easy to forget when working with
    imbalanced data where even a 'predict all negatives' classifier achieves
    ~74% accuracy.

    Args:
        y_test: True binary labels.

    Returns:
        0.5 — the expected AUC of a uniform random classifier.
    """
    return 0.5


def majority_class_metrics(y_test: pd.Series) -> Dict[str, float]:
    """Metrics for a classifier that always predicts the majority class (0).

    Demonstrates why accuracy is the wrong metric for churn detection.

    Args:
        y_test: True binary labels.

    Returns:
        Dict with accuracy, recall, precision, f1 for the majority-class model.
    """
    churn_rate = float(y_test.mean())
    return {
        "accuracy": round(1 - churn_rate, 4),
        "recall_churn_class": 0.0,
        "precision_churn_class": 0.0,
        "f1_churn_class": 0.0,
        "note": (
            f"Predicting 'no churn' for all {len(y_test)} customers achieves "
            f"{(1 - churn_rate) * 100:.1f}% accuracy but zero recall on the "
            f"churn class — useless for retention."
        ),
    }
