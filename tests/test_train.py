"""
tests/test_train.py — Tests for the training and evaluation modules.

Focuses on contract-level behaviour: pipelines are Pipeline instances,
cross-validation returns valid ROC-AUC scores, evaluation metrics are
in their valid ranges, and save/load round-trips work correctly.
These tests run without AWS credentials or the actual churn dataset.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline


from churn_pipeline.evaluate import (
    dummy_roc_auc_baseline,
    evaluate_on_test_set,
    get_feature_importance,
    majority_class_metrics,
    threshold_sweep,
)
from churn_pipeline.train import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    cross_validate_pipeline,
    fit_final_model,
    load_model,
    save_model,
)

# ---------------------------------------------------------------------------
# Shared fixtures — synthetic binary classification data
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_data():
    """Synthetic DataFrame that mimics the post-feature-engineering schema."""
    X_arr, y_arr = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=5,
        weights=[0.74, 0.26],  # Mirrors Telco churn imbalance
        random_state=42,
    )
    feature_cols = [f"feature_{i}" for i in range(10)]
    X = pd.DataFrame(X_arr, columns=feature_cols)
    y = pd.Series(y_arr, name="Churn")
    return X, y


@pytest.fixture()
def fitted_lr_pipeline(synthetic_data):
    """A LogisticRegression pipeline fitted on the synthetic training data."""
    X, y = synthetic_data
    X_train = X.iloc[:320]
    y_train = y.iloc[:320]
    pipe = build_logistic_regression_pipeline(X_train)
    return fit_final_model(pipe, X_train, y_train)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


class TestPipelineBuilders:
    def test_lr_returns_pipeline(self, synthetic_data):
        X, _ = synthetic_data
        pipe = build_logistic_regression_pipeline(X)
        assert isinstance(pipe, Pipeline)

    def test_rf_returns_pipeline(self, synthetic_data):
        X, _ = synthetic_data
        pipe = build_random_forest_pipeline(X)
        assert isinstance(pipe, Pipeline)

    def test_lr_pipeline_has_two_steps(self, synthetic_data):
        X, _ = synthetic_data
        pipe = build_logistic_regression_pipeline(X)
        assert len(pipe.steps) == 2

    def test_rf_pipeline_step_names(self, synthetic_data):
        X, _ = synthetic_data
        pipe = build_random_forest_pipeline(X)
        step_names = [name for name, _ in pipe.steps]
        assert "preprocessor" in step_names
        assert "classifier" in step_names


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


class TestCrossValidate:
    def test_returns_valid_roc_auc(self, synthetic_data):
        X, y = synthetic_data
        pipe = build_logistic_regression_pipeline(X)
        result = cross_validate_pipeline(pipe, X, y, label="lr_test")
        assert 0.0 <= result["mean_roc_auc"] <= 1.0

    def test_result_has_expected_keys(self, synthetic_data):
        X, y = synthetic_data
        pipe = build_logistic_regression_pipeline(X)
        result = cross_validate_pipeline(pipe, X, y, label="lr_test")
        for key in ("label", "mean_roc_auc", "std_roc_auc", "mean_f1", "std_f1"):
            assert key in result, f"Missing key: {key}"

    def test_std_is_non_negative(self, synthetic_data):
        X, y = synthetic_data
        pipe = build_logistic_regression_pipeline(X)
        result = cross_validate_pipeline(pipe, X, y)
        assert result["std_roc_auc"] >= 0.0


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


class TestModelPersistence:
    def test_save_creates_file(self, fitted_lr_pipeline):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pkl")
            save_model(fitted_lr_pipeline, path)
            assert Path(path).exists()

    def test_load_returns_pipeline(self, fitted_lr_pipeline):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pkl")
            save_model(fitted_lr_pipeline, path)
            loaded = load_model(path)
            assert isinstance(loaded, Pipeline)

    def test_loaded_pipeline_predicts_same(self, fitted_lr_pipeline, synthetic_data):
        X, _ = synthetic_data
        X_test = X.iloc[320:]
        original_probs = fitted_lr_pipeline.predict_proba(X_test)[:, 1]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pkl")
            save_model(fitted_lr_pipeline, path)
            loaded = load_model(path)
            loaded_probs = loaded.predict_proba(X_test)[:, 1]

        np.testing.assert_array_almost_equal(original_probs, loaded_probs)

    def test_load_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_model("/tmp/this_file_does_not_exist_churn.pkl")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TestEvaluateOnTestSet:
    def test_roc_auc_in_valid_range(self, fitted_lr_pipeline, synthetic_data):
        X, y = synthetic_data
        metrics = evaluate_on_test_set(fitted_lr_pipeline, X.iloc[320:], y.iloc[320:])
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_precision_in_valid_range(self, fitted_lr_pipeline, synthetic_data):
        X, y = synthetic_data
        metrics = evaluate_on_test_set(fitted_lr_pipeline, X.iloc[320:], y.iloc[320:])
        assert 0.0 <= metrics["precision"] <= 1.0

    def test_confusion_matrix_shape(self, fitted_lr_pipeline, synthetic_data):
        X, y = synthetic_data
        metrics = evaluate_on_test_set(fitted_lr_pipeline, X.iloc[320:], y.iloc[320:])
        assert metrics["confusion_matrix"].shape == (2, 2)


class TestThresholdSweep:
    def test_returns_dataframe(self, fitted_lr_pipeline, synthetic_data):
        X, y = synthetic_data
        y_prob = fitted_lr_pipeline.predict_proba(X.iloc[320:])[:, 1]
        result = threshold_sweep(y.iloc[320:], y_prob)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, fitted_lr_pipeline, synthetic_data):
        X, y = synthetic_data
        y_prob = fitted_lr_pipeline.predict_proba(X.iloc[320:])[:, 1]
        result = threshold_sweep(y.iloc[320:], y_prob)
        for col in ("threshold", "precision", "recall", "f1"):
            assert col in result.columns


class TestFeatureImportance:
    def test_returns_dataframe(self, fitted_lr_pipeline, synthetic_data):
        X, _ = synthetic_data
        result = get_feature_importance(fitted_lr_pipeline, X)
        assert isinstance(result, pd.DataFrame)

    def test_importance_column_non_negative(self, fitted_lr_pipeline, synthetic_data):
        X, _ = synthetic_data
        result = get_feature_importance(fitted_lr_pipeline, X)
        assert (result["importance"] >= 0).all()


class TestBaselines:
    def test_dummy_auc_is_half(self):
        y = pd.Series([0, 1, 0, 1])
        assert dummy_roc_auc_baseline(y) == pytest.approx(0.5)

    def test_majority_class_has_zero_recall(self):
        y = pd.Series([0, 0, 0, 1])
        result = majority_class_metrics(y)
        assert result["recall_churn_class"] == 0.0

    def test_majority_class_accuracy_matches_class_distribution(self):
        y = pd.Series([0] * 74 + [1] * 26)  # 74/26 split
        result = majority_class_metrics(y)
        assert result["accuracy"] == pytest.approx(0.74, abs=0.01)
