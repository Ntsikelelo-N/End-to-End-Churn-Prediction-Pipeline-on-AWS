"""
tests/test_features.py — Unit tests for feature engineering.

Tests verify that derived features are computed correctly, that the
train/test split preserves the class ratio (stratification), and that
the ColumnTransformer is built with the expected transformer types.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from churn_pipeline.features import (
    add_charges_per_month,
    add_service_count,
    add_tenure_group,
    build_preprocessor,
    engineer_features,
    make_train_test_split,
    split_features_target,
)

# ---------------------------------------------------------------------------
# Shared fixture — pre-cleaned DataFrame (binary encoded, no customerID)
# ---------------------------------------------------------------------------


@pytest.fixture()
def cleaned_sample() -> pd.DataFrame:
    """Synthetic DataFrame in the state produced by run_cleaning_pipeline()."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "gender": np.random.randint(0, 2, n),
            "SeniorCitizen": np.random.randint(0, 2, n),
            "Partner": np.random.randint(0, 2, n),
            "Dependents": np.random.randint(0, 2, n),
            "tenure": np.random.randint(0, 73, n),
            "PhoneService": np.random.randint(0, 2, n),
            "MultipleLines": np.random.choice(["No", "Yes", "No phone service"], n),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": np.random.randint(0, 2, n),
            "OnlineBackup": np.random.randint(0, 2, n),
            "DeviceProtection": np.random.randint(0, 2, n),
            "TechSupport": np.random.randint(0, 2, n),
            "StreamingTV": np.random.randint(0, 2, n),
            "StreamingMovies": np.random.randint(0, 2, n),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": np.random.randint(0, 2, n),
            "PaymentMethod": np.random.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": np.random.uniform(20, 120, n),
            "TotalCharges": np.random.uniform(100, 8000, n),
            "Churn": np.random.choice([0, 1], n, p=[0.74, 0.26]),
        }
    )


# ---------------------------------------------------------------------------
# add_charges_per_month
# ---------------------------------------------------------------------------


class TestAddChargesPerMonth:
    def test_column_is_added(self, cleaned_sample):
        result = add_charges_per_month(cleaned_sample)
        assert "AvgMonthlyCharges" in result.columns

    def test_tenure_zero_uses_monthly_charges(self):
        df = pd.DataFrame({"tenure": [0], "TotalCharges": [0.0], "MonthlyCharges": [55.0]})
        result = add_charges_per_month(df)
        assert result.loc[0, "AvgMonthlyCharges"] == pytest.approx(55.0)

    def test_positive_tenure_divides_correctly(self):
        df = pd.DataFrame({"tenure": [10], "TotalCharges": [500.0], "MonthlyCharges": [55.0]})
        result = add_charges_per_month(df)
        assert result.loc[0, "AvgMonthlyCharges"] == pytest.approx(50.0)

    def test_no_nulls_produced(self, cleaned_sample):
        result = add_charges_per_month(cleaned_sample)
        assert result["AvgMonthlyCharges"].isna().sum() == 0

    def test_does_not_mutate_input(self, cleaned_sample):
        original_cols = list(cleaned_sample.columns)
        _ = add_charges_per_month(cleaned_sample)
        assert list(cleaned_sample.columns) == original_cols


# ---------------------------------------------------------------------------
# add_tenure_group
# ---------------------------------------------------------------------------


class TestAddTenureGroup:
    def test_column_is_added(self, cleaned_sample):
        result = add_tenure_group(cleaned_sample)
        assert "TenureGroup" in result.columns

    def test_zero_tenure_maps_to_first_bucket(self):
        df = pd.DataFrame({"tenure": [0]})
        result = add_tenure_group(df)
        assert result.loc[0, "TenureGroup"] == "0-1yr"

    def test_twelve_months_maps_to_first_bucket(self):
        df = pd.DataFrame({"tenure": [12]})
        result = add_tenure_group(df)
        assert result.loc[0, "TenureGroup"] == "0-1yr"

    def test_seventy_two_months_maps_to_last_bucket(self):
        df = pd.DataFrame({"tenure": [72]})
        result = add_tenure_group(df)
        assert result.loc[0, "TenureGroup"] == "4+yr"

    def test_all_labels_are_known_values(self, cleaned_sample):
        result = add_tenure_group(cleaned_sample)
        valid = {"0-1yr", "1-2yr", "2-4yr", "4+yr"}
        unexpected = set(result["TenureGroup"].unique()) - valid
        assert unexpected == set(), f"Unexpected tenure groups: {unexpected}"


# ---------------------------------------------------------------------------
# add_service_count
# ---------------------------------------------------------------------------


class TestAddServiceCount:
    def test_column_is_added(self, cleaned_sample):
        result = add_service_count(cleaned_sample)
        assert "ServiceCount" in result.columns

    def test_all_services_on_gives_six(self):
        df = pd.DataFrame(
            {
                "OnlineSecurity": [1],
                "OnlineBackup": [1],
                "DeviceProtection": [1],
                "TechSupport": [1],
                "StreamingTV": [1],
                "StreamingMovies": [1],
            }
        )
        result = add_service_count(df)
        assert result.loc[0, "ServiceCount"] == 6

    def test_no_services_gives_zero(self):
        df = pd.DataFrame(
            {
                col: [0]
                for col in [
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                ]
            }
        )
        result = add_service_count(df)
        assert result.loc[0, "ServiceCount"] == 0

    def test_range_is_valid(self, cleaned_sample):
        result = add_service_count(cleaned_sample)
        assert result["ServiceCount"].between(0, 6).all()


# ---------------------------------------------------------------------------
# engineer_features (integration)
# ---------------------------------------------------------------------------


class TestEngineerFeatures:
    def test_all_derived_cols_present(self, cleaned_sample):
        result = engineer_features(cleaned_sample)
        for col in ("AvgMonthlyCharges", "TenureGroup", "ServiceCount"):
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_unchanged(self, cleaned_sample):
        result = engineer_features(cleaned_sample)
        assert len(result) == len(cleaned_sample)


# ---------------------------------------------------------------------------
# split_features_target
# ---------------------------------------------------------------------------


class TestSplitFeaturesTarget:
    def test_target_not_in_features(self, cleaned_sample):
        X, y = split_features_target(cleaned_sample)
        assert "Churn" not in X.columns

    def test_target_series_correct(self, cleaned_sample):
        X, y = split_features_target(cleaned_sample)
        assert y.name == "Churn"
        assert set(y.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# make_train_test_split
# ---------------------------------------------------------------------------


class TestMakeTrainTestSplit:
    def test_sizes_are_correct(self, cleaned_sample):
        X, y = split_features_target(cleaned_sample)
        X_train, X_test, y_train, y_test = make_train_test_split(X, y)
        total = len(X_train) + len(X_test)
        assert total == len(cleaned_sample)

    def test_stratification_preserves_class_ratio(self, cleaned_sample):
        X, y = split_features_target(cleaned_sample)
        X_train, X_test, y_train, y_test = make_train_test_split(X, y)
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        # Allow 5% tolerance — stratify guarantees near-equal rates
        assert abs(train_rate - test_rate) < 0.05


# ---------------------------------------------------------------------------
# build_preprocessor
# ---------------------------------------------------------------------------


class TestBuildPreprocessor:
    def test_returns_column_transformer(self, cleaned_sample):
        from sklearn.compose import ColumnTransformer

        X, _ = split_features_target(engineer_features(cleaned_sample))
        prep = build_preprocessor(X)
        assert isinstance(prep, ColumnTransformer)

    def test_fits_without_error(self, cleaned_sample):
        X, y = split_features_target(engineer_features(cleaned_sample))
        prep = build_preprocessor(X)
        prep.fit(X)  # Should not raise
