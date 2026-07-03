"""
tests/test_preprocess.py — Unit tests for the data cleaning pipeline.

CHANGED: The notebook had no tests at all. These tests cover every function in
preprocess.py and are designed to catch regressions when the cleaning logic
changes. They use a minimal synthetic fixture so no network or AWS access is
required — tests run offline in CI.
"""

import numpy as np
import pandas as pd
import pytest

# Adjust path if running tests from repo root without `pip install -e .`
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from churn_pipeline.preprocess import (
    drop_id_column,
    encode_binary_columns,
    encode_gender,
    encode_target,
    fix_total_charges,
    run_cleaning_pipeline,
    validate_no_nulls,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def raw_sample() -> pd.DataFrame:
    """Minimal representative sample of the IBM Telco Churn CSV."""
    return pd.DataFrame(
        {
            "customerID": ["1111-AAAA", "2222-BBBB", "3333-CCCC"],
            "gender": ["Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "Yes", "No"],
            "tenure": [12, 0, 48],
            "PhoneService": ["Yes", "No", "Yes"],
            "MultipleLines": ["No", "No phone service", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["No", "No internet service", "Yes"],
            "OnlineBackup": ["Yes", "No internet service", "No"],
            "DeviceProtection": ["No", "No internet service", "Yes"],
            "TechSupport": ["No", "No internet service", "No"],
            "StreamingTV": ["No", "No internet service", "Yes"],
            "StreamingMovies": ["No", "No internet service", "Yes"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
            "MonthlyCharges": [50.0, 70.0, 90.0],
            # Second row has a blank TotalCharges — the known IBM Telco quirk
            "TotalCharges": ["600.0", " ", "4320.0"],
            "Churn": ["No", "Yes", "No"],
        }
    )


# ---------------------------------------------------------------------------
# drop_id_column
# ---------------------------------------------------------------------------

class TestDropIdColumn:
    def test_removes_customer_id(self, raw_sample):
        result = drop_id_column(raw_sample)
        assert "customerID" not in result.columns

    def test_other_columns_preserved(self, raw_sample):
        original_cols = set(raw_sample.columns) - {"customerID"}
        result = drop_id_column(raw_sample)
        assert original_cols.issubset(set(result.columns))

    def test_does_not_mutate_input(self, raw_sample):
        _ = drop_id_column(raw_sample)
        assert "customerID" in raw_sample.columns  # Original untouched


# ---------------------------------------------------------------------------
# fix_total_charges
# ---------------------------------------------------------------------------

class TestFixTotalCharges:
    def test_blank_is_imputed_with_monthly_charges(self, raw_sample):
        result = fix_total_charges(raw_sample)
        # Row 1 had tenure=0 and blank TotalCharges; expect MonthlyCharges value
        assert result.loc[1, "TotalCharges"] == pytest.approx(70.0)

    def test_output_dtype_is_float(self, raw_sample):
        result = fix_total_charges(raw_sample)
        assert pd.api.types.is_float_dtype(result["TotalCharges"])

    def test_no_nulls_after_imputation(self, raw_sample):
        result = fix_total_charges(raw_sample)
        assert result["TotalCharges"].isna().sum() == 0

    def test_valid_values_are_unchanged(self, raw_sample):
        result = fix_total_charges(raw_sample)
        assert result.loc[0, "TotalCharges"] == pytest.approx(600.0)
        assert result.loc[2, "TotalCharges"] == pytest.approx(4320.0)

    def test_does_not_mutate_input(self, raw_sample):
        original_tc = raw_sample["TotalCharges"].copy()
        _ = fix_total_charges(raw_sample)
        pd.testing.assert_series_equal(raw_sample["TotalCharges"], original_tc)


# ---------------------------------------------------------------------------
# encode_binary_columns
# ---------------------------------------------------------------------------

class TestEncodeBinaryColumns:
    def test_yes_maps_to_one(self, raw_sample):
        result = encode_binary_columns(raw_sample)
        # Row 0: Partner=Yes → 1
        assert result.loc[0, "Partner"] == 1

    def test_no_maps_to_zero(self, raw_sample):
        result = encode_binary_columns(raw_sample)
        # Row 1: Partner=No → 0
        assert result.loc[1, "Partner"] == 0

    def test_no_internet_service_maps_to_zero(self, raw_sample):
        result = encode_binary_columns(raw_sample)
        # Row 1: OnlineSecurity='No internet service' → 0
        assert result.loc[1, "OnlineSecurity"] == 0

    def test_no_phone_service_maps_to_zero(self, raw_sample):
        result = encode_binary_columns(raw_sample)
        # PhoneService='No' on row 1 → 0
        assert result.loc[1, "PhoneService"] == 0

    def test_output_dtype_is_int(self, raw_sample):
        result = encode_binary_columns(raw_sample)
        assert result["Partner"].dtype in (np.int8, np.int16, np.int32, np.int64)


# ---------------------------------------------------------------------------
# encode_target
# ---------------------------------------------------------------------------

class TestEncodeTarget:
    def test_yes_maps_to_one(self, raw_sample):
        result = encode_target(raw_sample)
        assert result.loc[1, "Churn"] == 1

    def test_no_maps_to_zero(self, raw_sample):
        result = encode_target(raw_sample)
        assert result.loc[0, "Churn"] == 0

    def test_no_nulls_in_target(self, raw_sample):
        result = encode_target(raw_sample)
        assert result["Churn"].isna().sum() == 0


# ---------------------------------------------------------------------------
# encode_gender
# ---------------------------------------------------------------------------

class TestEncodeGender:
    def test_male_maps_to_one(self, raw_sample):
        result = encode_gender(raw_sample)
        assert result.loc[0, "gender"] == 1

    def test_female_maps_to_zero(self, raw_sample):
        result = encode_gender(raw_sample)
        assert result.loc[1, "gender"] == 0


# ---------------------------------------------------------------------------
# validate_no_nulls
# ---------------------------------------------------------------------------

class TestValidateNoNulls:
    def test_raises_on_nulls(self):
        df_with_nulls = pd.DataFrame({"a": [1, None], "b": [3, 4]})
        with pytest.raises(ValueError, match="nulls"):
            validate_no_nulls(df_with_nulls)

    def test_passes_clean_dataframe(self):
        df_clean = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = validate_no_nulls(df_clean)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# run_cleaning_pipeline (integration)
# ---------------------------------------------------------------------------

class TestRunCleaningPipeline:
    def test_output_shape_has_no_id_column(self, raw_sample):
        result = run_cleaning_pipeline(raw_sample)
        assert "customerID" not in result.columns

    def test_no_nulls_in_output(self, raw_sample):
        result = run_cleaning_pipeline(raw_sample)
        assert result.isnull().sum().sum() == 0

    def test_target_is_binary_integer(self, raw_sample):
        result = run_cleaning_pipeline(raw_sample)
        assert set(result["Churn"].unique()).issubset({0, 1})

    def test_total_charges_is_float(self, raw_sample):
        result = run_cleaning_pipeline(raw_sample)
        assert pd.api.types.is_float_dtype(result["TotalCharges"])

    def test_does_not_mutate_input(self, raw_sample):
        original_shape = raw_sample.shape
        _ = run_cleaning_pipeline(raw_sample)
        assert raw_sample.shape == original_shape
