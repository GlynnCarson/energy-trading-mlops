"""
tests/test_transformation.py
Unit tests for the transformation and feature engineering pipeline.
"""

import pytest
import numpy as np
import pandas as pd

from src.ingestion.ingest import SyntheticEnergyGenerator
from src.transformation.transform import (
    clean_raw,
    add_time_features,
    add_lag_features,
    add_rolling_features,
    add_technical_indicators,
    engineer_features,
    temporal_split,
    validate_dataframe,
)

TARGET = "price_usd"


@pytest.fixture(scope="module")
def raw_df():
    return SyntheticEnergyGenerator(seed=42).generate(days=90)


@pytest.fixture(scope="module")
def cleaned_df(raw_df):
    return clean_raw(raw_df)


@pytest.fixture(scope="module")
def featured_df(cleaned_df):
    return engineer_features(cleaned_df.copy())


# ─────────────────────────────────────────────
# Cleaning Tests
# ─────────────────────────────────────────────

class TestCleanRaw:

    def test_returns_dataframe(self, raw_df):
        result = clean_raw(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_datetime(self, cleaned_df):
        assert isinstance(cleaned_df.index, pd.DatetimeIndex)

    def test_index_is_monotonic(self, cleaned_df):
        assert cleaned_df.index.is_monotonic_increasing

    def test_no_duplicates(self, cleaned_df):
        assert not cleaned_df.index.duplicated().any()

    def test_no_null_target(self, cleaned_df):
        assert cleaned_df[TARGET].isnull().sum() == 0

    def test_handles_injected_duplicates(self):
        gen = SyntheticEnergyGenerator(seed=1)
        df = gen.generate(days=10)
        # Inject duplicate
        dup = df.iloc[[0]]
        df_with_dup = pd.concat([df, dup])
        cleaned = clean_raw(df_with_dup)
        assert not cleaned.index.duplicated().any()


# ─────────────────────────────────────────────
# Feature Engineering Tests
# ─────────────────────────────────────────────

class TestTimeFeatures:

    def test_all_time_features_present(self, cleaned_df):
        df = add_time_features(cleaned_df.copy())
        for col in ["hour_of_day", "day_of_week", "month", "quarter", "is_weekend",
                    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            assert col in df.columns, f"Missing: {col}"

    def test_hour_range(self, cleaned_df):
        df = add_time_features(cleaned_df.copy())
        assert df["hour_of_day"].between(0, 23).all()

    def test_cyclical_sin_cos_bounds(self, cleaned_df):
        df = add_time_features(cleaned_df.copy())
        assert df["hour_sin"].between(-1.01, 1.01).all()
        assert df["hour_cos"].between(-1.01, 1.01).all()


class TestLagFeatures:

    def test_lag_columns_created(self, cleaned_df):
        df = add_lag_features(cleaned_df.copy(), windows=[1, 24])
        assert f"{TARGET}_lag_1h" in df.columns
        assert f"{TARGET}_lag_24h" in df.columns

    def test_lag_values_shifted_correctly(self, cleaned_df):
        df = add_lag_features(cleaned_df.copy(), windows=[1])
        col = f"{TARGET}_lag_1h"
        # Second row lag should equal first row target
        assert df[col].iloc[1] == pytest.approx(df[TARGET].iloc[0])


class TestRollingFeatures:

    def test_rolling_columns_created(self, cleaned_df):
        df = add_rolling_features(cleaned_df.copy(), windows=[24])
        assert f"{TARGET}_roll_24h_mean" in df.columns
        assert f"{TARGET}_roll_24h_std" in df.columns


class TestTechnicalIndicators:

    def test_rsi_created(self, cleaned_df):
        df = add_technical_indicators(cleaned_df.copy())
        assert "rsi_14" in df.columns

    def test_rsi_bounds(self, cleaned_df):
        df = add_technical_indicators(cleaned_df.copy()).dropna()
        assert df["rsi_14"].between(0, 100).all()

    def test_macd_created(self, cleaned_df):
        df = add_technical_indicators(cleaned_df.copy())
        assert "macd" in df.columns

    def test_bollinger_bands_created(self, cleaned_df):
        df = add_technical_indicators(cleaned_df.copy())
        assert "bollinger_upper" in df.columns
        assert "bollinger_lower" in df.columns

    def test_bollinger_upper_gte_lower(self, cleaned_df):
        df = add_technical_indicators(cleaned_df.copy()).dropna()
        assert (df["bollinger_upper"] >= df["bollinger_lower"]).all()


# ─────────────────────────────────────────────
# Full pipeline Tests
# ─────────────────────────────────────────────

class TestEngineerFeatures:

    def test_returns_more_columns_than_input(self, cleaned_df, featured_df):
        assert len(featured_df.columns) > len(cleaned_df.columns)

    def test_no_nulls_after_engineering(self, featured_df):
        assert featured_df.isnull().sum().sum() == 0

    def test_target_column_preserved(self, featured_df):
        assert TARGET in featured_df.columns


# ─────────────────────────────────────────────
# Split Tests
# ─────────────────────────────────────────────

class TestTemporalSplit:

    def test_split_sizes_correct(self, featured_df):
        train, val, test = temporal_split(featured_df)
        total = len(train) + len(val) + len(test)
        assert total == len(featured_df)

    def test_splits_are_chronological(self, featured_df):
        train, val, test = temporal_split(featured_df)
        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]

    def test_no_data_leakage(self, featured_df):
        train, val, test = temporal_split(featured_df)
        assert len(set(train.index) & set(test.index)) == 0


# ─────────────────────────────────────────────
# Validation Tests
# ─────────────────────────────────────────────

class TestValidateDataframe:

    def test_valid_df_passes(self, cleaned_df):
        report = validate_dataframe(cleaned_df)
        assert report["passed"] is True

    def test_missing_target_raises(self, cleaned_df):
        df_bad = cleaned_df.drop(columns=[TARGET])
        with pytest.raises(ValueError):
            validate_dataframe(df_bad)

    def test_null_target_raises(self, cleaned_df):
        df_bad = cleaned_df.copy()
        df_bad.loc[df_bad.index[0], TARGET] = np.nan
        with pytest.raises(ValueError):
            validate_dataframe(df_bad)
