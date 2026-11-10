"""
tests/test_ingestion.py
Unit tests for the data ingestion module using moto (mock AWS).
"""

import pytest
import boto3
import pandas as pd
import numpy as np
from moto import mock_aws
from unittest.mock import patch, MagicMock

from src.ingestion.ingest import SyntheticEnergyGenerator


# ─────────────────────────────────────────────
# Synthetic Generator Tests
# ─────────────────────────────────────────────

class TestSyntheticEnergyGenerator:

    def test_generate_returns_dataframe(self):
        gen = SyntheticEnergyGenerator(seed=42)
        df = gen.generate(days=7)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        gen = SyntheticEnergyGenerator(seed=42)
        df = gen.generate(days=10)
        assert len(df) == 10 * 24  # hourly

    def test_required_columns_present(self):
        gen = SyntheticEnergyGenerator(seed=42)
        df = gen.generate(days=5)
        required = ["price_usd", "demand_mwh", "gas_price_mmbtu", "temperature_c", "renewable_pct"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_price_within_realistic_bounds(self):
        gen = SyntheticEnergyGenerator(seed=42)
        df = gen.generate(days=30)
        assert df["price_usd"].min() >= -10
        assert df["price_usd"].max() <= 600

    def test_index_is_datetime_utc(self):
        gen = SyntheticEnergyGenerator(seed=42)
        df = gen.generate(days=5)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_deterministic_with_same_seed(self):
        df1 = SyntheticEnergyGenerator(seed=99).generate(days=5)
        df2 = SyntheticEnergyGenerator(seed=99).generate(days=5)
        pd.testing.assert_frame_equal(df1, df2)

    def test_renewable_pct_between_0_and_1(self):
        gen = SyntheticEnergyGenerator(seed=42)
        df = gen.generate(days=30)
        assert (df["renewable_pct"] >= 0).all()
        assert (df["renewable_pct"] <= 1).all()

    def test_no_nulls_in_generated_data(self):
        gen = SyntheticEnergyGenerator(seed=42)
        df = gen.generate(days=30)
        assert df.isnull().sum().sum() == 0
