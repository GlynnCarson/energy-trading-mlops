"""
ingestion/ingest.py
Fetches raw energy market data from the EIA API (or generates synthetic data)
and persists it to S3 as Parquet.
"""

import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from src.utils.helpers import (
    load_pipeline_config,
    upload_df_to_s3,
    get_env,
)

cfg = load_pipeline_config()
ing = cfg["ingestion"]
aws = cfg["aws"]


# ─────────────────────────────────────────────
# EIA API Ingestion
# ─────────────────────────────────────────────

class EIAIngester:
    """
    Pulls time-series data from the U.S. Energy Information Administration API v2.
    Free API key at: https://www.eia.gov/opendata/
    """

    BASE_URL = ing["eia_base_url"]

    # Daily-price series available on EIA's free API v2. Each entry is
    # (route, series_id) verified against the live API.
    SERIES = {
        "gas": ("natural-gas/pri/fut/data", "RNGWHHD"),   # Henry Hub spot, $/MMBtu
        "oil": ("petroleum/pri/spt/data", "RWTC"),        # WTI crude spot, $/bbl
    }

    def __init__(self, api_key: str = None):
        self.api_key = api_key or get_env("EIA_API_KEY")
        self.session = requests.Session()

    def _get(self, route: str, params: dict) -> dict:
        params = {**params, "api_key": self.api_key}
        for attempt in range(ing["retry_attempts"]):
            try:
                resp = self.session.get(
                    f"{self.BASE_URL}/{route}",
                    params=params,
                    timeout=ing["timeout_seconds"],
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        raise RuntimeError(f"All {ing['retry_attempts']} attempts failed for {route}")

    def _fetch_series(self, route: str, series: str, start: str, colname: str) -> pd.DataFrame:
        """Fetch a single daily EIA price series into a one-column DataFrame."""
        data = self._get(
            route,
            {
                "frequency": "daily",
                "data[0]": "value",
                "facets[series][]": series,
                "start": start,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": 0,
                "length": 5000,
            },
        )
        records = data.get("response", {}).get("data", [])
        df = pd.DataFrame(records)
        if df.empty:
            raise RuntimeError(f"EIA returned no data for series '{series}' (route {route}).")
        df["timestamp"] = pd.to_datetime(df["period"], utc=True)
        df[colname] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["timestamp", colname]].dropna().set_index("timestamp")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        logger.info(
            f"Fetched {len(df)} '{colname}' records "
            f"({df.index.min().date()} → {df.index.max().date()})."
        )
        return df

    def build_dataset(self, start: str = "2010-01-01", commodity: str = "gas") -> pd.DataFrame:
        """
        Build a real daily energy dataset from the EIA API.
        The chosen `commodity` (gas|oil) becomes the tradeable target `price_usd`;
        the other commodity is joined in as an exogenous feature.
        """
        commodity = commodity if commodity in self.SERIES else "gas"
        other = "oil" if commodity == "gas" else "gas"

        price_route, price_series = self.SERIES[commodity]
        exog_route, exog_series = self.SERIES[other]

        price = self._fetch_series(price_route, price_series, start, "price_usd")
        exog = self._fetch_series(exog_route, exog_series, start, f"{other}_price_usd")

        df = price.join(exog, how="left").ffill().dropna()
        df["data_source"] = "eia_api"
        logger.success(
            f"[EIA] Built dataset: {len(df)} rows | target=price_usd ({commodity}) | "
            f"cols={list(df.columns)}"
        )
        return df


# ─────────────────────────────────────────────
# Synthetic Data Generator (dev / fallback)
# ─────────────────────────────────────────────

class SyntheticEnergyGenerator:
    """
    Generates realistic synthetic hourly energy trading data for
    development and testing purposes when the EIA API is unavailable.
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate(self, days: int = 365) -> pd.DataFrame:
        """
        Generate `days` days of synthetic hourly energy market data.
        Simulates:
          - price_usd: electricity spot price ($/MWh) with seasonality + noise
          - demand_mwh: demand with daily/weekly seasonality
          - gas_price_mmbtu: correlated gas price
          - temperature_c: outside temperature affecting demand
          - renewable_pct: % of generation from renewables
        """
        n_hours = days * 24
        idx = pd.date_range(
            end=pd.Timestamp.utcnow().floor("H"),
            periods=n_hours,
            freq="H",
            tz="UTC",
        )

        hours = idx.hour
        dow = idx.dayofweek
        month = idx.month

        # --- Temperature (seasonal + daily) ---
        temp_seasonal = 15 * np.sin(2 * np.pi * (month - 3) / 12)
        temp_daily = 5 * np.sin(2 * np.pi * (hours - 6) / 24)
        temperature = temp_seasonal + temp_daily + np.random.normal(0, 2, n_hours)

        # --- Demand (driven by temp, hour, weekday) ---
        demand_base = 350_000  # MWh
        demand_hourly = 40_000 * np.sin(np.pi * hours / 12 - np.pi / 2) + 40_000
        demand_weekend_adj = np.where(dow >= 5, -30_000, 0)
        demand_temp_adj = 2_000 * np.abs(temperature - 18)
        demand_noise = np.random.normal(0, 5_000, n_hours)
        demand_mwh = demand_base + demand_hourly + demand_weekend_adj + demand_temp_adj + demand_noise

        # --- Renewable penetration ---
        renewable_base = 0.25
        renewable_hour = 0.15 * np.clip(np.sin(np.pi * (hours - 6) / 12), 0, 1)
        renewable_seasonal = 0.10 * np.sin(2 * np.pi * (month - 4) / 12)
        renewable_pct = np.clip(renewable_base + renewable_hour + renewable_seasonal + np.random.normal(0, 0.02, n_hours), 0, 1)

        # --- Gas price (random walk with mean reversion) ---
        gas_price = np.zeros(n_hours)
        gas_price[0] = 3.5
        for i in range(1, n_hours):
            gas_price[i] = gas_price[i-1] + 0.01 * (3.5 - gas_price[i-1]) + np.random.normal(0, 0.05)
        gas_price = np.clip(gas_price, 1.5, 10.0)

        # --- Electricity price (correlated with demand + gas + renewables) ---
        price_base = 40.0
        price_demand_component = 0.0002 * (demand_mwh - demand_base)
        price_gas_component = 3.5 * (gas_price - 3.5)
        price_renewable_adj = -15 * (renewable_pct - 0.25)
        price_noise = np.random.normal(0, 3, n_hours)
        # Occasional price spikes (demand events)
        spike_mask = np.random.random(n_hours) < 0.005
        spike_magnitude = np.random.uniform(50, 200, n_hours) * spike_mask
        price_usd = (
            price_base
            + price_demand_component
            + price_gas_component
            + price_renewable_adj
            + price_noise
            + spike_magnitude
        )
        price_usd = np.clip(price_usd, -10, 500)

        df = pd.DataFrame(
            {
                "price_usd": price_usd,
                "demand_mwh": demand_mwh,
                "gas_price_mmbtu": gas_price,
                "temperature_c": temperature,
                "renewable_pct": renewable_pct,
                "is_weekend": (dow >= 5).astype(int),
                "hour_of_day": hours,
                "day_of_week": dow,
                "month": month,
            },
            index=idx,
        )
        df.index.name = "timestamp"
        logger.info(f"Generated {len(df)} rows of synthetic energy data.")
        return df


# ─────────────────────────────────────────────
# Main ingest runner
# ─────────────────────────────────────────────

def run_ingestion(execution_date: str = None) -> str:
    """
    Main entrypoint called by the Airflow DAG.
    Returns the S3 path of the saved raw data.
    """
    source = ing.get("source", "synthetic")
    bucket = aws["s3_bucket"]
    prefix = ing["raw_s3_prefix"]

    end_date = execution_date or datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=ing["lookback_days"])
    ).strftime("%Y-%m-%d")

    logger.info(f"[INGESTION] source={source} | {start_date} → {end_date}")

    if source == "eia_api":
        ingester = EIAIngester()
        eia_start = ing.get("eia_start_date", "2010-01-01")
        df = ingester.build_dataset(start=eia_start, commodity=ing.get("commodity", "gas"))
    else:
        gen = SyntheticEnergyGenerator()
        df = gen.generate(days=ing["lookback_days"])

    # Add metadata
    df["ingestion_timestamp"] = pd.Timestamp.utcnow()
    df["data_source"] = source

    # Persist to S3
    s3_key = f"{prefix}{start_date}_to_{end_date}/raw_energy.parquet"
    s3_path = upload_df_to_s3(df, bucket, s3_key)

    logger.success(f"[INGESTION] Complete. {len(df)} rows → {s3_path}")
    return s3_path


if __name__ == "__main__":
    run_ingestion()
