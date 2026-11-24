"""
transformation/transform.py
Cleans raw energy data, engineers features, validates with Great Expectations,
and writes train/val/test splits to S3.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.helpers import (
    download_df_from_s3,
    load_pipeline_config,
    upload_df_to_s3,
)

cfg = load_pipeline_config()
tr = cfg["transformation"]
aws = cfg["aws"]
BUCKET = aws["s3_bucket"]
TARGET = tr["target_column"]


# ─────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────


def clean_raw(df: pd.DataFrame, outlier_stats: dict | None = None) -> pd.DataFrame:
    """Drop duplicates, remove extreme outliers, ensure UTC index.

    outlier_stats: optional {column: {"mean": m, "std": s}} fitted on the training period. When given,
    the outlier rule uses these frozen statistics instead of the whole input's mean and std, so a row's
    cleaning never depends on later rows (used by paper trading; DEF-08). None keeps the original behaviour.
    """
    logger.info(f"[CLEAN] Input shape: {df.shape}")

    # Ensure DatetimeIndex with UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Sort chronologically
    df = df.sort_index()

    # Drop duplicate timestamps
    n_dupes = df.index.duplicated().sum()
    if n_dupes:
        logger.warning(f"Dropping {n_dupes} duplicate timestamps.")
        df = df[~df.index.duplicated(keep="last")]

    # Remove outliers using IQR on numeric columns
    threshold = tr["outlier_std_threshold"]
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if outlier_stats is not None and col in outlier_stats:
            mean, std = outlier_stats[col]["mean"], outlier_stats[col]["std"]
        else:
            mean, std = df[col].mean(), df[col].std()
        mask = (df[col] - mean).abs() <= threshold * std
        removed = (~mask).sum()
        if removed:
            logger.warning(f"Removed {removed} outliers in '{col}'.")
            df.loc[~mask, col] = np.nan

    # Ensure a regular frequency. Auto-detect hourly (synthetic) vs
    # business-daily (real EIA commodity prices) from the index spacing.
    if len(df) > 2:
        med = df.index.to_series().diff().median()
        freq = "H" if med <= pd.Timedelta(hours=1, minutes=30) else "B"
    else:
        freq = "H"
    ffill_limit = 3 if freq == "H" else 5
    df = df.asfreq(freq)
    df = df.ffill(limit=ffill_limit)
    logger.info(f"[CLEAN] Detected frequency='{freq}', ffill_limit={ffill_limit}")

    if tr["drop_na"]:
        before = len(df)
        df = df.dropna(subset=[TARGET])
        logger.info(f"Dropped {before - len(df)} rows missing target '{TARGET}'.")

    logger.info(f"[CLEAN] Output shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-based features."""
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["day_of_year"] = df.index.dayofyear

    # Cyclical encoding (sin/cos) to preserve periodicity
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_lag_features(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """Lag features for the target and key exogenous columns."""
    windows = windows or tr["feature_window_hours"]
    lag_cols = [TARGET]
    if "demand_mwh" in df.columns:
        lag_cols.append("demand_mwh")

    for col in lag_cols:
        for lag in windows:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """Rolling statistics on the target column."""
    windows = windows or tr["rolling_windows"]
    for w in windows:
        rolled = df[TARGET].shift(1).rolling(w, min_periods=1)
        df[f"{TARGET}_roll_{w}h_mean"] = rolled.mean()
        df[f"{TARGET}_roll_{w}h_std"] = rolled.std()
        df[f"{TARGET}_roll_{w}h_min"] = rolled.min()
        df[f"{TARGET}_roll_{w}h_max"] = rolled.max()

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Price-based technical indicators used in energy trading."""
    price = df[TARGET].shift(1)  # avoid look-ahead

    # RSI (14-period)
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12-26 EMA difference)
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26

    # Bollinger Bands (20-period)
    roll20 = price.rolling(20)
    bb_mean = roll20.mean()
    bb_std = roll20.std()
    df["bollinger_upper"] = bb_mean + 2 * bb_std
    df["bollinger_lower"] = bb_mean - 2 * bb_std
    df["bollinger_width"] = df["bollinger_upper"] - df["bollinger_lower"]

    # Price rate of change
    df["price_roc_1h"] = price.pct_change(1)
    df["price_roc_24h"] = price.pct_change(24)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps."""
    logger.info("[FEATURES] Engineering features...")
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_technical_indicators(df)
    # Drop rows with NaN introduced by lags/rolling
    n_before = len(df)
    df = df.dropna()
    logger.info(f"[FEATURES] Dropped {n_before - len(df)} NaN rows after feature engineering.")
    logger.info(f"[FEATURES] Final shape: {df.shape} | {len(df.columns)} features")
    return df


# ─────────────────────────────────────────────
# Train / Val / Test Split
# ─────────────────────────────────────────────


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split preserving time order.
    train_ratio / val_ratio / test_ratio from config.
    """
    n = len(df)
    train_end = int(n * tr["train_ratio"])
    val_end = train_end + int(n * tr["val_ratio"])

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    logger.info(
        f"[SPLIT] train={len(train)} | val={len(val)} | test={len(test)} "
        f"({tr['train_ratio']}/{tr['val_ratio']}/{tr['test_ratio']})"
    )
    return train, val, test


# ─────────────────────────────────────────────
# Data Validation (Great Expectations-style rules)
# ─────────────────────────────────────────────


def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Lightweight schema + quality checks.
    Returns a report dict; raises on critical failures.
    """
    report = {"passed": True, "checks": []}

    def check(name, condition, critical=True):
        result = {"name": name, "passed": bool(condition)}
        if not condition:
            if critical:
                report["passed"] = False
                logger.error(f"[VALIDATION] FAILED (critical): {name}")
            else:
                logger.warning(f"[VALIDATION] FAILED (warning): {name}")
        else:
            logger.debug(f"[VALIDATION] OK: {name}")
        report["checks"].append(result)

    target_exists = TARGET in df.columns
    check("target_column_exists", target_exists)
    if target_exists:
        check("no_null_target", df[TARGET].isnull().sum() == 0)
        check("target_is_numeric", pd.api.types.is_numeric_dtype(df[TARGET]))
        check("price_non_negative", (df[TARGET] >= -10).all(), critical=False)
        check("price_not_all_zero", df[TARGET].std() > 0)
    check("min_row_count", len(df) >= 100)
    check("index_is_datetime", isinstance(df.index, pd.DatetimeIndex))
    check("index_monotonic", df.index.is_monotonic_increasing)

    if not report["passed"]:
        raise ValueError(f"Data validation failed. Report: {report}")

    logger.success("[VALIDATION] All critical checks passed.")
    return report


# ─────────────────────────────────────────────
# Main transform runner
# ─────────────────────────────────────────────


def run_transformation(raw_s3_path: str) -> dict:
    """
    Main entrypoint called by the Airflow DAG.
    Returns dict of S3 paths for train/val/test splits.
    """
    logger.info(f"[TRANSFORM] Loading raw data from {raw_s3_path}")

    # Parse bucket/key from s3://bucket/key
    _, _, bucket_key = raw_s3_path.partition("s3://")
    bucket, _, key = bucket_key.partition("/")

    df = download_df_from_s3(bucket, key)
    df = clean_raw(df)
    validate_dataframe(df)
    df = engineer_features(df)

    train, val, test = temporal_split(df)
    prefix = tr["processed_s3_prefix"]
    date_slug = datetime.utcnow().strftime("%Y%m%d_%H")

    paths = {}
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        key_out = f"{prefix}{date_slug}/{split_name}.parquet"
        paths[split_name] = upload_df_to_s3(split_df, bucket, key_out)

    logger.success(f"[TRANSFORM] Complete. Splits → {paths}")
    return paths


if __name__ == "__main__":
    # Quick local smoke test using synthetic data
    from src.ingestion.ingest import SyntheticEnergyGenerator

    gen = SyntheticEnergyGenerator()
    raw_df = gen.generate(days=90)
    cleaned = clean_raw(raw_df)
    validate_dataframe(cleaned)
    featured = engineer_features(cleaned)
    train, val, test = temporal_split(featured)
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
    print(f"Features: {list(featured.columns)}")
