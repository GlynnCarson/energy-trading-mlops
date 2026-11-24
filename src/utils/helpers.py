"""
utils/helpers.py
Shared utilities: config loading, S3 I/O, logging, env helpers.
"""

import io
import os
from pathlib import Path

import boto3
import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"


# ─────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return as dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_pipeline_config() -> dict:
    return load_yaml(CONFIG_DIR / "pipeline.yaml")


# ─────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────


def get_env(key: str, default: str = None) -> str:
    """Get environment variable with optional default."""
    val = os.getenv(key, default)
    if val is None:
        raise EnvironmentError(f"Required environment variable '{key}' is not set.")
    return val


# ─────────────────────────────────────────────
# S3 helpers
# ─────────────────────────────────────────────


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def upload_df_to_s3(df: pd.DataFrame, bucket: str, key: str, fmt: str = "parquet") -> str:
    """Upload a DataFrame to S3 as parquet or CSV."""
    s3 = get_s3_client()
    buffer = io.BytesIO()

    if fmt == "parquet":
        df.to_parquet(buffer, index=True, engine="pyarrow")
        content_type = "application/octet-stream"
    else:
        df.to_csv(buffer, index=True)
        content_type = "text/csv"

    buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue(), ContentType=content_type)
    logger.info(f"Uploaded {len(df)} rows → s3://{bucket}/{key}")
    return f"s3://{bucket}/{key}"


def download_df_from_s3(bucket: str, key: str, fmt: str = "parquet") -> pd.DataFrame:
    """Download a DataFrame from S3."""
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    buffer = io.BytesIO(body)

    if fmt == "parquet":
        df = pd.read_parquet(buffer)
    else:
        df = pd.read_csv(buffer)

    logger.info(f"Downloaded {len(df)} rows ← s3://{bucket}/{key}")
    return df


def list_s3_keys(bucket: str, prefix: str) -> list[str]:
    """List all object keys under a given S3 prefix."""
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


# ─────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────


def date_range_str(start: str, end: str) -> str:
    return f"{start}_to_{end}"


def safe_timestamp(ts) -> pd.Timestamp:
    return pd.Timestamp(ts, tz="UTC") if not isinstance(ts, pd.Timestamp) else ts
