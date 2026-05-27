"""Download Massive flat-file daily stock aggregates from S3."""

from __future__ import annotations

import gzip
import io
import os
from datetime import date, timedelta
from pathlib import Path

import boto3
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Known layouts (Massive / Polygon); first match wins at runtime
DAY_AGG_PREFIX_CANDIDATES = [
    "us_stocks_sip/day_aggs_v1",
    "us_stocks_sip/day_aggs",
    "stocks/day-aggregates",
]


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MASSIVE_S3_ENDPOINT", "https://files.massive.com"),
        aws_access_key_id=os.environ["MASSIVE_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["MASSIVE_S3_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
    )


def discover_day_agg_prefix(bucket: str | None = None) -> str:
    bucket = bucket or os.getenv("MASSIVE_S3_BUCKET", "flatfiles")
    s3 = _s3_client()
    for prefix in DAY_AGG_PREFIX_CANDIDATES:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/", MaxKeys=5)
        if resp.get("KeyCount", 0) > 0:
            return prefix
    raise FileNotFoundError(
        f"No day-aggregate prefix found in bucket {bucket}. "
        f"Tried: {DAY_AGG_PREFIX_CANDIDATES}"
    )


def _parse_day_file(body: bytes) -> pd.DataFrame:
    with gzip.GzipFile(fileobj=io.BytesIO(body)) as gz:
        df = pd.read_csv(gz)
    # Normalize column names from flat file schema
    colmap = {
        "window_start": "window_start",
        "ticker": "ticker",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    df = df.rename(columns={c: c for c in df.columns})
    if "window_start" in df.columns:
        # nanoseconds → date
        df["date"] = pd.to_datetime(df["window_start"], unit="ns").dt.tz_localize(None).dt.normalize()
    return df


def list_trading_day_keys(
    prefix: str,
    start: date,
    end: date,
    bucket: str | None = None,
) -> list[str]:
    """List S3 keys for daily aggregate files between start and end."""
    bucket = bucket or os.getenv("MASSIVE_S3_BUCKET", "flatfiles")
    s3 = _s3_client()
    keys: list[str] = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        month_prefix = f"{prefix}/{cur.year}/{cur.month:02d}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=month_prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                if not key.endswith(".csv.gz"):
                    continue
                # filename often YYYY-MM-DD.csv.gz
                fname = key.rsplit("/", 1)[-1].replace(".csv.gz", "")
                try:
                    file_date = date.fromisoformat(fname)
                except ValueError:
                    continue
                if start <= file_date <= end:
                    keys.append(key)
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return sorted(keys)


def download_day_range(
    start: str,
    end: str,
    out_dir: Path,
    tickers: set[str] | None = None,
    prefix: str | None = None,
) -> Path:
    """Download daily aggregate CSV.gz files and filter to tickers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    bucket = os.getenv("MASSIVE_S3_BUCKET", "flatfiles")
    prefix = prefix or discover_day_agg_prefix(bucket)
    s3 = _s3_client()
    keys = list_trading_day_keys(prefix, start_d, end_d, bucket)
    if not keys:
        raise FileNotFoundError(f"No S3 keys for {start}..{end} under {prefix}")

    frames: list[pd.DataFrame] = []
    for key in tqdm(keys, desc="S3 day files"):
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = _parse_day_file(obj["Body"].read())
        if tickers is not None:
            df = df[df["ticker"].isin(tickers)]
        if df.empty:
            continue
        frames.append(df[["ticker", "date", "open", "high", "low", "close", "volume"]])

    if not frames:
        raise ValueError("No rows after filtering tickers")
    combined = pd.concat(frames, ignore_index=True)
    out_path = out_dir / f"day_aggs_{start}_{end}.parquet"
    combined.to_parquet(out_path, index=False)
    return out_path
