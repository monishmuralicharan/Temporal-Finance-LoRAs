#!/usr/bin/env python3
"""Bulk-load universe tickers from Massive S3 day aggregates into data/raw/bars/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.data.config import load_data_config
from src.data.massive_s3 import download_day_range, discover_day_agg_prefix


def load_universe(path: Path) -> set[str]:
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "data_config.yaml")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    export_cfg = cfg.get("export", cfg["fetch"])
    start = args.start or export_cfg["start_date"]
    end = args.end or export_cfg["end_date"]
    tickers = load_universe(ROOT / cfg["paths"]["universe_file"])
    raw_dir = ROOT / cfg["paths"]["raw_bars_dir"]

    prefix = discover_day_agg_prefix()
    print(f"S3 prefix: {prefix}")
    combined_path = download_day_range(start, end, raw_dir, tickers=tickers, prefix=prefix)
    df = pd.read_parquet(combined_path)
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").drop(columns=["ticker"], errors="ignore")
        g.to_parquet(raw_dir / f"{ticker}.parquet", index=False)
    print(f"Wrote {df['ticker'].nunique()} tickers to {raw_dir}")
    print(f"Rows: {len(df):,}  dates {df['date'].min().date()} → {df['date'].max().date()}")


if __name__ == "__main__":
    main()
