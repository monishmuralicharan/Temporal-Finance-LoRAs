#!/usr/bin/env python3
"""Bulk-download day aggregates from Massive S3 flat files (faster than REST)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

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
    start = args.start or cfg["fetch"]["start_date"]
    end = args.end or cfg["fetch"]["end_date"]
    tickers = load_universe(ROOT / cfg["paths"]["universe_file"])
    out_dir = ROOT / cfg["paths"]["raw_bars_dir"]

    prefix = discover_day_agg_prefix()
    print(f"Using S3 prefix: {prefix}")
    out_path = download_day_range(start, end, out_dir, tickers=tickers, prefix=prefix)
    print(f"Wrote combined parquet: {out_path}")
    print("Split per-ticker files with: python scripts/split_combined_bars.py")


if __name__ == "__main__":
    main()
