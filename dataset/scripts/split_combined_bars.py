#!/usr/bin/env python3
"""Split _combined.parquet into per-ticker files for the sample builder."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_data_config


def main() -> None:
    cfg = load_data_config()
    raw_dir = ROOT / cfg["paths"]["raw_bars_dir"]
    combined = raw_dir / "_combined.parquet"
    if not combined.exists():
        raise SystemExit(f"Missing {combined}")
    df = pd.read_parquet(combined)
    for ticker, g in df.groupby("ticker"):
        g.sort_values("date").to_parquet(raw_dir / f"{ticker}.parquet", index=False)
    print(f"Wrote {df['ticker'].nunique()} ticker files to {raw_dir}")


if __name__ == "__main__":
    main()
