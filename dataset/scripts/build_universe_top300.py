#!/usr/bin/env python3
"""Rank U.S. stocks by recent dollar volume; write top 300 + SPY."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.data.config import load_data_config
from src.data.massive_rest import MassiveRestClient
from src.data.universe import build_top_liquid_universe, save_universe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--lookback", type=int, default=20)
    args = parser.parse_args()

    cfg = load_data_config()
    out = ROOT / "data" / "universe" / "top_liquid_300.txt"
    client = MassiveRestClient()
    tickers = build_top_liquid_universe(
        client, n=args.n, lookback_days=args.lookback, always_include=["SPY"]
    )
    save_universe(tickers, out)
    print(f"Wrote {len(tickers)} tickers → {out}")
    print("Top 15:", ", ".join(tickers[:15]))


if __name__ == "__main__":
    main()
