#!/usr/bin/env python3
"""
Build all_tickers_combined.csv: top-300 liquid names + SPY, 2014 → today.

Columns: ticker, x_timestamp, open, high, low, close, volume, amount
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.data.config import load_data_config
from src.data.kronos_export import KRONOS_COMBINED_COLUMNS, bars_to_kronos_combined_row
from src.data.massive_rest import MassiveRestClient
from src.data.universe import load_universe


def ticker_label(symbol: str) -> str:
    return "SPY_S&P500" if symbol == "SPY" else symbol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "data_config.yaml")
    parser.add_argument("--universe", type=Path, default=None)
    parser.add_argument("--start", type=str, default="2014-01-02")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--fetch", action="store_true", help="Download missing tickers from Massive")
    parser.add_argument("--force-fetch", action="store_true", help="Re-download all tickers")
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    end = args.end or date.today().isoformat()
    start = args.start
    universe_path = args.universe or ROOT / "data" / "universe" / "top_liquid_300.txt"
    if not universe_path.exists():
        raise SystemExit(f"Missing {universe_path}. Run: python scripts/build_universe_top300.py")

    tickers = load_universe(universe_path)
    raw_dir = ROOT / cfg["paths"]["raw_bars_dir"]
    out_dir = ROOT / "data" / "kronos" / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "all_tickers_combined.csv"

    client = MassiveRestClient() if args.fetch else None
    parts: list[pd.DataFrame] = []
    manifest: list[dict] = []

    for ticker in tqdm(tickers, desc="export"):
        label = ticker_label(ticker)
        raw_path = raw_dir / f"{ticker}.parquet"

        if args.fetch and client:
            if args.force_fetch or not raw_path.exists():
                try:
                    bars = client.get_daily_bars(ticker, start, end)
                    if not bars.empty:
                        bars["ticker"] = ticker
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        bars.to_parquet(raw_path, index=False)
                except Exception as exc:
                    manifest.append({"ticker": label, "status": f"fetch_error: {exc}", "rows": 0})
                    continue

        if not raw_path.exists():
            manifest.append({"ticker": label, "status": "missing_raw", "rows": 0})
            continue

        bars = pd.read_parquet(raw_path)
        block = bars_to_kronos_combined_row(bars, label, start, end)
        if block.empty:
            manifest.append({"ticker": label, "status": "empty_window", "rows": 0})
            continue
        parts.append(block)
        manifest.append(
            {
                "ticker": label,
                "status": "ok",
                "rows": len(block),
                "start": block["x_timestamp"].iloc[0],
                "end": block["x_timestamp"].iloc[-1],
            }
        )

    if not parts:
        raise SystemExit("No data exported.")

    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(combined_path, index=False)
    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)

    ok = sum(1 for m in manifest if m["status"] == "ok")
    print(f"\nWrote {combined_path}")
    print(f"Shape: {combined.shape[0]:,} rows × {combined.shape[1]} cols")
    print(f"Tickers OK: {ok}/{len(tickers)}  |  {start} → {end}")
    print(f"Columns: {list(combined.columns)}")


if __name__ == "__main__":
    main()
