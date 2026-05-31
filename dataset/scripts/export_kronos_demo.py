#!/usr/bin/env python3
"""
Export 5 stocks + SPY (S&P 500 proxy) as Kronos-style CSVs.

Columns match KronosPredictor inputs:
  x_timestamp, open, high, low, close, volume, amount

When calling Kronos, split into:
  x_df = df[["open", "high", "low", "close", "volume", "amount"]]
  x_timestamp = pd.to_datetime(df["x_timestamp"])
  y_timestamp = pd.Series(future trading days for prediction horizon)
"""

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
from src.data.massive_rest import MassiveRestClient

DEMO_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "SPY"]
KRONOS_CSV_COLUMNS = [
    "x_timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]


def bars_to_kronos_csv_df(bars: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    df = bars.sort_values("date").copy()
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    if df.empty:
        return pd.DataFrame(columns=KRONOS_CSV_COLUMNS)

    vol = df["volume"].astype(float) if "volume" in df.columns else 0.0
    has_vol = "volume" in df.columns and df["volume"].notna().all()

    return pd.DataFrame(
        {
            "x_timestamp": pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"),
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "volume": vol if has_vol else 0.0,
            "amount": (df["close"].astype(float) * vol) if has_vol else 0.0,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "data_config.yaml")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--fetch", action="store_true", help="Download missing tickers via REST")
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    export_cfg = cfg.get("export", cfg["fetch"])
    start = args.start or export_cfg["start_date"]
    end = args.end or export_cfg["end_date"]
    raw_dir = ROOT / cfg["paths"]["raw_bars_dir"]
    out_dir = ROOT / "data" / "kronos" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = MassiveRestClient(request_delay=0.4) if args.fetch else None
    manifest: list[dict] = []
    combined_parts: list[pd.DataFrame] = []

    for ticker in DEMO_TICKERS:
        raw_path = raw_dir / f"{ticker}.parquet"
        if args.fetch and client:
            try:
                bars = client.get_daily_bars(ticker, start, end)
                bars["ticker"] = ticker
                raw_dir.mkdir(parents=True, exist_ok=True)
                bars.to_parquet(raw_path, index=False)
            except Exception as exc:
                print(f"WARN fetch {ticker}: {exc}")

        if not raw_path.exists():
            manifest.append({"ticker": ticker, "status": "missing", "rows": 0})
            continue

        bars = pd.read_parquet(raw_path)
        kdf = bars_to_kronos_csv_df(bars, start, end)
        if kdf.empty:
            manifest.append({"ticker": ticker, "status": "empty_window", "rows": 0})
            continue

        label = "SPY_S&P500" if ticker == "SPY" else ticker
        out_path = out_dir / f"{label}.csv"
        kdf.to_csv(out_path, index=False)

        part = kdf.copy()
        part.insert(0, "ticker", label)
        combined_parts.append(part)

        manifest.append(
            {
                "ticker": label,
                "status": "ok",
                "rows": len(kdf),
                "start": kdf["x_timestamp"].iloc[0],
                "end": kdf["x_timestamp"].iloc[-1],
                "file": str(out_path.relative_to(ROOT)),
            }
        )

    if combined_parts:
        combined = pd.concat(combined_parts, ignore_index=True)
        combined.to_csv(out_dir / "all_tickers_combined.csv", index=False)

    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)

    print(f"Wrote Kronos-style CSVs → {out_dir}")
    print(f"Window: {start} → {end}")
    print(f"Columns: {KRONOS_CSV_COLUMNS}")
    print(pd.DataFrame(manifest).to_string(index=False))


if __name__ == "__main__":
    main()
