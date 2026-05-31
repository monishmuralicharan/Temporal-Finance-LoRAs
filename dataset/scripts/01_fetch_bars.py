#!/usr/bin/env python3
"""Download daily OHLCV bars via Massive REST API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_data_config
from src.data.massive_rest import MassiveRestClient

load_dotenv(ROOT / ".env")


# Some list symbols differ from Massive/Polygon API symbols
API_TICKER_ALIASES: dict[str, str] = {
    "BRK.B": "BRK.B",
    "SQ": "SQ",
}


def load_universe(path: Path) -> list[str]:
    tickers = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return tickers


def api_ticker(symbol: str) -> str:
    return API_TICKER_ALIASES.get(symbol, symbol)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch daily bars from Massive REST")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "data_config.yaml")
    parser.add_argument("--tickers", nargs="*", help="Override universe tickers")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument(
        "--exact-range",
        action="store_true",
        help="Do not prepend history_buffer days (use for Kronos Jan 2025–Apr 2026 pulls)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if parquet exists")
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    universe_path = ROOT / cfg["paths"]["universe_file"]
    tickers = args.tickers or load_universe(universe_path)

    fetch = cfg["fetch"]
    start = args.start or fetch["start_date"]
    end = args.end or fetch["end_date"]
    if args.exact_range:
        start_fetch = start
    else:
        buffer = int(fetch["history_buffer_trading_days"])
        start_ts = pd.Timestamp(start) - pd.offsets.BDay(buffer)
        start_fetch = start_ts.strftime("%Y-%m-%d")

    out_dir = ROOT / cfg["paths"]["raw_bars_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    client = MassiveRestClient()
    all_frames: list[pd.DataFrame] = []
    failed: list[str] = []
    skip_existing = not args.force
    for ticker in tqdm(tickers, desc="tickers"):
        path = out_dir / f"{ticker}.parquet"
        if skip_existing and path.exists():
            all_frames.append(pd.read_parquet(path))
            continue
        sym = api_ticker(ticker)
        try:
            df = client.get_daily_bars(sym, start_fetch, end)
        except Exception as exc:
            print(f"WARN {ticker} ({sym}): {exc}")
            failed.append(ticker)
            continue
        if df.empty:
            print(f"WARN {ticker}: no data")
            failed.append(ticker)
            continue
        df["ticker"] = ticker
        df.to_parquet(path, index=False)
        all_frames.append(df)

    if not all_frames:
        raise SystemExit("No tickers downloaded — check MASSIVE_API_KEY and plan limits.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_parquet(out_dir / "_combined.parquet", index=False)
    print(f"Saved {len(all_frames)} tickers → {out_dir}")
    print(f"Date range: {combined['date'].min().date()} → {combined['date'].max().date()}")
    if failed:
        print(f"Failed/missing ({len(failed)}): {', '.join(failed[:20])}{'...' if len(failed) > 20 else ''}")


if __name__ == "__main__":
    main()
