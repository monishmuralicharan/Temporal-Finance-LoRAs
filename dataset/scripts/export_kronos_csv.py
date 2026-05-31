#!/usr/bin/env python3
"""
Export per-ticker CSVs in KronosPredictor layout for a date window.

Output columns: timestamps, open, high, low, close, volume, amount
See: https://github.com/shiyu-coder/Kronos#3-prepare-input-data
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
from src.data.kronos_export import KRONOS_COLUMNS, export_ticker_csv, write_manifest


def load_universe(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Kronos-ready daily CSVs")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "data_config.yaml")
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (default: config export.start_date)")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (default: config export.end_date)")
    parser.add_argument("--tickers", nargs="*", help="Subset of tickers; default = all raw parquets")
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    export_cfg = cfg.get("export", cfg["fetch"])
    start = args.start or export_cfg.get("start_date", cfg["fetch"]["start_date"])
    end = args.end or export_cfg.get("end_date", cfg["fetch"]["end_date"])

    raw_dir = ROOT / cfg["paths"]["raw_bars_dir"]
    out_dir = ROOT / cfg["paths"]["kronos_dir"]
    manifest_path = ROOT / cfg["paths"]["kronos_manifest"]

    if args.tickers:
        tickers = args.tickers
    else:
        tickers = sorted(
            p.stem for p in raw_dir.glob("*.parquet") if not p.name.startswith("_")
        )

    if not tickers:
        raise SystemExit(f"No tickers found in {raw_dir}. Run scripts/01_fetch_bars.py first.")

    manifest_rows: list[dict] = []
    exported = 0
    for ticker in tickers:
        raw_path = raw_dir / f"{ticker}.parquet"
        if not raw_path.exists():
            manifest_rows.append(
                {"ticker": ticker, "status": "missing_raw", "rows": 0, "start": "", "end": ""}
            )
            continue
        bars = pd.read_parquet(raw_path)
        out_path = out_dir / f"{ticker}.csv"
        n = export_ticker_csv(bars, ticker, out_path, start=start, end=end)
        if n == 0:
            manifest_rows.append(
                {"ticker": ticker, "status": "empty_window", "rows": 0, "start": "", "end": ""}
            )
            continue
        sub = pd.read_csv(out_path, parse_dates=["timestamps"])
        manifest_rows.append(
            {
                "ticker": ticker,
                "status": "ok",
                "rows": n,
                "start": sub["timestamps"].min().strftime("%Y-%m-%d"),
                "end": sub["timestamps"].max().strftime("%Y-%m-%d"),
                "file": str(out_path.relative_to(ROOT)),
            }
        )
        exported += 1

    write_manifest(manifest_rows, manifest_path)

    ok = sum(1 for r in manifest_rows if r["status"] == "ok")
    print(f"Exported {ok}/{len(tickers)} tickers → {out_dir}")
    print(f"Window: {start} → {end}")
    print(f"Columns: {KRONOS_COLUMNS}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
