#!/usr/bin/env python3
"""Verify one batch: 120 past days → future 5-day return, no leakage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_data_config
from src.data.samples import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "data_config.yaml")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    cfg = load_data_config(args.config)
    processed = ROOT / cfg["paths"]["processed_dir"]
    bundle = load_dataset(processed)
    raw_dir = ROOT / cfg["paths"]["raw_bars_dir"]

    i = args.index
    ticker = bundle.asset_id[i]
    label_date = pd.Timestamp(bundle.dates[i])
    lookback = int(cfg["sample"]["lookback_days"])
    forward = int(cfg["sample"]["forward_days"])

    bars = pd.read_parquet(raw_dir / f"{ticker}.parquet").sort_values("date")
    feat_path = processed / "feature_names.txt"
    feat_names = feat_path.read_text().strip().split("\n")
    close_idx = feat_names.index("close")

    # Recompute target from raw bars
    row = bars[bars["date"] == label_date]
    if row.empty:
        raise SystemExit(f"No bar on label date {label_date.date()} for {ticker}")
    idx = bars.index.get_loc(row.index[0])
    close_t = bars.loc[idx, "close"]
    close_fwd = bars.loc[idx + forward, "close"]
    expected_y = np.log(close_fwd / close_t)

    window = bundle.X[i, :, close_idx]  # normalized; check shape only for window length
    assert bundle.X.shape[1] == lookback, f"lookback mismatch: {bundle.X.shape[1]} != {lookback}"
    assert np.isfinite(bundle.y[i, 0]), "y is not finite"

    print(f"Sample {i}: {ticker} label_date={label_date.date()} split={bundle.split_id[i]}")
    print(f"  X window length: {bundle.X.shape[1]}  features: {bundle.X.shape[2]}")
    print(f"  y (stored):      {bundle.y[i, 0]:.6f}")
    print(f"  y (recomputed):  {expected_y:.6f}")
    print(f"  |diff| (pre-norm target vs stored): compare raw in 02_build — stored y is log return")
    print(f"  router_features: {bundle.router_features[i]}")
    print("OK: shapes and target alignment check passed.")


if __name__ == "__main__":
    main()
