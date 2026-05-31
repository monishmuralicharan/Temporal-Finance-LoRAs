#!/usr/bin/env python3
"""Build X / router_features / y tensors from raw bars."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_data_config
from src.data.features import compute_features
from src.data.samples import build_samples, sample_count_table, save_dataset

load_dotenv = __import__("dotenv").load_dotenv
load_dotenv(ROOT / ".env")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "data_config.yaml")
    args = parser.parse_args()
    cfg = load_data_config(args.config)

    raw_dir = ROOT / cfg["paths"]["raw_bars_dir"]
    processed_dir = ROOT / cfg["paths"]["processed_dir"]
    paths = sorted(raw_dir.glob("*.parquet"))
    paths = [p for p in paths if not p.name.startswith("_")]

    if not paths:
        raise SystemExit(f"No parquet files in {raw_dir}. Run scripts/01_fetch_bars.py first.")

    # Market proxy
    spy_path = raw_dir / f"{cfg['features']['market_proxy_ticker']}.parquet"
    if not spy_path.exists():
        raise SystemExit(f"Missing {spy_path}; include SPY in universe or fetch separately.")
    market = pd.read_parquet(spy_path)

    frames: list[pd.DataFrame] = []
    for path in paths:
        ticker = path.stem
        bars = pd.read_parquet(path)
        feat = compute_features(bars, market=market)
        feat["ticker"] = ticker
        frames.append(feat)

    panel = pd.concat(frames, ignore_index=True)
    bundle = build_samples(panel, cfg)
    save_dataset(bundle, processed_dir)

    table = sample_count_table(bundle)
    print(table.to_string(index=False))
    print(f"\nX shape: {bundle.X.shape}")
    print(f"router_features shape: {bundle.router_features.shape}")
    print(f"y shape: {bundle.y.shape}")
    print(f"Saved → {processed_dir}")


if __name__ == "__main__":
    main()
