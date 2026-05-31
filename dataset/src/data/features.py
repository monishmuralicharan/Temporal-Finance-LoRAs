"""Causal feature engineering for daily equity bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLS = ["open", "high", "low", "close", "volume"]
DERIVED_COLS = [
    "daily_return",
    "log_return",
    "return_5d",
    "vol_20d",
    "vol_60d",
    "volume_zscore",
    "ma_distance_20",
    "drawdown",
    "market_return",
]
ROUTER_COLS = ["vol_20d", "vol_60d", "drawdown", "ma_distance_20", "market_return"]


def feature_columns(include_raw: bool = True) -> list[str]:
    return (RAW_COLS if include_raw else []) + DERIVED_COLS


def compute_features(bars: pd.DataFrame, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    bars: columns [date, open, high, low, close, volume] sorted by date.
    market: SPY (or proxy) with columns [date, close] for market_return.
    """
    df = bars.sort_values("date").copy()
    close = df["close"]
    log_ret = np.log(close / close.shift(1))
    df["daily_return"] = close.pct_change()
    df["log_return"] = log_ret
    df["return_5d"] = close / close.shift(5) - 1.0
    df["vol_20d"] = log_ret.rolling(20, min_periods=20).std()
    df["vol_60d"] = log_ret.rolling(60, min_periods=60).std()
    vol_mean = df["volume"].rolling(60, min_periods=60).mean()
    vol_std = df["volume"].rolling(60, min_periods=60).std()
    df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    ma20 = close.rolling(20, min_periods=20).mean()
    df["ma_distance_20"] = close / ma20 - 1.0
    roll_max = close.rolling(60, min_periods=60).max()
    df["drawdown"] = close / roll_max - 1.0

    if market is not None:
        m = market[["date", "close"]].rename(columns={"close": "mkt_close"}).sort_values("date")
        m["market_return"] = np.log(m["mkt_close"] / m["mkt_close"].shift(1))
        df = df.merge(m[["date", "market_return"]], on="date", how="left")
    else:
        df["market_return"] = np.nan

    # Forward 5-day log return (target; kept separate in sample builder)
    df["target_fwd_5d_log"] = np.log(close.shift(-5) / close)

    return df
