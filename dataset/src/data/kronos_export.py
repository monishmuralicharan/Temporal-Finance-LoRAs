"""Convert Massive daily bars to KronosPredictor-ready CSV layout."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Massive symbol → file name (some symbols need aliases on the API)
TICKER_API_ALIASES: dict[str, str] = {
    "BRK.B": "BRK.B",
}

KRONOS_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
KRONOS_COMBINED_COLUMNS = ["ticker", "x_timestamp", "open", "high", "low", "close", "volume", "amount"]


def bars_to_kronos_df(bars: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    Build a DataFrame matching KronosPredictor expectations.

    Required: open, high, low, close
    Optional: volume, amount (we set amount = close * volume as dollar turnover proxy)
    """
    df = bars.sort_values("date").copy()
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    if df.empty:
        return pd.DataFrame(columns=KRONOS_COLUMNS)

    out = pd.DataFrame(
        {
            "timestamps": pd.to_datetime(df["date"]),
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "volume": df["volume"].astype(float),
            "amount": (df["close"].astype(float) * df["volume"].astype(float)),
        }
    )
    return out.reset_index(drop=True)


def bars_to_kronos_combined_row(
    bars: pd.DataFrame,
    ticker_label: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Same as bars_to_kronos_df but with leading ticker column for combined CSV."""
    kdf = bars_to_kronos_df(bars, start=start, end=end)
    if kdf.empty:
        return pd.DataFrame(columns=KRONOS_COMBINED_COLUMNS)
    kdf = kdf.rename(columns={"timestamps": "x_timestamp"})
    kdf.insert(0, "ticker", ticker_label)
    return kdf


def export_ticker_csv(
    bars: pd.DataFrame,
    ticker: str,
    out_path: Path,
    start: str,
    end: str,
) -> int:
    kronos_df = bars_to_kronos_df(bars, start=start, end=end)
    if kronos_df.empty:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kronos_df.to_csv(out_path, index=False)
    return len(kronos_df)


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
