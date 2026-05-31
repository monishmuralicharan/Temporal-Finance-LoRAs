"""Market data loading utilities."""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import yfinance


def load_yfinance_ohlcv(
    ticker: str, start_date: str, end_date: str
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "yfinance is required for Kronos market data loading. "
            "Install it with `pip install -r requirements.txt`."
        ) from exc

    end_inclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    history = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_inclusive.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if history.empty:
        raise RuntimeError(
            "No market history returned for ticker {0}.".format(ticker)
        )
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    history = history.sort_index()
    history.index = pd.to_datetime(history.index)
    if getattr(history.index, "tz", None) is not None:
        history.index = history.index.tz_localize(None)
    required_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing_columns = [
        column for column in required_columns if column not in history.columns
    ]
    if missing_columns:
        raise RuntimeError(
            "Missing OHLCV columns for ticker {0}: {1}. Columns: {2}".format(
                ticker,
                ", ".join(missing_columns),
                ", ".join(str(col) for col in history.columns),
            )
        )
    history = history[required_columns].copy()
    history["Volume"] = pd.to_numeric(history["Volume"], errors="coerce")
    if history.empty:
        raise RuntimeError("No OHLCV values found for ticker {0}.".format(ticker))
    return history


def load_local_kronos_ohlcva(
    ticker: str,
    data_dir: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load a local Kronos-format OHLCVA CSV for one ticker."""
    root = Path(data_dir)
    candidates = [root / f"{ticker}.csv"]
    if "." in ticker:
        candidates.append(root / f"{ticker.replace('.', '-')}.csv")
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise RuntimeError(
            "No local Kronos CSV found for ticker {0} in {1}.".format(
                ticker, data_dir
            )
        )

    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower().lstrip("\ufeff") for column in frame.columns]
    timestamp_column = "timestamps" if "timestamps" in frame.columns else "timestamp"
    required = [timestamp_column, "open", "high", "low", "close", "volume", "amount"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(
            "Missing local Kronos columns for ticker {0}: {1}.".format(
                ticker, ", ".join(missing),
            )
        )

    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column])
    frame = frame.sort_values(timestamp_column).set_index(timestamp_column)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[["open", "high", "low", "close", "volume", "amount"]].dropna()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    frame = frame[(frame.index >= start) & (frame.index <= end)]
    if frame.empty:
        raise RuntimeError(
            "No local Kronos rows for ticker {0} between {1} and {2}.".format(
                ticker, start_date, end_date
            )
        )
    return frame
