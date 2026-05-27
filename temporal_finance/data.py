"""Market data loading utilities."""

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
