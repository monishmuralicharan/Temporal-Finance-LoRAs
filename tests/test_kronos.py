import numpy as np
import pandas as pd
import pytest

from temporal_finance.kronos import build_kronos_kline_frame


def test_build_kronos_kline_frame_adjusts_ohlc_to_adj_close():
    history = pd.DataFrame(
        {
            "Open": [100.0, 102.0],
            "High": [110.0, 112.0],
            "Low": [90.0, 92.0],
            "Close": [100.0, 100.0],
            "Adj Close": [50.0, 80.0],
            "Volume": [1000.0, 2000.0],
        },
        index=pd.bdate_range("2020-01-01", periods=2),
    )

    frame = build_kronos_kline_frame(history)

    assert list(frame.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert frame["open"].iloc[0] == pytest.approx(50.0)
    assert frame["high"].iloc[1] == pytest.approx(89.6)
    assert frame["close"].iloc[1] == pytest.approx(80.0)
    assert frame["amount"].iloc[0] == pytest.approx(
        np.mean([50.0, 55.0, 45.0, 50.0]) * 1000.0
    )


def test_build_kronos_kline_frame_accepts_native_ohlcva():
    history = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [12.0, 13.0],
            "low": [9.0, 10.0],
            "close": [11.0, 12.0],
            "volume": [1000.0, 1100.0],
            "amount": [11000.0, 13200.0],
        },
        index=pd.bdate_range("2025-01-01", periods=2),
    )

    frame = build_kronos_kline_frame(history, target_column="close")

    assert list(frame.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert frame["amount"].iloc[1] == pytest.approx(13200.0)
