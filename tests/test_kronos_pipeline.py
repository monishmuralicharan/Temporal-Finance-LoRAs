import numpy as np
import pandas as pd
import pytest

from temporal_finance.kronos_config import KronosCheckpoint4Config
from temporal_finance.kronos_pipeline import run_kronos_checkpoint4


class DummyKronosForecaster:
    def forecast_batch(self, contexts, x_timestamps, y_timestamps):
        forecasts = []
        for context, y_timestamp in zip(contexts, y_timestamps):
            last_close = float(context["close"].iloc[-1])
            pred_len = len(y_timestamp)
            forecasts.append(
                pd.DataFrame(
                    {
                        "open": [last_close] * pred_len,
                        "high": [last_close] * pred_len,
                        "low": [last_close] * pred_len,
                        "close": [last_close] * pred_len,
                        "volume": [1.0] * pred_len,
                        "amount": [last_close] * pred_len,
                    },
                    index=pd.to_datetime(y_timestamp),
                )
            )
        return forecasts


def test_run_kronos_checkpoint4_writes_outputs(tmp_path):
    config = _make_config(tmp_path)
    histories = {
        "AAPL": _make_history(100.0),
        "MSFT": _make_history(200.0),
    }

    result = run_kronos_checkpoint4(
        config=config,
        forecaster=DummyKronosForecaster(),
        histories=histories,
    )

    assert result.predictions_path.exists()
    assert result.metrics_path.exists()
    assert result.per_ticker_metrics_path.exists()
    assert result.metrics["model"] == "Kronos"
    assert result.metrics["model_id"] == "NeoQuasar/Kronos-base"
    assert result.metrics["successful_tickers"] == ["AAPL", "MSFT"]
    assert "pred_abs_return_multiple" in result.metrics
    assert "pred_close_mape" in result.metrics
    assert set(result.predictions["ticker"]) == {"AAPL", "MSFT"}
    assert set(["pred_open_t5", "pred_volume_t5"]).issubset(
        result.predictions.columns
    )


def test_run_kronos_checkpoint4_emits_progress(tmp_path):
    config = _make_config(tmp_path)
    messages = []

    run_kronos_checkpoint4(
        config=config,
        forecaster=DummyKronosForecaster(),
        histories={"AAPL": _make_history(100.0), "MSFT": _make_history(200.0)},
        progress_callback=messages.append,
    )

    assert any(message == "Starting ticker AAPL" for message in messages)
    assert any("Ticker AAPL: batch" in message for message in messages)
    assert any("Finished ticker MSFT" in message for message in messages)


def test_run_kronos_checkpoint4_raises_when_all_tickers_fail(tmp_path):
    config = _make_config(tmp_path)

    with pytest.raises(RuntimeError, match="All tickers failed"):
        run_kronos_checkpoint4(
            config=config,
            forecaster=DummyKronosForecaster(),
            histories={},
        )


def _make_config(tmp_path):
    return KronosCheckpoint4Config(
        ticker="AAPL",
        tickers=["AAPL", "MSFT"],
        start_date="2020-01-01",
        eval_start_date="2020-04-15",
        end_date="2020-06-30",
        context_length=32,
        forecast_horizon=5,
        target_column="Adj Close",
        output_dir=str(tmp_path / "outputs"),
        batch_size=3,
    )


def _make_history(start_price):
    periods = 130
    dates = pd.bdate_range("2020-01-01", periods=periods)
    step = np.arange(periods, dtype=float)
    close = start_price + step + np.sin(step / 5.0)
    volume = 1_000_000.0 + (step * 1_000.0)
    return pd.DataFrame(
        {
            "Open": close - 0.25,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        },
        index=dates,
    )
