"""Kronos checkpoint 4 execution pipeline."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

from temporal_finance.data import load_yfinance_ohlcv
from temporal_finance.evaluation import compute_metrics, compute_simple_return
from temporal_finance.kronos import KronosForecaster, build_kronos_kline_frame
from temporal_finance.kronos_config import KronosCheckpoint4Config
from temporal_finance.kronos_experiments import compute_kronos_prediction_diagnostics
from temporal_finance.windows import EvaluationWindow, build_evaluation_windows


@dataclass
class KronosCheckpoint4RunResult:
    predictions_path: Path
    metrics_path: Path
    per_ticker_metrics_path: Path
    predictions: pd.DataFrame
    metrics: Dict[str, object]
    per_ticker_metrics: pd.DataFrame


def run_kronos_checkpoint4(
    config: KronosCheckpoint4Config,
    forecaster: Optional[object] = None,
    histories: Optional[Dict[str, pd.DataFrame]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> KronosCheckpoint4RunResult:
    config.validate()
    if forecaster is None:
        forecaster = KronosForecaster(config)

    rows = []
    failed_tickers = []
    for ticker in config.effective_tickers:
        try:
            _emit_progress(progress_callback, "Starting ticker {0}".format(ticker))
            history = _resolve_history(config, ticker, histories)
            ticker_rows = _run_ticker_windows(
                config=config,
                ticker=ticker,
                history=history,
                forecaster=forecaster,
                progress_callback=progress_callback,
            )
            rows.extend(ticker_rows)
            _emit_progress(
                progress_callback,
                "Finished ticker {0}: {1} forecast rows".format(
                    ticker,
                    len(ticker_rows),
                ),
            )
        except Exception as exc:
            failed_tickers.append({"ticker": ticker, "error": str(exc)})

    if not rows:
        failure_summary = "; ".join(
            "{0}: {1}".format(item["ticker"], item["error"])
            for item in failed_tickers
        )
        raise RuntimeError("All tickers failed. {0}".format(failure_summary))

    predictions = pd.DataFrame(rows)
    per_ticker_metrics = _compute_per_ticker_metrics(predictions)
    metrics = compute_metrics(predictions)
    metrics.update(compute_kronos_prediction_diagnostics(predictions))
    metrics.update(
        {
            "model": "Kronos",
            "model_id": config.model_id,
            "tokenizer_id": config.tokenizer_id,
            "ticker": config.ticker,
            "tickers": config.effective_tickers,
            "successful_tickers": list(per_ticker_metrics["ticker"]),
            "failed_tickers": failed_tickers,
            "ticker_count": len(config.effective_tickers),
            "successful_ticker_count": int(len(per_ticker_metrics)),
            "failed_ticker_count": int(len(failed_tickers)),
            "target_column": config.target_column,
            "context_length": config.context_length,
            "forecast_horizon": config.forecast_horizon,
            "eval_stride": config.eval_stride,
            "eval_start_date": config.eval_start_date,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "kronos_sample_count": config.sample_count,
            "use_adjusted_ohlc": config.use_adjusted_ohlc,
        }
    )
    predictions_path, metrics_path, per_ticker_metrics_path = _write_outputs(
        output_dir=config.resolved_output_dir,
        predictions=predictions,
        metrics=metrics,
        per_ticker_metrics=per_ticker_metrics,
    )
    return KronosCheckpoint4RunResult(
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        per_ticker_metrics_path=per_ticker_metrics_path,
        predictions=predictions,
        metrics=metrics,
        per_ticker_metrics=per_ticker_metrics,
    )


def _resolve_history(
    config: KronosCheckpoint4Config,
    ticker: str,
    histories: Optional[Dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    if histories is not None:
        if ticker not in histories:
            raise RuntimeError(
                "No provided OHLCV history for ticker {0}.".format(ticker)
            )
        return histories[ticker]
    return load_yfinance_ohlcv(
        ticker=ticker,
        start_date=config.start_date,
        end_date=config.end_date,
    )


def _run_ticker_windows(
    config: KronosCheckpoint4Config,
    ticker: str,
    history: pd.DataFrame,
    forecaster: object,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, object]]:
    kline = build_kronos_kline_frame(
        history=history,
        target_column=config.target_column,
        use_adjusted_ohlc=config.use_adjusted_ohlc,
    )
    windows = build_evaluation_windows(
        series=kline["close"],
        context_length=config.context_length,
        forecast_horizon=config.forecast_horizon,
        eval_start_date=config.eval_start_date,
    )
    windows = windows[:: config.eval_stride]
    date_index = pd.DatetimeIndex(kline.index)

    rows = []
    total_batches = (len(windows) + config.batch_size - 1) // config.batch_size
    for batch_index, batch in enumerate(_chunked(windows, config.batch_size), start=1):
        _emit_progress(
            progress_callback,
            "Ticker {0}: batch {1}/{2} ({3} windows)".format(
                ticker,
                batch_index,
                total_batches,
                len(batch),
            ),
        )
        contexts = []
        x_timestamps = []
        y_timestamps = []
        batch_windows = []
        for window in batch:
            end_idx = date_index.get_loc(window.context_end_date)
            if isinstance(end_idx, slice):
                raise RuntimeError("Duplicate dates are not supported.")
            start_idx = end_idx - config.context_length + 1
            target_stop_idx = end_idx + config.forecast_horizon + 1
            if start_idx < 0 or target_stop_idx > len(date_index):
                continue
            context_index = date_index[start_idx : end_idx + 1]
            target_index = date_index[end_idx + 1 : target_stop_idx]
            contexts.append(kline.loc[context_index])
            x_timestamps.append(pd.Series(context_index))
            y_timestamps.append(pd.Series(target_index))
            batch_windows.append(window)

        if not batch_windows:
            continue

        forecasts = forecaster.forecast_batch(contexts, x_timestamps, y_timestamps)
        for window, forecast in zip(batch_windows, forecasts):
            pred_close_t5 = float(forecast["close"].iloc[config.forecast_horizon - 1])
            rows.append(
                {
                    "ticker": ticker,
                    "date": window.context_end_date.strftime("%Y-%m-%d"),
                    "target_date": window.target_date.strftime("%Y-%m-%d"),
                    "last_context_close": window.last_context_close,
                    "pred_close_t5": pred_close_t5,
                    "actual_close_t5": window.actual_close_t5,
                    "pred_5d_return": compute_simple_return(
                        window.last_context_close, pred_close_t5
                    ),
                    "actual_5d_return": compute_simple_return(
                        window.last_context_close, window.actual_close_t5
                    ),
                    "pred_open_t5": float(
                        forecast["open"].iloc[config.forecast_horizon - 1]
                    ),
                    "pred_high_t5": float(
                        forecast["high"].iloc[config.forecast_horizon - 1]
                    ),
                    "pred_low_t5": float(
                        forecast["low"].iloc[config.forecast_horizon - 1]
                    ),
                    "pred_volume_t5": float(
                        forecast["volume"].iloc[config.forecast_horizon - 1]
                    ),
                }
            )
    return rows


def _compute_per_ticker_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, ticker_predictions in predictions.groupby("ticker", sort=False):
        metrics = compute_metrics(ticker_predictions)
        metrics.update(compute_kronos_prediction_diagnostics(ticker_predictions))
        metrics["ticker"] = ticker
        rows.append(metrics)
    columns = [
        "ticker",
        "sample_count",
        "mse",
        "mae",
        "directional_accuracy",
        "pearson_ic",
        "spearman_rank_ic",
        "pred_abs_return_mean",
        "actual_abs_return_mean",
        "pred_abs_return_multiple",
        "pred_positive_rate",
        "actual_positive_rate",
        "positive_rate_gap",
        "pred_close_mae",
        "pred_close_mape",
        "pred_actual_close_ratio_mean",
        "pred_actual_close_ratio_std",
        "invalid_pred_ohlc_count",
        "negative_pred_volume_count",
    ]
    return pd.DataFrame(rows, columns=columns)


def _write_outputs(
    output_dir: Path,
    predictions: pd.DataFrame,
    metrics: Dict[str, object],
    per_ticker_metrics: pd.DataFrame,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    per_ticker_metrics_path = output_dir / "per_ticker_metrics.csv"
    predictions.to_csv(predictions_path, index=False)
    per_ticker_metrics.to_csv(per_ticker_metrics_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return predictions_path, metrics_path, per_ticker_metrics_path


def _emit_progress(
    progress_callback: Optional[Callable[[str], None]],
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _chunked(
    items: List[EvaluationWindow], size: int
) -> Iterable[List[EvaluationWindow]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
