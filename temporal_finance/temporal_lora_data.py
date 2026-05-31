"""Data preparation for the Temporal LoRA signal run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from temporal_finance.data import load_local_kronos_ohlcva
from temporal_finance.kronos import KRONOS_COLUMNS
from temporal_finance.temporal_lora_config import (
    TemporalLoraSignalConfig,
    TemporalLoraWindowConfig,
)


@dataclass(frozen=True)
class TemporalLoraExample:
    ticker: str
    window_id: str
    split: str
    adapter_name: str
    context_start_index: int
    context_end_index: int
    target_index: int
    context_start_date: str
    context_end_date: str
    target_date: str

    def to_row(self) -> Dict[str, object]:
        return {
            "ticker": self.ticker,
            "window_id": self.window_id,
            "split": self.split,
            "adapter_name": self.adapter_name,
            "context_start_index": self.context_start_index,
            "context_end_index": self.context_end_index,
            "target_index": self.target_index,
            "context_start_date": self.context_start_date,
            "context_end_date": self.context_end_date,
            "target_date": self.target_date,
        }


@dataclass(frozen=True)
class TemporalLoraWindowPlan:
    window_id: str
    slow_train_start: str
    slow_train_end: str
    medium_train_start: str
    medium_train_end: str
    fast_train_start: str
    fast_train_end: str
    validation_start: str
    validation_end: str

    def to_row(self) -> Dict[str, object]:
        return {
            "window_id": self.window_id,
            "slow_train_start": self.slow_train_start,
            "slow_train_end": self.slow_train_end,
            "medium_train_start": self.medium_train_start,
            "medium_train_end": self.medium_train_end,
            "fast_train_start": self.fast_train_start,
            "fast_train_end": self.fast_train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
        }


def resolve_tickers(config: TemporalLoraSignalConfig) -> List[str]:
    """Resolve the ordered ticker universe for the signal run."""
    if config.tickers:
        tickers = [str(ticker).strip() for ticker in config.tickers if str(ticker).strip()]
    elif config.universe_file:
        path = Path(config.universe_file)
        tickers = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        tickers = sorted(path.stem for path in Path(config.data_dir).glob("*.csv"))
    return tickers[: config.ticker_limit]


def load_signal_histories(
    config: TemporalLoraSignalConfig,
    tickers: Optional[Iterable[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load Kronos-format OHLCVA histories for all selected tickers."""
    selected = list(tickers or resolve_tickers(config))
    if not selected:
        raise RuntimeError("No Temporal LoRA tickers resolved.")
    start_date = min(window.slow_train_start for window in config.windows)
    end_date = max(window.validation_end for window in config.windows)
    histories = {}
    failed = []
    for ticker in selected:
        try:
            histories[ticker] = load_local_kronos_ohlcva(
                ticker=ticker,
                data_dir=config.data_dir,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:
            failed.append({"ticker": ticker, "error": str(exc)})
    if not histories:
        summary = "; ".join(
            "{0}: {1}".format(item["ticker"], item["error"]) for item in failed
        )
        raise RuntimeError("No Temporal LoRA histories loaded. {0}".format(summary))
    return histories


def build_window_plans(
    config: TemporalLoraSignalConfig,
    histories: Dict[str, pd.DataFrame],
) -> List[TemporalLoraWindowPlan]:
    """Build explicit slow/medium/fast train windows without forward leakage."""
    plans = []
    for window in config.windows:
        medium_start = _trailing_start_date(
            histories,
            end_date=window.slow_train_end,
            trading_days=window.medium_trading_days,
        )
        fast_start = _trailing_start_date(
            histories,
            end_date=window.slow_train_end,
            trading_days=window.fast_trading_days,
        )
        plans.append(
            TemporalLoraWindowPlan(
                window_id=window.window_id,
                slow_train_start=window.slow_train_start,
                slow_train_end=window.slow_train_end,
                medium_train_start=medium_start,
                medium_train_end=window.slow_train_end,
                fast_train_start=fast_start,
                fast_train_end=window.slow_train_end,
                validation_start=window.validation_start,
                validation_end=window.validation_end,
            )
        )
    return plans


def build_training_examples_for_window(
    histories: Dict[str, pd.DataFrame],
    plan: TemporalLoraWindowPlan,
    adapter_name: str,
    context_length: int,
    stride: int = 1,
    max_examples: Optional[int] = None,
) -> List[TemporalLoraExample]:
    if adapter_name == "slow":
        start_date = plan.slow_train_start
        end_date = plan.slow_train_end
    elif adapter_name == "medium":
        start_date = plan.medium_train_start
        end_date = plan.medium_train_end
    elif adapter_name == "fast":
        start_date = plan.fast_train_start
        end_date = plan.fast_train_end
    else:
        raise ValueError("Unknown adapter_name: {0}".format(adapter_name))
    examples = _build_examples(
        histories=histories,
        window_id=plan.window_id,
        split="train",
        adapter_name=adapter_name,
        start_date=start_date,
        end_date=end_date,
        context_length=context_length,
        stride=stride,
    )
    if max_examples is not None:
        examples = _cap_examples_evenly(examples, max_examples)
    return examples


def build_validation_examples_for_window(
    histories: Dict[str, pd.DataFrame],
    plan: TemporalLoraWindowPlan,
    context_length: int,
    stride: int = 1,
) -> List[TemporalLoraExample]:
    return _build_examples(
        histories=histories,
        window_id=plan.window_id,
        split="validation",
        adapter_name="validation",
        start_date=plan.validation_start,
        end_date=plan.validation_end,
        context_length=context_length,
        stride=stride,
    )


def temporal_lora_examples_to_frame(
    examples: Iterable[TemporalLoraExample],
) -> pd.DataFrame:
    return pd.DataFrame([example.to_row() for example in examples])


def temporal_lora_window_plans_to_frame(
    plans: Iterable[TemporalLoraWindowPlan],
) -> pd.DataFrame:
    return pd.DataFrame([plan.to_row() for plan in plans])


def summarize_examples(
    examples: Iterable[TemporalLoraExample],
) -> pd.DataFrame:
    frame = temporal_lora_examples_to_frame(examples)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "window_id",
                "split",
                "adapter_name",
                "ticker_count",
                "example_count",
                "first_target_date",
                "last_target_date",
            ]
        )
    grouped = frame.groupby(["window_id", "split", "adapter_name"], sort=True)
    return grouped.agg(
        ticker_count=("ticker", "nunique"),
        example_count=("ticker", "size"),
        first_target_date=("target_date", "min"),
        last_target_date=("target_date", "max"),
    ).reset_index()


def build_normalized_sequence(
    history: pd.DataFrame,
    example: TemporalLoraExample,
    clip: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return context-normalized K-line values and stats for context+target rows."""
    full = history.iloc[
        example.context_start_index : example.target_index + 1
    ][KRONOS_COLUMNS]
    context = full.iloc[: example.context_end_index - example.context_start_index + 1]
    values = full.to_numpy(dtype=np.float32)
    context_values = context.to_numpy(dtype=np.float32)
    mean = np.mean(context_values, axis=0).astype(np.float32)
    std = np.std(context_values, axis=0).astype(np.float32)
    normalized = (values - mean) / (std + 1e-5)
    normalized = np.clip(normalized, -clip, clip).astype(np.float32)
    return normalized, mean, std, values


def time_features(index: pd.Index) -> np.ndarray:
    stamps = pd.to_datetime(pd.Series(index))
    frame = pd.DataFrame(
        {
            "minute": stamps.dt.minute,
            "hour": stamps.dt.hour,
            "weekday": stamps.dt.weekday,
            "day": stamps.dt.day,
            "month": stamps.dt.month,
        }
    )
    return frame.to_numpy(dtype=np.float32)


def build_prediction_row_from_forecast(
    example: TemporalLoraExample,
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    variant: str,
) -> Dict[str, object]:
    context_end_row = history.iloc[example.context_end_index]
    target_row = history.iloc[example.target_index]
    last_context_close = float(context_end_row["close"])
    actual_close = float(target_row["close"])
    pred_row = forecast.iloc[-1]
    pred_close = float(pred_row["close"])
    pred_return = _simple_return(last_context_close, pred_close)
    actual_return = _simple_return(last_context_close, actual_close)
    return {
        "ticker": example.ticker,
        "window_id": example.window_id,
        "variant": variant,
        "date": example.context_end_date,
        "target_date": example.target_date,
        "forecast_horizon": 1,
        "last_context_close": last_context_close,
        "pred_close_target": pred_close,
        "actual_close_target": actual_close,
        "pred_return": pred_return,
        "actual_return": actual_return,
        "pred_open_target": float(pred_row["open"]),
        "pred_high_target": float(pred_row["high"]),
        "pred_low_target": float(pred_row["low"]),
        "pred_volume_target": float(pred_row["volume"]),
        "pred_close_t5": pred_close,
        "actual_close_t5": actual_close,
        "pred_5d_return": pred_return,
        "actual_5d_return": actual_return,
        "pred_open_t5": float(pred_row["open"]),
        "pred_high_t5": float(pred_row["high"]),
        "pred_low_t5": float(pred_row["low"]),
        "pred_volume_t5": float(pred_row["volume"]),
    }


def build_naive_predictions(
    examples: Iterable[TemporalLoraExample],
    histories: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    rows_by_variant = {
        "naive_zero_return": [],
        "naive_last_return": [],
        "naive_trailing_20_mean": [],
    }
    for example in examples:
        history = histories[example.ticker]
        context = history.iloc[example.context_start_index : example.context_end_index + 1]
        target_row = history.iloc[example.target_index]
        last_close = float(context["close"].iloc[-1])
        actual_close = float(target_row["close"])
        actual_return = _simple_return(last_close, actual_close)
        returns = context["close"].pct_change().dropna()
        predictions = {
            "naive_zero_return": 0.0,
            "naive_last_return": float(returns.iloc[-1]) if not returns.empty else 0.0,
            "naive_trailing_20_mean": (
                float(returns.tail(20).mean()) if not returns.empty else 0.0
            ),
        }
        for variant, pred_return in predictions.items():
            pred_close = last_close * (1.0 + pred_return)
            rows_by_variant[variant].append(
                _prediction_row_from_values(
                    example=example,
                    variant=variant,
                    last_context_close=last_close,
                    actual_close=actual_close,
                    actual_return=actual_return,
                    pred_return=pred_return,
                    pred_open=float(context["open"].iloc[-1]) * (1.0 + pred_return),
                    pred_high=max(
                        pred_close,
                        float(context["high"].iloc[-1]) * (1.0 + pred_return),
                    ),
                    pred_low=min(
                        pred_close,
                        float(context["low"].iloc[-1]) * (1.0 + pred_return),
                    ),
                    pred_close=pred_close,
                    pred_volume=float(context["volume"].iloc[-1]),
                )
            )
    return {
        variant: pd.DataFrame(rows)
        for variant, rows in rows_by_variant.items()
    }


def _build_examples(
    histories: Dict[str, pd.DataFrame],
    window_id: str,
    split: str,
    adapter_name: str,
    start_date: str,
    end_date: str,
    context_length: int,
    stride: int = 1,
) -> List[TemporalLoraExample]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    rows = []
    for ticker, history in histories.items():
        history = history.sort_index()
        date_index = pd.DatetimeIndex(history.index)
        if len(date_index) <= context_length:
            continue
        candidate_indices = [
            index
            for index, target_date in enumerate(date_index)
            if index >= context_length and start <= target_date <= end
        ]
        for offset, target_index in enumerate(candidate_indices):
            if offset % stride != 0:
                continue
            context_end_index = target_index - 1
            context_start_index = context_end_index - context_length + 1
            if context_start_index < 0:
                continue
            rows.append(
                TemporalLoraExample(
                    ticker=ticker,
                    window_id=window_id,
                    split=split,
                    adapter_name=adapter_name,
                    context_start_index=context_start_index,
                    context_end_index=context_end_index,
                    target_index=target_index,
                    context_start_date=_date_string(date_index[context_start_index]),
                    context_end_date=_date_string(date_index[context_end_index]),
                    target_date=_date_string(date_index[target_index]),
                )
            )
    rows.sort(key=lambda item: (item.target_date, item.ticker))
    return rows


def _cap_examples_evenly(
    examples: List[TemporalLoraExample],
    max_examples: int,
) -> List[TemporalLoraExample]:
    if len(examples) <= max_examples:
        return examples
    if max_examples == 1:
        return [examples[-1]]
    indices = np.linspace(0, len(examples) - 1, num=max_examples, dtype=int)
    return [examples[int(index)] for index in indices]


def _trailing_start_date(
    histories: Dict[str, pd.DataFrame],
    end_date: str,
    trading_days: int,
) -> str:
    end = pd.Timestamp(end_date)
    dates = sorted(
        {
            pd.Timestamp(date).normalize()
            for history in histories.values()
            for date in history.index
            if pd.Timestamp(date).normalize() <= end
        }
    )
    if not dates:
        return pd.Timestamp(end_date).strftime("%Y-%m-%d")
    start_index = max(0, len(dates) - trading_days)
    return dates[start_index].strftime("%Y-%m-%d")


def _date_string(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _simple_return(start_value: float, end_value: float) -> float:
    if start_value == 0:
        return 0.0
    return float((end_value / start_value) - 1.0)


def _prediction_row_from_values(
    example: TemporalLoraExample,
    variant: str,
    last_context_close: float,
    actual_close: float,
    actual_return: float,
    pred_return: float,
    pred_open: float,
    pred_high: float,
    pred_low: float,
    pred_close: float,
    pred_volume: float,
) -> Dict[str, object]:
    return {
        "ticker": example.ticker,
        "window_id": example.window_id,
        "variant": variant,
        "date": example.context_end_date,
        "target_date": example.target_date,
        "forecast_horizon": 1,
        "last_context_close": last_context_close,
        "pred_close_target": pred_close,
        "actual_close_target": actual_close,
        "pred_return": pred_return,
        "actual_return": actual_return,
        "pred_open_target": pred_open,
        "pred_high_target": pred_high,
        "pred_low_target": pred_low,
        "pred_volume_target": pred_volume,
        "pred_close_t5": pred_close,
        "actual_close_t5": actual_close,
        "pred_5d_return": pred_return,
        "actual_5d_return": actual_return,
        "pred_open_t5": pred_open,
        "pred_high_t5": pred_high,
        "pred_low_t5": pred_low,
        "pred_volume_t5": pred_volume,
    }
