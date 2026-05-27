"""Kronos forecasting wrapper and OHLCV preparation."""

from typing import List

import numpy as np
import pandas as pd

from temporal_finance.kronos_config import KronosCheckpoint4Config


KRONOS_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


class KronosForecaster:
    """Thin wrapper around the upstream Kronos predictor."""

    def __init__(self, config: KronosCheckpoint4Config):
        self._config = config
        self._predictor = self._load_predictor()

    def forecast_batch(
        self,
        contexts: List[pd.DataFrame],
        x_timestamps: List[pd.Series],
        y_timestamps: List[pd.Series],
    ) -> List[pd.DataFrame]:
        if not contexts:
            return []
        return self._predictor.predict_batch(
            df_list=contexts,
            x_timestamp_list=x_timestamps,
            y_timestamp_list=y_timestamps,
            pred_len=self._config.forecast_horizon,
            T=self._config.temperature,
            top_k=self._config.top_k,
            top_p=self._config.top_p,
            sample_count=self._config.sample_count,
            verbose=False,
        )

    def _load_predictor(self):
        try:
            import torch
            from model import Kronos, KronosPredictor, KronosTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Kronos is not importable. Add the upstream Kronos repo to "
                "PYTHONPATH or use the Modal runner."
            ) from exc

        torch.manual_seed(self._config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._config.random_seed)

        kwargs = {}
        if self._config.kronos_cache_dir:
            kwargs["cache_dir"] = self._config.kronos_cache_dir
        tokenizer = KronosTokenizer.from_pretrained(
            self._config.tokenizer_id, **kwargs
        )
        model = Kronos.from_pretrained(self._config.model_id, **kwargs)
        device = _resolve_device(self._config.device, torch)
        return KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_context=self._config.context_length,
            clip=self._config.clip,
        )


def build_kronos_kline_frame(
    history: pd.DataFrame,
    target_column: str = "Adj Close",
    use_adjusted_ohlc: bool = True,
) -> pd.DataFrame:
    history = history.copy().sort_index()
    history.index = pd.to_datetime(history.index)
    if getattr(history.index, "tz", None) is not None:
        history.index = history.index.tz_localize(None)

    required = ["Open", "High", "Low", "Close", "Volume", target_column]
    missing = [column for column in required if column not in history.columns]
    if missing:
        raise RuntimeError(
            "Missing OHLCV columns for Kronos: {0}".format(", ".join(missing))
        )

    for column in required:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history = history.dropna(subset=required)
    if history.empty:
        raise RuntimeError("No usable OHLCV rows after Kronos preprocessing.")

    if use_adjusted_ohlc:
        adjustment = (history[target_column] / history["Close"]).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        adjustment = pd.Series(1.0, index=history.index)
    adjusted = pd.DataFrame(index=history.index)
    adjusted["open"] = history["Open"] * adjustment
    adjusted["high"] = history["High"] * adjustment
    adjusted["low"] = history["Low"] * adjustment
    if use_adjusted_ohlc:
        adjusted["close"] = history[target_column]
    else:
        adjusted["close"] = history["Close"]
    adjusted["volume"] = history["Volume"]
    adjusted["amount"] = (
        adjusted["volume"]
        * adjusted[["open", "high", "low", "close"]].mean(axis=1)
    )
    adjusted = adjusted.replace([np.inf, -np.inf], np.nan).dropna()
    if adjusted.empty:
        raise RuntimeError("No usable Kronos K-line rows after adjustment.")
    return adjusted[KRONOS_COLUMNS]


def _resolve_device(device: str, torch_module) -> str:
    if device == "gpu":
        device = "cuda"
    if device == "auto":
        if torch_module.cuda.is_available():
            return "cuda:0"
        if (
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"
    if device == "cuda":
        return "cuda:0"
    return device
