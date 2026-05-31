"""Configuration helpers for the Kronos checkpoint 4 baseline."""

from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class KronosConfigurationError(ValueError):
    """Raised when the Kronos config is invalid."""


@dataclass
class KronosCheckpoint4Config:
    ticker: str
    start_date: str
    eval_start_date: str
    end_date: str
    context_length: int
    forecast_horizon: int
    target_column: str
    output_dir: str
    batch_size: int = 32
    eval_stride: int = 1
    model_id: str = "NeoQuasar/Kronos-base"
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base"
    kronos_cache_dir: Optional[str] = None
    device: str = "auto"
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9
    sample_count: int = 1
    clip: float = 5.0
    random_seed: int = 42
    use_adjusted_ohlc: bool = True
    tickers: Optional[List[str]] = None
    data_source: str = "yfinance"
    local_data_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "KronosCheckpoint4Config":
        defaults = {
            "ticker": "",
            "batch_size": 32,
            "eval_stride": 1,
            "model_id": "NeoQuasar/Kronos-base",
            "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
            "kronos_cache_dir": None,
            "device": "auto",
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 0.9,
            "sample_count": 1,
            "clip": 5.0,
            "random_seed": 42,
            "use_adjusted_ohlc": True,
            "tickers": None,
            "data_source": "yfinance",
            "local_data_dir": None,
        }
        expected = {
            "ticker",
            "tickers",
            "start_date",
            "eval_start_date",
            "end_date",
            "context_length",
            "forecast_horizon",
            "target_column",
            "output_dir",
            "batch_size",
            "eval_stride",
            "model_id",
            "tokenizer_id",
            "kronos_cache_dir",
            "device",
            "temperature",
            "top_k",
            "top_p",
            "sample_count",
            "clip",
            "random_seed",
            "use_adjusted_ohlc",
            "data_source",
            "local_data_dir",
        }
        unknown = sorted(set(raw) - expected)
        if unknown:
            raise KronosConfigurationError(
                "Unknown config keys: {0}".format(", ".join(unknown))
            )
        required = {
            "start_date",
            "eval_start_date",
            "end_date",
            "context_length",
            "forecast_horizon",
            "target_column",
            "output_dir",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise KronosConfigurationError(
                "Missing config keys: {0}".format(", ".join(missing))
            )
        payload = defaults.copy()
        payload.update(raw)
        for key in ("start_date", "eval_start_date", "end_date"):
            payload[key] = _coerce_datetime(payload[key]).strftime("%Y-%m-%d")
        return cls(**payload)

    def validate(self) -> None:
        self._validate_dates()
        self._validate_tickers()
        if self.context_length <= 0:
            raise KronosConfigurationError("context_length must be positive.")
        if self.context_length > 512:
            raise KronosConfigurationError(
                "context_length must be <= 512 for Kronos-base/small."
            )
        if self.forecast_horizon <= 0:
            raise KronosConfigurationError("forecast_horizon must be positive.")
        if self.batch_size <= 0:
            raise KronosConfigurationError("batch_size must be positive.")
        if self.eval_stride <= 0:
            raise KronosConfigurationError("eval_stride must be positive.")
        if not self.target_column:
            raise KronosConfigurationError("target_column must be non-empty.")
        if not self.output_dir:
            raise KronosConfigurationError("output_dir must be non-empty.")
        if not self.model_id:
            raise KronosConfigurationError("model_id must be non-empty.")
        if not self.tokenizer_id:
            raise KronosConfigurationError("tokenizer_id must be non-empty.")
        if self.device not in ("auto", "cpu", "cuda", "gpu", "mps"):
            raise KronosConfigurationError(
                "device must be one of: auto, cpu, cuda, gpu, mps."
            )
        if self.temperature <= 0:
            raise KronosConfigurationError("temperature must be positive.")
        if self.top_k < 0:
            raise KronosConfigurationError("top_k must be non-negative.")
        if not 0 < self.top_p <= 1:
            raise KronosConfigurationError("top_p must be in (0, 1].")
        if self.sample_count <= 0:
            raise KronosConfigurationError("sample_count must be positive.")
        if self.clip <= 0:
            raise KronosConfigurationError("clip must be positive.")
        if self.data_source not in ("yfinance", "local_kronos_csv"):
            raise KronosConfigurationError(
                "data_source must be one of: yfinance, local_kronos_csv."
            )
        if self.data_source == "local_kronos_csv" and not self.local_data_dir:
            raise KronosConfigurationError(
                "local_data_dir is required for local_kronos_csv data_source."
            )

    @property
    def effective_tickers(self) -> List[str]:
        if self.tickers is None:
            return [str(self.ticker).strip()]
        return [str(ticker).strip() for ticker in self.tickers]

    @property
    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir).resolve()

    @property
    def start_timestamp(self):
        return _coerce_datetime(self.start_date)

    @property
    def eval_start_timestamp(self):
        return _coerce_datetime(self.eval_start_date)

    @property
    def end_timestamp(self):
        return _coerce_datetime(self.end_date)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in ("start_date", "eval_start_date", "end_date"):
            payload[key] = _coerce_datetime(payload[key]).strftime("%Y-%m-%d")
        return payload

    def _validate_dates(self) -> None:
        start = self.start_timestamp
        eval_start = self.eval_start_timestamp
        end = self.end_timestamp
        if start >= end:
            raise KronosConfigurationError("start_date must be before end_date.")
        if eval_start < start:
            raise KronosConfigurationError(
                "eval_start_date must be on or after start_date."
            )
        if eval_start >= end:
            raise KronosConfigurationError(
                "eval_start_date must be before end_date."
            )

    def _validate_tickers(self) -> None:
        if self.tickers is None:
            if not isinstance(self.ticker, str) or not self.ticker.strip():
                raise KronosConfigurationError("ticker must be non-empty.")
            return

        if not isinstance(self.tickers, list) or not self.tickers:
            raise KronosConfigurationError(
                "tickers must contain at least one ticker."
            )
        normalized = []
        for ticker in self.tickers:
            if not isinstance(ticker, str) or not ticker.strip():
                raise KronosConfigurationError(
                    "tickers must contain only non-empty strings."
                )
            normalized.append(ticker.strip())
        if len(set(normalized)) != len(normalized):
            raise KronosConfigurationError("tickers must be unique.")


def load_kronos_checkpoint4_config(
    config_path: str, validate: bool = True
) -> KronosCheckpoint4Config:
    path = Path(config_path)
    if not path.exists():
        raise KronosConfigurationError(
            "Config file does not exist: {0}".format(path)
        )
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise KronosConfigurationError("Config file must contain a YAML mapping.")
    config = KronosCheckpoint4Config.from_dict(raw)
    if validate:
        config.validate()
    return config


def _coerce_datetime(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    return datetime.strptime(str(value), "%Y-%m-%d")
