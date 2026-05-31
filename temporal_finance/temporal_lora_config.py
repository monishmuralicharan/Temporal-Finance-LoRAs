"""Configuration helpers for the Temporal LoRA signal run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class TemporalLoraConfigurationError(ValueError):
    """Raised when the Temporal LoRA signal-run config is invalid."""


@dataclass(frozen=True)
class TemporalLoraAdapterConfig:
    name: str
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 3
    batch_size: int = 32
    max_examples: Optional[int] = None
    train_stride: int = 1
    patience: int = 1

    @classmethod
    def from_dict(
        cls,
        name: str,
        raw: Optional[Dict[str, Any]],
    ) -> "TemporalLoraAdapterConfig":
        payload = {"name": name}
        if raw:
            payload.update(raw)
        return cls(**payload)

    def validate(self) -> None:
        if not self.name:
            raise TemporalLoraConfigurationError("adapter name must be non-empty.")
        if self.rank <= 0:
            raise TemporalLoraConfigurationError(
                "adapter {0} rank must be positive.".format(self.name)
            )
        if self.alpha <= 0:
            raise TemporalLoraConfigurationError(
                "adapter {0} alpha must be positive.".format(self.name)
            )
        if not 0 <= self.dropout < 1:
            raise TemporalLoraConfigurationError(
                "adapter {0} dropout must be in [0, 1).".format(self.name)
            )
        if self.learning_rate <= 0:
            raise TemporalLoraConfigurationError(
                "adapter {0} learning_rate must be positive.".format(self.name)
            )
        if self.weight_decay < 0:
            raise TemporalLoraConfigurationError(
                "adapter {0} weight_decay must be non-negative.".format(self.name)
            )
        if self.epochs <= 0:
            raise TemporalLoraConfigurationError(
                "adapter {0} epochs must be positive.".format(self.name)
            )
        if self.batch_size <= 0:
            raise TemporalLoraConfigurationError(
                "adapter {0} batch_size must be positive.".format(self.name)
            )
        if self.max_examples is not None and self.max_examples <= 0:
            raise TemporalLoraConfigurationError(
                "adapter {0} max_examples must be positive when set.".format(
                    self.name
                )
            )
        if self.train_stride <= 0:
            raise TemporalLoraConfigurationError(
                "adapter {0} train_stride must be positive.".format(self.name)
            )
        if self.patience < 0:
            raise TemporalLoraConfigurationError(
                "adapter {0} patience must be non-negative.".format(self.name)
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalLoraWindowConfig:
    window_id: str
    slow_train_start: str
    slow_train_end: str
    validation_start: str
    validation_end: str
    medium_trading_days: int = 63
    fast_trading_days: int = 20

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TemporalLoraWindowConfig":
        payload = dict(raw)
        for key in (
            "slow_train_start",
            "slow_train_end",
            "validation_start",
            "validation_end",
        ):
            payload[key] = _coerce_datetime(payload[key]).strftime("%Y-%m-%d")
        return cls(**payload)

    def validate(self) -> None:
        if not self.window_id:
            raise TemporalLoraConfigurationError("window_id must be non-empty.")
        slow_start = _coerce_datetime(self.slow_train_start)
        slow_end = _coerce_datetime(self.slow_train_end)
        validation_start = _coerce_datetime(self.validation_start)
        validation_end = _coerce_datetime(self.validation_end)
        if slow_start > slow_end:
            raise TemporalLoraConfigurationError(
                "{0}: slow_train_start must be on or before slow_train_end.".format(
                    self.window_id
                )
            )
        if slow_end >= validation_start:
            raise TemporalLoraConfigurationError(
                "{0}: slow_train_end must be before validation_start.".format(
                    self.window_id
                )
            )
        if validation_start > validation_end:
            raise TemporalLoraConfigurationError(
                "{0}: validation_start must be on or before validation_end.".format(
                    self.window_id
                )
            )
        if self.medium_trading_days <= 0 or self.fast_trading_days <= 0:
            raise TemporalLoraConfigurationError(
                "{0}: trailing adapter windows must be positive.".format(
                    self.window_id
                )
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalLoraSignalConfig:
    data_dir: str
    output_dir: str
    context_length: int = 512
    forecast_horizon: int = 1
    eval_stride: int = 1
    tickers: Optional[List[str]] = None
    universe_file: Optional[str] = None
    ticker_limit: int = 100
    model_id: str = "NeoQuasar/Kronos-small"
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base"
    kronos_cache_dir: Optional[str] = None
    device: str = "auto"
    temperature: float = 0.6
    top_k: int = 0
    top_p: float = 0.9
    sample_count: int = 10
    clip: float = 5.0
    random_seed: int = 42
    calibration_mode: str = "vol_rescale_mean_center"
    calibration_lookback: int = 20
    min_assets_per_date: int = 2
    ensemble_weights: Dict[str, float] = field(
        default_factory=lambda: {"slow": 0.25, "medium": 0.45, "fast": 0.30}
    )
    adapters: Dict[str, TemporalLoraAdapterConfig] = field(
        default_factory=lambda: {
            "slow": TemporalLoraAdapterConfig(name="slow"),
            "medium": TemporalLoraAdapterConfig(name="medium"),
            "fast": TemporalLoraAdapterConfig(name="fast"),
        }
    )
    windows: List[TemporalLoraWindowConfig] = field(default_factory=list)
    target_module_prefixes: List[str] = field(
        default_factory=lambda: ["transformer", "dep_layer"]
    )
    exclude_module_keywords: List[str] = field(
        default_factory=lambda: ["embedding", "time_emb", "head", "tokenizer"]
    )
    modal_gpu_seconds_price: float = 0.000222
    budget_cap_usd: float = 75.0
    include_shuffled_control: bool = False

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TemporalLoraSignalConfig":
        expected = {
            "data_dir",
            "output_dir",
            "context_length",
            "forecast_horizon",
            "eval_stride",
            "tickers",
            "universe_file",
            "ticker_limit",
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
            "calibration_mode",
            "calibration_lookback",
            "min_assets_per_date",
            "ensemble_weights",
            "adapters",
            "windows",
            "target_module_prefixes",
            "exclude_module_keywords",
            "modal_gpu_seconds_price",
            "budget_cap_usd",
            "include_shuffled_control",
        }
        unknown = sorted(set(raw) - expected)
        if unknown:
            raise TemporalLoraConfigurationError(
                "Unknown config keys: {0}".format(", ".join(unknown))
            )
        if "data_dir" not in raw or "output_dir" not in raw:
            raise TemporalLoraConfigurationError(
                "Temporal LoRA config requires data_dir and output_dir."
            )

        raw_adapters = raw.get("adapters") or {}
        if raw_adapters and not isinstance(raw_adapters, dict):
            raise TemporalLoraConfigurationError("adapters must be a mapping.")
        adapter_names = ["slow", "medium", "fast"]
        if raw_adapters:
            adapter_names = list(raw_adapters)
        adapters = {
            name: TemporalLoraAdapterConfig.from_dict(
                name,
                raw_adapters.get(name) if isinstance(raw_adapters, dict) else None,
            )
            for name in adapter_names
        }
        windows = [
            TemporalLoraWindowConfig.from_dict(item)
            for item in raw.get("windows", [])
        ]
        payload = dict(raw)
        payload["adapters"] = adapters
        payload["windows"] = windows
        if "ensemble_weights" in payload:
            payload["ensemble_weights"] = {
                str(k): float(v) for k, v in dict(payload["ensemble_weights"]).items()
            }
        return cls(**payload)

    def validate(self) -> None:
        if self.context_length <= 0 or self.context_length > 512:
            raise TemporalLoraConfigurationError(
                "context_length must be in the range [1, 512]."
            )
        if self.forecast_horizon != 1:
            raise TemporalLoraConfigurationError(
                "The first Temporal LoRA signal run only supports forecast_horizon=1."
            )
        if self.eval_stride <= 0:
            raise TemporalLoraConfigurationError("eval_stride must be positive.")
        if self.ticker_limit <= 0:
            raise TemporalLoraConfigurationError("ticker_limit must be positive.")
        if not self.model_id:
            raise TemporalLoraConfigurationError("model_id must be non-empty.")
        if not self.tokenizer_id:
            raise TemporalLoraConfigurationError("tokenizer_id must be non-empty.")
        if self.device not in ("auto", "cpu", "cuda", "gpu", "mps"):
            raise TemporalLoraConfigurationError(
                "device must be one of: auto, cpu, cuda, gpu, mps."
            )
        if self.temperature <= 0:
            raise TemporalLoraConfigurationError("temperature must be positive.")
        if self.top_k < 0:
            raise TemporalLoraConfigurationError("top_k must be non-negative.")
        if not 0 < self.top_p <= 1:
            raise TemporalLoraConfigurationError("top_p must be in (0, 1].")
        if self.sample_count <= 0:
            raise TemporalLoraConfigurationError("sample_count must be positive.")
        if self.clip <= 0:
            raise TemporalLoraConfigurationError("clip must be positive.")
        if self.calibration_lookback <= 0:
            raise TemporalLoraConfigurationError(
                "calibration_lookback must be positive."
            )
        if self.min_assets_per_date <= 0:
            raise TemporalLoraConfigurationError(
                "min_assets_per_date must be positive."
            )
        if not self.adapters:
            raise TemporalLoraConfigurationError("At least one adapter is required.")
        required_adapters = {"slow", "medium", "fast"}
        missing_adapters = sorted(required_adapters - set(self.adapters))
        if missing_adapters:
            raise TemporalLoraConfigurationError(
                "adapters missing: {0}".format(", ".join(missing_adapters))
            )
        for adapter in self.adapters.values():
            adapter.validate()
        required_weights = {"slow", "medium", "fast"}
        missing_weights = sorted(required_weights - set(self.ensemble_weights))
        if missing_weights:
            raise TemporalLoraConfigurationError(
                "ensemble_weights missing: {0}".format(", ".join(missing_weights))
            )
        if sum(float(value) for value in self.ensemble_weights.values()) <= 0:
            raise TemporalLoraConfigurationError(
                "ensemble_weights must have a positive total weight."
            )
        if not self.windows:
            raise TemporalLoraConfigurationError("At least one window is required.")
        seen = set()
        for window in self.windows:
            window.validate()
            if window.window_id in seen:
                raise TemporalLoraConfigurationError(
                    "Duplicate window_id: {0}".format(window.window_id)
                )
            seen.add(window.window_id)
        if not self.target_module_prefixes:
            raise TemporalLoraConfigurationError(
                "target_module_prefixes must contain at least one prefix."
            )
        if self.modal_gpu_seconds_price <= 0:
            raise TemporalLoraConfigurationError(
                "modal_gpu_seconds_price must be positive."
            )
        if self.budget_cap_usd <= 0:
            raise TemporalLoraConfigurationError("budget_cap_usd must be positive.")

    @property
    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir).resolve()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["adapters"] = {
            name: adapter.to_dict() for name, adapter in self.adapters.items()
        }
        payload["windows"] = [window.to_dict() for window in self.windows]
        return payload


def load_temporal_lora_signal_config(
    config_path: str,
    validate: bool = True,
) -> TemporalLoraSignalConfig:
    path = Path(config_path)
    if not path.exists():
        raise TemporalLoraConfigurationError(
            "Config file does not exist: {0}".format(path)
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TemporalLoraConfigurationError("Config file must contain a YAML mapping.")
    config = TemporalLoraSignalConfig.from_dict(raw)
    if validate:
        config.validate()
    return config


def _coerce_datetime(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    return datetime.strptime(str(value), "%Y-%m-%d")
