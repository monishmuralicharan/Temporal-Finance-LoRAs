import pytest

from temporal_finance.temporal_lora_config import (
    TemporalLoraConfigurationError,
    TemporalLoraSignalConfig,
    load_temporal_lora_signal_config,
)


def test_load_temporal_lora_signal_config_defaults():
    config = load_temporal_lora_signal_config("configs/temporal_lora_signal_run.yaml")

    assert config.model_id == "NeoQuasar/Kronos-small"
    assert config.context_length == 512
    assert config.forecast_horizon == 1
    assert config.ensemble_weights == {"slow": 0.25, "medium": 0.45, "fast": 0.3}
    assert [window.window_id for window in config.windows] == ["W1", "W2", "W3"]
    assert config.adapters["slow"].rank == 8


def test_temporal_lora_config_rejects_missing_weight():
    raw = _minimal_raw_config()
    raw["ensemble_weights"] = {"slow": 0.5, "medium": 0.5}

    config = TemporalLoraSignalConfig.from_dict(raw)

    with pytest.raises(TemporalLoraConfigurationError, match="ensemble_weights missing"):
        config.validate()


def test_temporal_lora_config_rejects_forward_leakage_window():
    raw = _minimal_raw_config()
    raw["windows"][0]["slow_train_end"] = "2025-01-02"

    config = TemporalLoraSignalConfig.from_dict(raw)

    with pytest.raises(TemporalLoraConfigurationError, match="before validation_start"):
        config.validate()


def _minimal_raw_config():
    return {
        "data_dir": "dataset/data/kronos/combined_by_ticker",
        "output_dir": "outputs/test_temporal_lora",
        "context_length": 5,
        "forecast_horizon": 1,
        "ensemble_weights": {"slow": 0.25, "medium": 0.45, "fast": 0.30},
        "windows": [
            {
                "window_id": "W1",
                "slow_train_start": "2024-01-01",
                "slow_train_end": "2024-12-31",
                "validation_start": "2025-01-02",
                "validation_end": "2025-01-31",
            }
        ],
    }
