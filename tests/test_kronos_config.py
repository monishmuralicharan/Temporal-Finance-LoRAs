import pytest

from temporal_finance.kronos_config import (
    KronosConfigurationError,
    load_kronos_checkpoint4_config,
)


def test_load_kronos_config_accepts_tickers_list_without_ticker(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "tickers:",
                "  - AAPL",
                "  - MSFT",
                "start_date: 2015-01-01",
                "eval_start_date: 2020-01-01",
                "end_date: 2024-12-31",
                "context_length: 512",
                "forecast_horizon: 5",
                "target_column: Adj Close",
                "output_dir: outputs/test",
            ]
        ),
        encoding="utf-8",
    )

    config = load_kronos_checkpoint4_config(str(config_path))

    assert config.ticker == ""
    assert config.effective_tickers == ["AAPL", "MSFT"]
    assert config.model_id == "NeoQuasar/Kronos-base"
    assert config.tokenizer_id == "NeoQuasar/Kronos-Tokenizer-base"


def test_load_kronos_config_rejects_context_above_model_limit(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "ticker: AAPL",
                "start_date: 2015-01-01",
                "eval_start_date: 2020-01-01",
                "end_date: 2024-12-31",
                "context_length: 1024",
                "forecast_horizon: 5",
                "target_column: Adj Close",
                "output_dir: outputs/test",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(KronosConfigurationError, match="<= 512"):
        load_kronos_checkpoint4_config(str(config_path))


def test_load_kronos_config_rejects_duplicate_tickers(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "tickers:",
                "  - AAPL",
                "  - AAPL",
                "start_date: 2015-01-01",
                "eval_start_date: 2020-01-01",
                "end_date: 2024-12-31",
                "context_length: 512",
                "forecast_horizon: 5",
                "target_column: Adj Close",
                "output_dir: outputs/test",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(KronosConfigurationError, match="unique"):
        load_kronos_checkpoint4_config(str(config_path))


def test_load_paper_proximal_stride_config():
    config = load_kronos_checkpoint4_config(
        "configs/kronos_paper_proximal_stride21.yaml"
    )

    assert len(config.effective_tickers) >= 30
    assert config.model_id == "NeoQuasar/Kronos-base"
    assert config.temperature == pytest.approx(0.6)
    assert config.top_k == 0
    assert config.top_p == pytest.approx(0.9)
    assert config.sample_count == 10
    assert config.eval_stride == 21
    assert config.use_adjusted_ohlc is True


def test_load_paper_proximal_smoke_config():
    config = load_kronos_checkpoint4_config(
        "configs/kronos_paper_proximal_smoke.yaml"
    )

    assert len(config.effective_tickers) == 10
    assert config.temperature == pytest.approx(0.6)
    assert config.top_k == 0
    assert config.sample_count == 10
    assert config.batch_size == 8
