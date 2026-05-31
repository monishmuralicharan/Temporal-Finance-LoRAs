import json
from pathlib import Path

import pandas as pd

from temporal_finance.temporal_lora_config import TemporalLoraSignalConfig
from temporal_finance.temporal_lora_signal import (
    build_metrics_tables,
    decide_temporal_lora_signal,
    ensemble_predictions,
    run_temporal_lora_signal,
    serialize_temporal_lora_artifacts,
    write_temporal_lora_artifacts,
)


def test_ensemble_predictions_combines_returns_and_reconstructs_close():
    base = _prediction_frame("slow", [0.01, 0.02])
    medium = _prediction_frame("medium", [0.02, 0.01])
    fast = _prediction_frame("fast", [-0.01, 0.03])

    result = ensemble_predictions(
        {"slow": base, "medium": medium, "fast": fast},
        weights={"slow": 0.25, "medium": 0.45, "fast": 0.30},
        variant="weighted_ensemble",
    )

    expected_return = 0.25 * 0.01 + 0.45 * 0.02 + 0.30 * -0.01
    assert result.loc[0, "variant"] == "weighted_ensemble"
    assert result.loc[0, "pred_return"] == expected_return
    assert result.loc[0, "pred_close_target"] == 100.0 * (1.0 + expected_return)
    assert result.loc[0, "pred_high_target"] >= result.loc[0, "pred_close_target"]
    assert result.loc[0, "pred_low_target"] <= result.loc[0, "pred_close_target"]


def test_decision_logic_strong_weak_and_fail():
    strong = _decision_metrics(weighted_rank=0.04, frozen_rank=0.01, stale_rank=0.02)
    assert decide_temporal_lora_signal(strong)["decision"] == "strong_pass"

    weak = _decision_metrics(
        weighted_rank=0.04,
        frozen_rank=0.01,
        stale_rank=0.05,
        weighted_directional=0.515,
    )
    assert decide_temporal_lora_signal(weak)["decision"] == "weak_pass"

    fail = _decision_metrics(weighted_rank=0.012, frozen_rank=0.01, stale_rank=0.00)
    assert decide_temporal_lora_signal(fail)["decision"] == "fail"


def test_build_metrics_tables_includes_pooled_and_daily_rows():
    frame = _prediction_frame("weighted_ensemble", [0.01, 0.02, 0.03, 0.04])
    frame.loc[:, "window_id"] = ["W1", "W1", "W2", "W2"]
    frame.loc[:, "ticker"] = ["AAA", "BBB", "AAA", "BBB"]
    frame.loc[:, "date"] = ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
    frame.loc[:, "actual_return"] = [0.011, 0.021, 0.029, 0.041]

    metrics, daily = build_metrics_tables({"weighted_ensemble": frame}, min_assets_per_date=2)

    assert set(metrics["window_id"]) == {"W1", "W2", "pooled"}
    assert not daily.empty
    assert set(daily["variant"]) == {"weighted_ensemble"}


def test_artifact_serialization_round_trip_includes_adapter_weights(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "metrics_by_window.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    adapter_path = source / "adapters" / "W1" / "slow.pt"
    adapter_path.parent.mkdir(parents=True)
    adapter_path.write_bytes(b"adapter-bytes")

    artifacts = serialize_temporal_lora_artifacts(source)
    paths = write_temporal_lora_artifacts(tmp_path / "out", artifacts)

    assert "adapters/W1/slow.pt" in artifacts
    assert paths["adapters/W1/slow.pt"].read_bytes() == b"adapter-bytes"
    assert paths["metrics_by_window.csv"].read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_run_temporal_lora_signal_with_fake_engine_writes_outputs(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_history_csv(data_dir / "AAA.csv", start="2024-01-01", periods=12)
    _write_history_csv(data_dir / "BBB.csv", start="2024-01-01", periods=12)
    config = TemporalLoraSignalConfig.from_dict(
        {
            "data_dir": str(data_dir),
            "output_dir": str(tmp_path / "outputs"),
            "tickers": ["AAA", "BBB"],
            "context_length": 3,
            "forecast_horizon": 1,
            "eval_stride": 1,
            "calibration_mode": "none",
            "min_assets_per_date": 2,
            "adapters": {
                "slow": {"epochs": 1, "batch_size": 2},
                "medium": {"epochs": 1, "batch_size": 2},
                "fast": {"epochs": 1, "batch_size": 2},
            },
            "windows": [
                {
                    "window_id": "W1",
                    "slow_train_start": "2024-01-04",
                    "slow_train_end": "2024-01-10",
                    "medium_trading_days": 3,
                    "fast_trading_days": 2,
                    "validation_start": "2024-01-11",
                    "validation_end": "2024-01-12",
                }
            ],
        }
    )

    result = run_temporal_lora_signal(config, engine=FakeEngine())

    assert (tmp_path / "outputs" / "predictions_weighted_ensemble.csv").exists()
    assert (tmp_path / "outputs" / "metrics_by_window.csv").exists()
    assert result.decision["decision"] in {"strong_pass", "weak_pass", "fail"}
    registry = json.loads((tmp_path / "outputs" / "adapter_registry.json").read_text())
    assert len(registry["adapters"]) == 3


class FakeEngine:
    def train_adapter(self, adapter_name, examples, histories, adapter_config, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes((adapter_name + "-state").encode("ascii"))
        return {
            "adapter_name": adapter_name,
            "example_count": len(examples),
            "best_loss": 1.0,
            "adapter_path": str(output_path),
        }

    def predict_examples(self, examples, histories, variant, adapter_path=None):
        returns = {
            "frozen_base": 0.0,
            "slow": 0.01,
            "medium": 0.02,
            "fast": -0.01,
            "stale_slow": 0.005,
            "stale_medium": 0.005,
            "stale_fast": 0.005,
        }
        pred_return = returns.get(variant, 0.0)
        rows = []
        for example in examples:
            history = histories[example.ticker]
            last_close = float(history.iloc[example.context_end_index]["close"])
            actual_close = float(history.iloc[example.target_index]["close"])
            pred_close = last_close * (1.0 + pred_return)
            actual_return = (actual_close / last_close) - 1.0
            rows.append(
                {
                    "ticker": example.ticker,
                    "window_id": example.window_id,
                    "variant": variant,
                    "date": example.context_end_date,
                    "target_date": example.target_date,
                    "forecast_horizon": 1,
                    "last_context_close": last_close,
                    "pred_close_target": pred_close,
                    "actual_close_target": actual_close,
                    "pred_return": pred_return,
                    "actual_return": actual_return,
                    "pred_open_target": pred_close,
                    "pred_high_target": pred_close + 1.0,
                    "pred_low_target": pred_close - 1.0,
                    "pred_volume_target": 100.0,
                    "pred_close_t5": pred_close,
                    "actual_close_t5": actual_close,
                    "pred_5d_return": pred_return,
                    "actual_5d_return": actual_return,
                    "pred_open_t5": pred_close,
                    "pred_high_t5": pred_close + 1.0,
                    "pred_low_t5": pred_close - 1.0,
                    "pred_volume_t5": 100.0,
                }
            )
        return pd.DataFrame(rows)


def _decision_metrics(
    weighted_rank,
    frozen_rank,
    stale_rank,
    weighted_directional=0.53,
):
    rows = []
    for window_id, weighted, frozen in [
        ("W1", weighted_rank, frozen_rank),
        ("W2", weighted_rank, frozen_rank),
        ("W3", frozen_rank - 0.01, frozen_rank),
    ]:
        rows.extend(
            [
                _metric_row(window_id, "weighted_ensemble", weighted, weighted_directional),
                _metric_row(window_id, "frozen_base", frozen, 0.51),
            ]
        )
    rows.extend(
        [
            _metric_row("pooled", "weighted_ensemble", weighted_rank, weighted_directional),
            _metric_row("pooled", "frozen_base", frozen_rank, 0.51),
            _metric_row("pooled", "stale_ensemble", stale_rank, 0.50),
            _metric_row("pooled", "naive_zero_return", 0.0, 0.50),
        ]
    )
    return pd.DataFrame(rows)


def _metric_row(window_id, variant, rank, directional):
    return {
        "window_id": window_id,
        "variant": variant,
        "daily_mean_rank_ic": rank,
        "spearman_rank_ic": rank,
        "directional_accuracy": directional,
        "pred_abs_return_multiple": 1.0,
        "invalid_ohlc_rate": 0.0,
        "sample_count": 100,
    }


def _prediction_frame(variant, returns):
    rows = []
    for index, pred_return in enumerate(returns):
        last_close = 100.0 + index
        actual_return = pred_return + 0.001
        rows.append(
            {
                "ticker": f"T{index}",
                "window_id": "W1",
                "variant": variant,
                "date": f"2024-01-{index + 1:02d}",
                "target_date": f"2024-01-{index + 2:02d}",
                "forecast_horizon": 1,
                "last_context_close": last_close,
                "pred_close_target": last_close * (1.0 + pred_return),
                "actual_close_target": last_close * (1.0 + actual_return),
                "pred_return": pred_return,
                "actual_return": actual_return,
                "pred_open_target": last_close * (1.0 + pred_return),
                "pred_high_target": last_close * (1.0 + pred_return) + 1.0,
                "pred_low_target": last_close * (1.0 + pred_return) - 1.0,
                "pred_volume_target": 1000.0,
                "pred_close_t5": last_close * (1.0 + pred_return),
                "actual_close_t5": last_close * (1.0 + actual_return),
                "pred_5d_return": pred_return,
                "actual_5d_return": actual_return,
                "pred_open_t5": last_close * (1.0 + pred_return),
                "pred_high_t5": last_close * (1.0 + pred_return) + 1.0,
                "pred_low_t5": last_close * (1.0 + pred_return) - 1.0,
                "pred_volume_t5": 1000.0,
            }
        )
    return pd.DataFrame(rows)


def _write_history_csv(path: Path, start, periods):
    dates = pd.bdate_range(start, periods=periods)
    close = pd.Series(range(10, 10 + periods), dtype=float)
    frame = pd.DataFrame(
        {
            "timestamps": dates.strftime("%Y-%m-%d"),
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0,
            "amount": close * 1000.0,
        }
    )
    frame.to_csv(path, index=False)
