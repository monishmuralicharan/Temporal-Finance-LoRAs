import json

import numpy as np
import pandas as pd
import pytest

from temporal_finance.kronos_benchmark import (
    CALIBRATION_VOL_RESCALE,
    benchmark_run_name,
    calibrate_kronos_predictions,
    evaluate_promotion_gate,
    load_kronos_benchmark_manifest,
    run_benchmark_manifest_local,
)


class DummyKronosForecaster:
    def forecast_batch(self, contexts, x_timestamps, y_timestamps):
        forecasts = []
        for context, y_timestamp in zip(contexts, y_timestamps):
            last_close = float(context["close"].iloc[-1])
            pred_len = len(y_timestamp)
            forecasts.append(
                pd.DataFrame(
                    {
                        "open": [last_close * 1.01] * pred_len,
                        "high": [last_close * 1.02] * pred_len,
                        "low": [last_close * 0.99] * pred_len,
                        "close": [last_close * 1.01] * pred_len,
                        "volume": [1.0] * pred_len,
                        "amount": [last_close] * pred_len,
                    },
                    index=pd.to_datetime(y_timestamp),
                )
            )
        return forecasts


def test_load_base_selection_manifest_expands_grid():
    manifest = load_kronos_benchmark_manifest(
        "configs/kronos_base_selection_smoke.yaml"
    )

    assert manifest.stage == "smoke"
    assert manifest.base_config_path == "configs/kronos_paper_proximal_smoke.yaml"
    assert len(manifest.experiments) == 8
    assert manifest.experiments[0].name == "small_greedy_adjusted"
    assert manifest.experiments[-1].calibration_modes == [
        "none",
        "vol_rescale",
        "vol_rescale_mean_center",
    ]


def test_benchmark_run_name_includes_calibration_mode():
    assert benchmark_run_name("small", "none") == "small"
    assert benchmark_run_name("small", "vol_rescale") == "small__vol_rescale"


def test_vol_rescale_preserves_rows_and_reduces_overamplification():
    predictions = _make_prediction_frame(row_count=40, pred_return=0.10, actual_return=0.02)

    calibrated = calibrate_kronos_predictions(
        predictions,
        CALIBRATION_VOL_RESCALE,
        lookback=5,
    )

    assert len(calibrated) == len(predictions)
    assert calibrated["calibration_mode"].eq(CALIBRATION_VOL_RESCALE).all()
    original_error = abs(predictions["pred_5d_return"].abs().mean() - 0.02)
    calibrated_error = abs(calibrated["pred_5d_return"].abs().mean() - 0.02)
    assert calibrated_error < original_error
    assert set(["pred_open_t5", "pred_high_t5", "pred_low_t5"]).issubset(
        calibrated.columns
    )


def test_promotion_gate_accepts_balanced_candidate():
    result = evaluate_promotion_gate(
        metrics={
            "pearson_ic": 0.05,
            "spearman_rank_ic": 0.06,
            "daily_ic_positive_rate": 0.60,
            "daily_rank_ic_positive_rate": 0.58,
            "directional_accuracy": 0.56,
            "pred_abs_return_multiple": 1.1,
            "positive_rate_gap": 0.05,
            "output_total_invalid_ohlc_rate": 0.0,
        },
        references={"current_kronos_rank_ic": 0.015, "ridge_rank_ic": 0.08},
    )

    assert result.passed
    assert result.reasons == []


def test_promotion_gate_rejects_miscalibrated_candidate():
    result = evaluate_promotion_gate(
        metrics={
            "pearson_ic": -0.08,
            "spearman_rank_ic": -0.07,
            "daily_ic_positive_rate": 0.43,
            "daily_rank_ic_positive_rate": 0.48,
            "directional_accuracy": 0.37,
            "pred_abs_return_multiple": 3.8,
            "positive_rate_gap": 0.41,
            "output_total_invalid_ohlc_rate": 0.0,
        },
        references={"current_kronos_rank_ic": 0.015},
    )

    assert not result.passed
    assert any("return IC" in reason for reason in result.reasons)
    assert any("predicted/actual" in reason for reason in result.reasons)


def test_run_benchmark_manifest_local_with_mock_forecaster(tmp_path):
    base_config_path = tmp_path / "base.yaml"
    base_config_path.write_text(
        "\n".join(
            [
                "tickers:",
                "  - AAPL",
                "  - MSFT",
                "start_date: 2020-01-01",
                "eval_start_date: 2020-04-15",
                "end_date: 2020-06-30",
                "context_length: 32",
                "forecast_horizon: 5",
                "target_column: Adj Close",
                "output_dir: outputs/ignored",
                "batch_size: 3",
                "eval_stride: 7",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "name: mock",
                "stage: smoke",
                "base_config: {0}".format(base_config_path),
                "output_dir: {0}".format(tmp_path / "benchmark"),
                "experiments:",
                "  - name: dummy_small",
                "    description: Dummy small run",
                "    overrides:",
                "      model_id: NeoQuasar/Kronos-small",
                "      sample_count: 1",
                "    calibration_modes: [none, vol_rescale]",
            ]
        ),
        encoding="utf-8",
    )

    records = run_benchmark_manifest_local(
        manifest_path=str(manifest_path),
        forecaster=DummyKronosForecaster(),
        histories={"AAPL": _make_history(100.0), "MSFT": _make_history(200.0)},
    )

    assert [record.name for record in records] == [
        "dummy_small",
        "dummy_small__vol_rescale",
    ]
    assert (tmp_path / "benchmark" / "benchmark_summary.csv").exists()
    assert (tmp_path / "benchmark" / "promoted_run_names.txt").exists()
    metrics = json.loads(
        (tmp_path / "benchmark" / "dummy_small" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert metrics["model_id"] == "NeoQuasar/Kronos-small"


def _make_prediction_frame(row_count, pred_return, actual_return):
    dates = pd.bdate_range("2020-01-01", periods=row_count)
    last_close = pd.Series(100.0 + np.arange(row_count), dtype=float)
    return pd.DataFrame(
        {
            "ticker": ["AAPL"] * row_count,
            "date": dates.strftime("%Y-%m-%d"),
            "target_date": (dates + pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
            "last_context_close": last_close,
            "pred_close_t5": last_close * (1.0 + pred_return),
            "actual_close_t5": last_close * (1.0 + actual_return),
            "pred_5d_return": [pred_return] * row_count,
            "actual_5d_return": [actual_return] * row_count,
            "pred_open_t5": last_close * (1.0 + pred_return),
            "pred_high_t5": last_close * (1.0 + pred_return + 0.01),
            "pred_low_t5": last_close * (1.0 + pred_return - 0.01),
            "pred_volume_t5": [1_000.0] * row_count,
        }
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
