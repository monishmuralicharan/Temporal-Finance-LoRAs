import json

import pandas as pd
import pytest

from temporal_finance.kronos_output_evaluation import (
    compute_daily_cross_section_metrics,
    evaluate_kronos_output,
    run_kronos_output_evaluation,
)


def test_evaluate_kronos_output_scores_available_columns():
    predictions = _make_predictions()

    metrics = evaluate_kronos_output(predictions, min_assets_per_date=2)

    assert metrics["row_count"] == 4
    assert metrics["ticker_count"] == 2
    assert metrics["return_metrics"]["directional_accuracy"] == pytest.approx(0.75)
    assert metrics["paired_series"]["close_t5"]["mape"] == pytest.approx(
        ((2.0 / 102.0) + (1.0 / 99.0) + (1.0 / 104.0) + (1.0 / 98.0)) / 4.0
    )
    assert metrics["ohlc_validity"]["total_invalid_count"] == 1
    assert metrics["ohlc_validity"]["total_negative_volume_count"] == 1
    assert metrics["daily_cross_section"]["window_count"] == 2


def test_daily_cross_section_metrics_require_multiple_assets():
    daily = compute_daily_cross_section_metrics(
        _make_predictions(),
        min_assets_per_date=3,
    )

    assert daily.empty


def test_run_kronos_output_evaluation_writes_outputs(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    _make_predictions().to_csv(predictions_path, index=False)

    result = run_kronos_output_evaluation(
        predictions_path=str(predictions_path),
        output_dir=str(tmp_path / "evaluation"),
        group_columns=["adapter_stage"],
    )

    assert result.metrics_path.exists()
    assert result.daily_metrics_path.exists()
    assert result.group_metrics_path.exists()
    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert payload["group_metrics"][0]["adapter_stage"] == "fast"


def _make_predictions():
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "date": "2020-01-01",
                "target_date": "2020-01-08",
                "adapter_stage": "fast",
                "pred_5d_return": 0.04,
                "actual_5d_return": 0.02,
                "pred_close_t5": 104.0,
                "actual_close_t5": 102.0,
                "pred_open_t5": 103.0,
                "pred_high_t5": 105.0,
                "pred_low_t5": 101.0,
                "pred_volume_t5": 1000.0,
            },
            {
                "ticker": "MSFT",
                "date": "2020-01-01",
                "target_date": "2020-01-08",
                "adapter_stage": "fast",
                "pred_5d_return": -0.02,
                "actual_5d_return": -0.01,
                "pred_close_t5": 98.0,
                "actual_close_t5": 99.0,
                "pred_open_t5": 99.0,
                "pred_high_t5": 97.0,
                "pred_low_t5": 98.0,
                "pred_volume_t5": -1.0,
            },
            {
                "ticker": "AAPL",
                "date": "2020-01-02",
                "target_date": "2020-01-09",
                "adapter_stage": "slow",
                "pred_5d_return": -0.01,
                "actual_5d_return": 0.04,
                "pred_close_t5": 103.0,
                "actual_close_t5": 104.0,
                "pred_open_t5": 102.0,
                "pred_high_t5": 104.0,
                "pred_low_t5": 101.0,
                "pred_volume_t5": 1000.0,
            },
            {
                "ticker": "MSFT",
                "date": "2020-01-02",
                "target_date": "2020-01-09",
                "adapter_stage": "slow",
                "pred_5d_return": -0.03,
                "actual_5d_return": -0.02,
                "pred_close_t5": 97.0,
                "actual_close_t5": 98.0,
                "pred_open_t5": 98.0,
                "pred_high_t5": 99.0,
                "pred_low_t5": 96.0,
                "pred_volume_t5": 1000.0,
            },
        ]
    )
