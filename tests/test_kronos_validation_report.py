from pathlib import Path

import json

import pandas as pd
import pytest

from temporal_finance.kronos_benchmark import (
    build_benchmark_config,
    load_kronos_benchmark_manifest,
)
from temporal_finance.kronos_config import load_kronos_checkpoint4_config
from temporal_finance.kronos_validation_report import (
    build_naive_baseline_predictions,
    run_validation_report,
)


def test_validation_manifests_apply_expected_overrides():
    stride5 = load_kronos_benchmark_manifest(
        "configs/kronos_dataset_common100_context512_s10_stride5_base_selection.yaml"
    )
    stride5_base = load_kronos_checkpoint4_config(stride5.base_config_path)
    stride5_config = build_benchmark_config(
        stride5_base,
        stride5.experiments[0],
        Path(stride5.output_dir),
    )

    recent_daily = load_kronos_benchmark_manifest(
        "configs/kronos_dataset_common100_context512_s10_recent_daily_base_selection.yaml"
    )
    recent_daily_base = load_kronos_checkpoint4_config(recent_daily.base_config_path)
    recent_daily_config = build_benchmark_config(
        recent_daily_base,
        recent_daily.experiments[0],
        Path(recent_daily.output_dir),
    )

    mixed200 = load_kronos_benchmark_manifest(
        "configs/kronos_dataset_mixed200_context512_s10_stride21_base_selection.yaml"
    )
    mixed200_base = load_kronos_checkpoint4_config(mixed200.base_config_path)
    mixed200_config = build_benchmark_config(
        mixed200_base,
        mixed200.experiments[0],
        Path(mixed200.output_dir),
    )

    assert stride5_config.eval_stride == 5
    assert stride5_config.eval_start_date == "2025-01-02"
    assert recent_daily_config.eval_stride == 1
    assert recent_daily_config.eval_start_date == "2025-10-01"
    assert mixed200.output_dir.endswith("dataset_mixed200_context512_s10_stride21")
    assert mixed200_config.eval_stride == 21


def test_naive_baselines_use_only_history_at_or_before_origin(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dates = pd.date_range("2025-01-01", periods=12, freq="D")
    closes = [100, 101, 102, 103, 104, 110, 300, 301, 302, 303, 304, 305]
    pd.DataFrame(
        {
            "timestamps": dates.strftime("%Y-%m-%d"),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000] * len(closes),
            "amount": [100000] * len(closes),
        }
    ).to_csv(data_dir / "AAA.csv", index=False)
    predictions = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "date": "2025-01-06",
                "target_date": "2025-01-11",
                "last_context_close": 110.0,
                "pred_close_t5": 110.0,
                "actual_close_t5": 304.0,
                "pred_5d_return": 0.0,
                "actual_5d_return": (304.0 / 110.0) - 1.0,
            }
        ]
    )

    baselines = build_naive_baseline_predictions(predictions, data_dir)

    assert baselines["prior_5d_momentum"].loc[0, "pred_5d_return"] == pytest.approx(
        0.10
    )
    assert baselines["prior_20d_momentum_scaled_to_5d"].loc[
        0, "pred_5d_return"
    ] != pytest.approx((304.0 / 110.0) - 1.0)


def test_validation_report_writes_decision_and_baseline_outputs(tmp_path):
    benchmark_dir = tmp_path / "dataset_common100_context512_s10_stride5"
    run_dir = benchmark_dir / "candidate"
    run_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = []
    for offset, ticker in enumerate(["AAA", "BBB", "CCC"]):
        dates = pd.date_range("2025-01-01", periods=40, freq="D")
        closes = [100 + offset + i for i in range(40)]
        pd.DataFrame(
            {
                "timestamps": dates.strftime("%Y-%m-%d"),
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [1000] * len(closes),
                "amount": [100000] * len(closes),
            }
        ).to_csv(data_dir / f"{ticker}.csv", index=False)
        origin = 25
        target = 30
        last_close = closes[origin]
        actual = (closes[target] / last_close) - 1.0
        pred = actual + (offset * 0.001)
        rows.append(
            {
                "ticker": ticker,
                "date": dates[origin].strftime("%Y-%m-%d"),
                "target_date": dates[target].strftime("%Y-%m-%d"),
                "last_context_close": last_close,
                "pred_close_t5": last_close * (1.0 + pred),
                "actual_close_t5": closes[target],
                "pred_5d_return": pred,
                "actual_5d_return": actual,
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "name": "candidate",
                "stage": "test",
                "status": "completed",
                "calibration_mode": "vol_rescale_mean_center",
                "passed_gate": True,
                "output_dir": str(run_dir),
                "paper_reference_return_ic": 0.0665,
                "paper_reference_return_rank_ic": 0.0622,
                "return_ic_gap_to_paper": 0.01,
                "return_rank_ic_gap_to_paper": 0.01,
                "output_total_invalid_ohlc_rate": 0.0,
            }
        ]
    ).to_csv(benchmark_dir / "benchmark_summary.csv", index=False)

    paths = run_validation_report(
        benchmark_dirs=[str(benchmark_dir)],
        local_data_dir=str(data_dir),
        output_dir=str(tmp_path / "report"),
        min_assets_per_date=2,
    )

    assert paths["validation_summary"].exists()
    assert paths["baseline_comparison"].exists()
    assert paths["period_metrics"].exists()
    decision = json.loads(paths["base_model_decision"].read_text(encoding="utf-8"))
    assert decision["decision"] == "reject"
    assert "dataset_common100_context512_s10_recent_daily" in " ".join(
        decision["rejection_reasons"]
    )
