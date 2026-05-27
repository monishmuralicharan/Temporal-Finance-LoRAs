import json

import pandas as pd
import pytest

from temporal_finance.kronos_config import KronosCheckpoint4Config
from temporal_finance.kronos_experiments import (
    PAPER_REFERENCE_RETURN_IC,
    PAPER_REFERENCE_RETURN_RANK_IC,
    build_experiment_config,
    build_experiment_record,
    build_kronos_benchmark_summary,
    compute_kronos_prediction_diagnostics,
    flatten_kronos_output_evaluation_metrics,
    kronos_experiment_definitions_for_suite,
    ordered_kronos_experiment_definitions,
    paper_proximal_kronos_experiment_definitions,
    select_best_kronos_experiment,
    should_skip_kronos_experiment,
    write_kronos_experiment_outputs,
)


def test_paper_proximal_suite_uses_sampling_profile():
    definitions = paper_proximal_kronos_experiment_definitions()

    assert [definition.name for definition in definitions] == [
        "base_sample10_adjusted",
        "small_sample10_adjusted",
    ]
    assert definitions[0].overrides["model_id"] == "NeoQuasar/Kronos-base"
    assert definitions[0].overrides["temperature"] == pytest.approx(0.6)
    assert definitions[0].overrides["top_k"] == 0
    assert definitions[0].overrides["top_p"] == pytest.approx(0.9)
    assert definitions[0].overrides["sample_count"] == 10


def test_paper_proximal_suite_can_include_sample20():
    definitions = kronos_experiment_definitions_for_suite(
        "paper-proximal",
        include_sample20=True,
    )

    assert definitions[-1].name == "base_sample20_adjusted"
    assert definitions[-1].overrides["sample_count"] == 20


def test_flatten_output_evaluation_metrics_extracts_daily_summary():
    metrics = {
        "row_count": 20,
        "ticker_count": 5,
        "daily_cross_section": {
            "window_count": 4,
            "mean_ic": 0.03,
            "mean_rank_ic": 0.04,
            "ic_positive_rate": 0.75,
            "rank_ic_positive_rate": 0.5,
            "mean_directional_accuracy": 0.55,
        },
        "return_metrics": {
            "mae": 0.02,
            "pearson_ic": 0.06,
            "spearman_rank_ic": 0.05,
        },
        "ohlc_validity": {"total_invalid_rate": 0.01},
    }

    flattened = flatten_kronos_output_evaluation_metrics(metrics)

    assert flattened["daily_window_count"] == 4
    assert flattened["daily_mean_ic"] == pytest.approx(0.03)
    assert flattened["output_return_rank_ic"] == pytest.approx(0.05)
    assert flattened["output_total_invalid_ohlc_rate"] == pytest.approx(0.01)


def test_benchmark_summary_compares_to_paper_reference():
    config = _make_config()
    definition = paper_proximal_kronos_experiment_definitions()[0]
    record = build_experiment_record(
        definition=definition,
        config=config,
        metrics={
            "model_id": "NeoQuasar/Kronos-base",
            "eval_stride": 21,
            "temperature": 0.6,
            "top_k": 0,
            "top_p": 0.9,
            "kronos_sample_count": 10,
            "directional_accuracy": 0.55,
            "mae": 0.04,
            "pearson_ic": 0.05,
            "spearman_rank_ic": 0.04,
            "pred_abs_return_multiple": 1.0,
            "positive_rate_gap": 0.1,
            "daily_mean_ic": 0.02,
            "daily_mean_rank_ic": 0.01,
        },
    )

    summary = build_kronos_benchmark_summary([record])

    assert summary.loc[0, "paper_reference_return_ic"] == pytest.approx(
        PAPER_REFERENCE_RETURN_IC
    )
    assert summary.loc[0, "paper_reference_return_rank_ic"] == pytest.approx(
        PAPER_REFERENCE_RETURN_RANK_IC
    )
    assert summary.loc[0, "return_ic_gap_to_paper"] == pytest.approx(
        0.05 - PAPER_REFERENCE_RETURN_IC
    )
    assert summary.loc[0, "sample_count"] == 10


def test_compute_kronos_prediction_diagnostics_reports_calibration():
    predictions = pd.DataFrame(
        {
            "pred_5d_return": [0.10, -0.10],
            "actual_5d_return": [0.05, -0.05],
            "pred_close_t5": [110.0, 90.0],
            "actual_close_t5": [105.0, 95.0],
            "pred_open_t5": [109.0, 91.0],
            "pred_high_t5": [111.0, 89.0],
            "pred_low_t5": [108.0, 92.0],
            "pred_volume_t5": [1000.0, -1.0],
        }
    )

    diagnostics = compute_kronos_prediction_diagnostics(predictions)

    assert diagnostics["pred_abs_return_mean"] == pytest.approx(0.10)
    assert diagnostics["actual_abs_return_mean"] == pytest.approx(0.05)
    assert diagnostics["pred_abs_return_multiple"] == pytest.approx(2.0)
    assert diagnostics["positive_rate_gap"] == pytest.approx(0.0)
    assert diagnostics["invalid_pred_ohlc_count"] == 1
    assert diagnostics["negative_pred_volume_count"] == 1


def test_select_best_rejects_miscalibrated_variant():
    config = _make_config()
    definitions = ordered_kronos_experiment_definitions()
    baseline = build_experiment_record(
        definition=definitions[0],
        config=config,
        metrics={
            "directional_accuracy": 0.55,
            "mae": 0.03,
            "pearson_ic": 0.02,
            "pred_abs_return_multiple": 1.4,
            "positive_rate_gap": 0.1,
        },
    )
    miscalibrated = build_experiment_record(
        definition=definitions[2],
        config=config,
        metrics={
            "directional_accuracy": 0.80,
            "mae": 0.20,
            "pearson_ic": 0.10,
            "pred_abs_return_multiple": 4.0,
            "positive_rate_gap": 0.4,
        },
    )

    best = select_best_kronos_experiment([baseline, miscalibrated])

    assert best is baseline
    assert not miscalibrated.passed_guardrails


def test_build_record_rejects_mae_worsening_against_reference():
    config = _make_config()
    definitions = ordered_kronos_experiment_definitions()

    record = build_experiment_record(
        definition=definitions[2],
        config=config,
        metrics={
            "directional_accuracy": 0.60,
            "mae": 0.12,
            "pearson_ic": 0.03,
            "pred_abs_return_multiple": 1.0,
            "positive_rate_gap": 0.1,
        },
        reference_metrics={
            "mae": 0.05,
            "pearson_ic": 0.02,
        },
    )

    assert not record.passed_guardrails
    assert "MAE worsened" in record.rejection_reasons[0]


def test_raw_ohlcv_definition_is_diagnostic_only():
    base = _make_config()
    raw_definition = ordered_kronos_experiment_definitions()[1]
    raw_config = build_experiment_config(
        base,
        raw_definition,
        base.resolved_output_dir.parent,
    )

    record = build_experiment_record(
        definition=raw_definition,
        config=raw_config,
        metrics={
            "directional_accuracy": 0.90,
            "mae": 0.01,
            "pearson_ic": 0.40,
            "pred_abs_return_multiple": 1.0,
            "positive_rate_gap": 0.0,
        },
    )

    assert raw_config.target_column == "Close"
    assert raw_config.use_adjusted_ohlc is False
    assert record.selectable is False
    assert select_best_kronos_experiment([record]) is None


def test_sampling_experiment_skips_until_calibrated_winner_exists():
    sampling_definition = ordered_kronos_experiment_definitions()[-1]

    skip_reason = should_skip_kronos_experiment(sampling_definition, [])

    assert skip_reason == "requires at least one calibrated selectable winner"


def test_write_kronos_experiment_outputs_marks_best(tmp_path):
    config = _make_config(tmp_path)
    definitions = ordered_kronos_experiment_definitions()
    first = build_experiment_record(
        definition=definitions[0],
        config=config,
        metrics={
            "directional_accuracy": 0.51,
            "mae": 0.04,
            "pearson_ic": 0.01,
            "pred_abs_return_multiple": 1.2,
            "positive_rate_gap": 0.1,
        },
    )
    second = build_experiment_record(
        definition=definitions[2],
        config=config,
        metrics={
            "directional_accuracy": 0.56,
            "mae": 0.04,
            "pearson_ic": 0.02,
            "pred_abs_return_multiple": 1.1,
            "positive_rate_gap": 0.1,
        },
    )

    paths = write_kronos_experiment_outputs([first, second], tmp_path)
    payload = json.loads(paths["summary_json"].read_text(encoding="utf-8"))

    assert paths["summary_csv"].exists()
    assert paths["benchmark_summary_csv"].exists()
    assert paths["benchmark_summary_json"].exists()
    assert paths["best_config"].exists()
    assert payload["best_experiment"] == "kronos_small_greedy_adjusted"
    assert second.selected_best is True


def _make_config(tmp_path=None):
    output_dir = "outputs/test"
    if tmp_path is not None:
        output_dir = str(tmp_path / "experiment")
    return KronosCheckpoint4Config(
        ticker="AAPL",
        tickers=["AAPL", "MSFT"],
        start_date="2020-01-01",
        eval_start_date="2020-04-15",
        end_date="2020-06-30",
        context_length=32,
        forecast_horizon=5,
        target_column="Adj Close",
        output_dir=output_dir,
        batch_size=3,
    )
