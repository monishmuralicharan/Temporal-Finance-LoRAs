import json
from pathlib import Path

import pandas as pd

from temporal_finance.modal_kronos_checkpoint4 import (
    _collect_detached_benchmark_artifacts,
    _load_detached_benchmark_state,
    _write_detached_benchmark_state,
)


def test_detached_benchmark_state_round_trip(tmp_path):
    state = {
        "version": 1,
        "status": "submitted",
        "manifest_path": "configs/example.yaml",
        "output_root": str(tmp_path),
        "experiment_name": "base",
        "calibration_modes": ["none"],
        "gpu": "L4",
        "only_promoted_from": "",
        "function_call_id": "fc-123",
        "function_call_dashboard_url": "https://modal.example/call",
        "submitted_at_unix": 100.0,
    }

    state_path = _write_detached_benchmark_state(tmp_path, state)

    assert state_path == tmp_path / "detached_state.json"
    assert _load_detached_benchmark_state(state_path) == state


def test_collect_detached_benchmark_artifacts_writes_summary(tmp_path):
    base_config_path = tmp_path / "base.yaml"
    base_config_path.write_text(
        "\n".join(
            [
                "tickers:",
                "  - AAPL",
                "  - MSFT",
                "start_date: 2021-05-28",
                "eval_start_date: 2025-01-02",
                "end_date: 2026-05-27",
                "context_length: 512",
                "forecast_horizon: 5",
                "target_column: close",
                "output_dir: outputs/ignored",
                "batch_size: 8",
                "eval_stride: 21",
                "data_source: local_kronos_csv",
                "local_data_dir: dataset/data/kronos/combined_by_ticker",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "name: detached_mock",
                "stage: detached_test",
                "base_config: {0}".format(base_config_path),
                "output_dir: {0}".format(tmp_path / "benchmark"),
                "min_assets_per_date: 2",
                "experiments:",
                "  - name: base_mock",
                "    description: Mock detached base run",
                "    overrides:",
                "      model_id: NeoQuasar/Kronos-base",
                "      sample_count: 10",
                "    calibration_modes: [none, vol_rescale]",
            ]
        ),
        encoding="utf-8",
    )
    state_path = tmp_path / "detached_state.json"
    state = {
        "version": 1,
        "status": "submitted",
        "manifest_path": str(manifest_path),
        "output_root": str(tmp_path / "benchmark"),
        "experiment_name": "base_mock",
        "calibration_modes": ["none"],
        "gpu": "L4",
        "only_promoted_from": "",
        "function_call_id": "fc-123",
        "function_call_dashboard_url": None,
        "submitted_at_unix": 100.0,
    }
    artifacts = {
        "predictions_csv": _make_prediction_frame().to_csv(index=False),
        "metrics_json": json.dumps(
            {"failed_tickers": [], "failed_ticker_count": 0}
        ),
    }

    paths = _collect_detached_benchmark_artifacts(
        state=state,
        artifacts=artifacts,
        state_path=state_path,
        collected_at_unix=115.0,
    )

    summary = pd.read_csv(paths["benchmark_summary_csv"])
    updated_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert list(summary["name"]) == ["base_mock"]
    assert summary.loc[0, "status"] == "completed"
    assert summary.loc[0, "modal_gpu"] == "L4"
    assert summary.loc[0, "runtime_seconds"] == 15.0
    assert updated_state["status"] == "collected"
    assert updated_state["outputs"]["benchmark_summary_csv"] == str(
        paths["benchmark_summary_csv"]
    )
    assert (tmp_path / "benchmark" / "base_mock" / "predictions.csv").exists()


def _make_prediction_frame():
    rows = []
    dates = pd.bdate_range("2025-01-02", periods=6)
    for ticker_index, ticker in enumerate(["AAPL", "MSFT"]):
        for index, date in enumerate(dates):
            last_close = 100.0 + ticker_index * 50.0 + index
            pred_return = 0.01 + 0.002 * index - 0.001 * ticker_index
            actual_return = 0.008 + 0.002 * index - 0.001 * ticker_index
            rows.append(
                {
                    "ticker": ticker,
                    "date": date.strftime("%Y-%m-%d"),
                    "target_date": (date + pd.offsets.BDay(5)).strftime(
                        "%Y-%m-%d"
                    ),
                    "last_context_close": last_close,
                    "pred_close_t5": last_close * (1.0 + pred_return),
                    "actual_close_t5": last_close * (1.0 + actual_return),
                    "pred_5d_return": pred_return,
                    "actual_5d_return": actual_return,
                    "pred_open_t5": last_close * (1.0 + pred_return),
                    "pred_high_t5": last_close * (1.0 + pred_return + 0.01),
                    "pred_low_t5": last_close * (1.0 + pred_return - 0.01),
                    "pred_volume_t5": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)
