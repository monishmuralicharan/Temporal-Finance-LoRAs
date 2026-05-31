import json
from pathlib import Path

from temporal_finance.modal_temporal_lora_signal import (
    _collect_temporal_lora_artifacts,
    _load_detached_temporal_lora_state,
    _write_detached_temporal_lora_state,
    build_modal_temporal_lora_config,
)
from temporal_finance.temporal_lora_config import TemporalLoraSignalConfig


def test_detached_temporal_lora_state_round_trip(tmp_path):
    state = {
        "version": 1,
        "status": "submitted",
        "config_path": "configs/temporal_lora_signal_run.yaml",
        "output_root": str(tmp_path),
        "gpu": "L4",
        "function_call_id": "fc-123",
        "function_call_dashboard_url": "https://modal.example/call",
        "submitted_at_unix": 100.0,
    }

    path = _write_detached_temporal_lora_state(tmp_path, state)

    assert path == tmp_path / "detached_state.json"
    assert _load_detached_temporal_lora_state(path) == state


def test_collect_temporal_lora_artifacts_writes_outputs_and_updates_state(tmp_path):
    state_path = tmp_path / "detached_state.json"
    state = {
        "version": 1,
        "status": "submitted",
        "config_path": "configs/temporal_lora_signal_run.yaml",
        "output_root": str(tmp_path / "outputs"),
        "gpu": "L4",
        "function_call_id": "fc-123",
        "function_call_dashboard_url": None,
        "submitted_at_unix": 100.0,
    }
    payload = {
        "decision": {"decision": "fail", "accepted": False},
        "artifacts": {
            "metrics_by_window.csv": "window_id,variant\npooled,weighted_ensemble\n",
            "signal_decision.json": json.dumps({"decision": "fail"}),
        },
    }

    paths = _collect_temporal_lora_artifacts(
        state=state,
        payload=payload,
        state_path=state_path,
        collected_at_unix=115.0,
    )

    assert paths["metrics_by_window.csv"].exists()
    updated = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated["status"] == "collected"
    assert updated["runtime_seconds"] == 15.0
    assert updated["decision"] == {"decision": "fail", "accepted": False}


def test_build_modal_temporal_lora_config_maps_dataset_path():
    config = TemporalLoraSignalConfig.from_dict(
        {
            "data_dir": "dataset/data/kronos/combined_by_ticker",
            "output_dir": "outputs/temporal_lora_signal_run",
            "context_length": 5,
            "tickers": ["AAA", "BBB"],
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
    )

    remote = build_modal_temporal_lora_config(config)

    assert remote.data_dir == "/root/dataset/data/kronos/combined_by_ticker"
    assert remote.kronos_cache_dir == "/cache/huggingface"
    assert remote.device == "gpu"
    assert remote.tickers == ["AAA", "BBB"]


def test_build_modal_temporal_lora_config_resolves_local_universe(tmp_path):
    universe = tmp_path / "universe.txt"
    universe.write_text("AAA\nBBB\nCCC\n", encoding="utf-8")
    config = TemporalLoraSignalConfig.from_dict(
        {
            "data_dir": "dataset/data/kronos/combined_by_ticker",
            "output_dir": "outputs/temporal_lora_signal_run",
            "context_length": 5,
            "universe_file": str(universe),
            "ticker_limit": 2,
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
    )

    remote = build_modal_temporal_lora_config(config)

    assert remote.tickers == ["AAA", "BBB"]
    assert remote.universe_file is None
