from temporal_finance.kronos_config import KronosCheckpoint4Config
from temporal_finance.kronos_modal_utils import (
    DEFAULT_KRONOS_MODAL_CACHE_DIR,
    DEFAULT_KRONOS_MODAL_OUTPUT_DIR,
    KRONOS_UPSTREAM_COMMIT,
    build_kronos_modal_run_config,
    _map_local_kronos_data_dir,
)


def test_build_kronos_modal_run_config_overrides_remote_paths(tmp_path):
    config = KronosCheckpoint4Config(
        ticker="AAPL",
        start_date="2020-01-01",
        eval_start_date="2020-01-07",
        end_date="2020-01-31",
        context_length=128,
        forecast_horizon=5,
        target_column="Adj Close",
        output_dir=str(tmp_path / "outputs"),
        kronos_cache_dir=None,
        device="cpu",
    )

    modal_config = build_kronos_modal_run_config(config)

    assert modal_config.kronos_cache_dir == DEFAULT_KRONOS_MODAL_CACHE_DIR
    assert modal_config.output_dir == DEFAULT_KRONOS_MODAL_OUTPUT_DIR
    assert modal_config.device == "gpu"


def test_kronos_upstream_commit_is_pinned():
    assert len(KRONOS_UPSTREAM_COMMIT) == 40



def test_map_local_kronos_data_dir_preserves_kronos_subdirectory():
    assert _map_local_kronos_data_dir(
        "dataset/data/kronos/combined_by_ticker"
    ) == "/root/dataset/data/kronos/combined_by_ticker"
