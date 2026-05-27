"""Pure helpers for the Modal Kronos checkpoint 4 runner."""

from dataclasses import replace

from temporal_finance.kronos_config import KronosCheckpoint4Config

KRONOS_UPSTREAM_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
DEFAULT_KRONOS_MODAL_APP_NAME = "temporal-finance-kronos-checkpoint4"
DEFAULT_KRONOS_MODAL_GPU = "T4"
DEFAULT_KRONOS_MODAL_CACHE_VOLUME_NAME = "temporal-finance-kronos-cache"
DEFAULT_KRONOS_MODAL_CACHE_DIR = "/cache/huggingface"
DEFAULT_KRONOS_MODAL_OUTPUT_DIR = "/tmp/kronos_checkpoint4"
KRONOS_REPO_PATH = "/opt/Kronos"
KRONOS_REPO_URL = "https://github.com/shiyu-coder/Kronos.git"


def build_kronos_modal_run_config(
    config: KronosCheckpoint4Config,
    remote_cache_dir: str = DEFAULT_KRONOS_MODAL_CACHE_DIR,
    remote_output_dir: str = DEFAULT_KRONOS_MODAL_OUTPUT_DIR,
) -> KronosCheckpoint4Config:
    modal_config = replace(
        config,
        kronos_cache_dir=remote_cache_dir,
        output_dir=remote_output_dir,
        device="gpu",
    )
    modal_config.validate()
    return modal_config
