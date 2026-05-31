"""Pure helpers for the Modal Kronos checkpoint 4 runner."""

from dataclasses import replace
from pathlib import PurePosixPath

from temporal_finance.kronos_config import KronosCheckpoint4Config

KRONOS_UPSTREAM_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
DEFAULT_KRONOS_MODAL_APP_NAME = "temporal-finance-kronos-checkpoint4"
DEFAULT_KRONOS_MODAL_GPU = "T4"
DEFAULT_KRONOS_MODAL_CACHE_VOLUME_NAME = "temporal-finance-kronos-cache"
DEFAULT_KRONOS_MODAL_CACHE_DIR = "/cache/huggingface"
DEFAULT_KRONOS_MODAL_OUTPUT_DIR = "/tmp/kronos_checkpoint4"
DEFAULT_KRONOS_MODAL_DATA_DIR = "/root/dataset/data/kronos/by_ticker"
DEFAULT_KRONOS_MODAL_DATA_ROOT = "/root/dataset/data/kronos"
KRONOS_REPO_PATH = "/opt/Kronos"
KRONOS_REPO_URL = "https://github.com/shiyu-coder/Kronos.git"


def build_kronos_modal_run_config(
    config: KronosCheckpoint4Config,
    remote_cache_dir: str = DEFAULT_KRONOS_MODAL_CACHE_DIR,
    remote_output_dir: str = DEFAULT_KRONOS_MODAL_OUTPUT_DIR,
    remote_data_dir: str = DEFAULT_KRONOS_MODAL_DATA_DIR,
) -> KronosCheckpoint4Config:
    replacements = {
        "kronos_cache_dir": remote_cache_dir,
        "output_dir": remote_output_dir,
        "device": "gpu",
    }
    if config.data_source == "local_kronos_csv":
        replacements["local_data_dir"] = _map_local_kronos_data_dir(
            config.local_data_dir,
            remote_data_dir=remote_data_dir,
        )
    modal_config = replace(
        config,
        **replacements,
    )
    modal_config.validate()
    return modal_config


def _map_local_kronos_data_dir(
    local_data_dir: str | None,
    remote_data_dir: str = DEFAULT_KRONOS_MODAL_DATA_DIR,
) -> str:
    if not local_data_dir:
        return remote_data_dir
    normalized = PurePosixPath(str(local_data_dir).replace("\\", "/"))
    marker = PurePosixPath("dataset/data/kronos")
    parts = normalized.parts
    marker_parts = marker.parts
    for index in range(0, len(parts) - len(marker_parts) + 1):
        if parts[index : index + len(marker_parts)] == marker_parts:
            relative = PurePosixPath(*parts[index + len(marker_parts) :])
            if str(relative) == ".":
                return DEFAULT_KRONOS_MODAL_DATA_ROOT
            return str(PurePosixPath(DEFAULT_KRONOS_MODAL_DATA_ROOT) / relative)
    return remote_data_dir
