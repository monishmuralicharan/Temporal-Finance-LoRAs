"""Modal orchestration for the Temporal LoRA signal run."""

import argparse
import json
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Dict

import modal

from temporal_finance.kronos_modal_utils import (
    DEFAULT_KRONOS_MODAL_CACHE_DIR,
    DEFAULT_KRONOS_MODAL_CACHE_VOLUME_NAME,
    DEFAULT_KRONOS_MODAL_DATA_ROOT,
    KRONOS_REPO_PATH,
    KRONOS_REPO_URL,
    KRONOS_UPSTREAM_COMMIT,
)
from temporal_finance.temporal_lora_config import (
    TemporalLoraSignalConfig,
    load_temporal_lora_signal_config,
)
from temporal_finance.temporal_lora_data import resolve_tickers
from temporal_finance.temporal_lora_signal import (
    run_temporal_lora_signal,
    serialize_temporal_lora_artifacts,
    write_temporal_lora_artifacts,
)


DEFAULT_TEMPORAL_LORA_MODAL_APP_NAME = "temporal-finance-temporal-lora-signal"
DEFAULT_TEMPORAL_LORA_MODAL_GPU = "L4"
DEFAULT_TEMPORAL_LORA_REMOTE_OUTPUT_DIR = "/tmp/temporal_lora_signal_run"

KRONOS_CACHE_VOLUME = modal.Volume.from_name(
    DEFAULT_KRONOS_MODAL_CACHE_VOLUME_NAME,
    create_if_missing=True,
)
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install(
        "torch==2.5.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "einops==0.8.1",
        "huggingface_hub[hf_xet]==0.33.1",
        "safetensors==0.6.2",
        "tqdm==4.67.1",
    )
    .run_commands(
        "git clone {0} {1} && cd {1} && git checkout {2}".format(
            KRONOS_REPO_URL,
            KRONOS_REPO_PATH,
            KRONOS_UPSTREAM_COMMIT,
        )
    )
    .env(
        {
            "PYTHONPATH": KRONOS_REPO_PATH,
            "HF_HOME": DEFAULT_KRONOS_MODAL_CACHE_DIR,
        }
    )
    .add_local_dir(
        "dataset/data/kronos",
        remote_path="/root/dataset/data/kronos",
        copy=True,
    )
    .add_local_python_source("temporal_finance")
)
APP = modal.App(DEFAULT_TEMPORAL_LORA_MODAL_APP_NAME, image=IMAGE)


@APP.cls(
    gpu=DEFAULT_TEMPORAL_LORA_MODAL_GPU,
    volumes={Path(DEFAULT_KRONOS_MODAL_CACHE_DIR): KRONOS_CACHE_VOLUME},
    timeout=24 * 60 * 60,
    scaledown_window=300,
)
class RemoteTemporalLoraSignalRunner:
    config_json: str = modal.parameter()

    @modal.enter()
    def load(self):
        os.environ["HF_HOME"] = DEFAULT_KRONOS_MODAL_CACHE_DIR
        self._config = TemporalLoraSignalConfig.from_dict(json.loads(self.config_json))
        self._config.validate()
        KRONOS_CACHE_VOLUME.commit()

    @modal.method()
    def run(self) -> dict:
        run_config = replace(
            self._config,
            output_dir=tempfile.mkdtemp(prefix="temporal-lora-signal-"),
        )
        result = run_temporal_lora_signal(
            run_config,
            progress_callback=lambda message: print(message, flush=True),
        )
        return {
            "artifacts": serialize_temporal_lora_artifacts(result.output_dir),
            "decision": result.decision,
        }


@APP.function(
    image=IMAGE,
    volumes={Path(DEFAULT_KRONOS_MODAL_CACHE_DIR): KRONOS_CACHE_VOLUME},
    timeout=7200,
    scaledown_window=60,
)
def download_temporal_lora_models(config_json: str) -> str:
    from model import Kronos, KronosTokenizer

    os.environ["HF_HOME"] = DEFAULT_KRONOS_MODAL_CACHE_DIR
    config = TemporalLoraSignalConfig.from_dict(json.loads(config_json))
    config.validate()
    kwargs = {"cache_dir": DEFAULT_KRONOS_MODAL_CACHE_DIR}
    KronosTokenizer.from_pretrained(config.tokenizer_id, **kwargs)
    Kronos.from_pretrained(config.model_id, **kwargs)
    KRONOS_CACHE_VOLUME.commit()
    return (
        "Downloaded {0} and {1} into Modal Volume {2}:{3}.".format(
            config.tokenizer_id,
            config.model_id,
            DEFAULT_KRONOS_MODAL_CACHE_VOLUME_NAME,
            DEFAULT_KRONOS_MODAL_CACHE_DIR,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Temporal LoRA signal experiment on Modal."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download-models",
        help="Download Kronos-small model/tokenizer into the shared Modal cache.",
    )
    download_parser.add_argument(
        "--config",
        default="configs/temporal_lora_signal_run.yaml",
        help="Temporal LoRA signal-run config.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run the Temporal LoRA signal experiment remotely.",
    )
    run_parser.add_argument(
        "--config",
        default="configs/temporal_lora_signal_run.yaml",
        help="Temporal LoRA signal-run config.",
    )
    run_parser.add_argument(
        "--gpu",
        default=DEFAULT_TEMPORAL_LORA_MODAL_GPU,
        help="Modal GPU type, for example L4 or T4.",
    )
    run_parser.add_argument(
        "--detach",
        action="store_true",
        help="Submit the run and write detached state for later collection.",
    )

    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect a detached Temporal LoRA signal run.",
    )
    collect_parser.add_argument(
        "--state",
        required=True,
        help="Path to detached_state.json written by run --detach.",
    )

    args = parser.parse_args()
    if args.command == "download-models":
        _download_models(args.config)
        return
    if args.command == "run":
        _run_remote(args.config, args.gpu, detach=args.detach)
        return
    if args.command == "collect":
        _collect_detached(args.state)
        return
    parser.error("Unknown command: {0}".format(args.command))


def _download_models(config_path: str) -> None:
    base_config = load_temporal_lora_signal_config(config_path)
    remote_config = build_modal_temporal_lora_config(base_config)
    config_json = json.dumps(remote_config.to_dict(), sort_keys=True)
    with modal.enable_output():
        with APP.run():
            message = download_temporal_lora_models.remote(config_json)
    print(message)


def _run_remote(config_path: str, gpu: str, detach: bool = False) -> None:
    base_config = load_temporal_lora_signal_config(config_path)
    remote_config = build_modal_temporal_lora_config(base_config)
    config_json = json.dumps(remote_config.to_dict(), sort_keys=True)
    runner_cls = _runner_cls_for_gpu(gpu)
    if detach:
        _run_detached(base_config, config_path, config_json, gpu, runner_cls)
        return
    with modal.enable_output():
        with APP.run():
            runner = runner_cls(config_json=config_json)
            payload = runner.run.remote()
    paths = write_temporal_lora_artifacts(
        base_config.resolved_output_dir,
        payload["artifacts"],
    )
    print("Saved Temporal LoRA signal artifacts to {0}".format(base_config.resolved_output_dir))
    print(json.dumps({"outputs": {k: str(v) for k, v in paths.items()}}, indent=2))
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


def _run_detached(
    base_config: TemporalLoraSignalConfig,
    config_path: str,
    config_json: str,
    gpu: str,
    runner_cls,
) -> None:
    output_root = base_config.resolved_output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    with modal.enable_output():
        with APP.run(detach=True):
            runner = runner_cls(config_json=config_json)
            call = runner.run.spawn()
            state = _build_detached_temporal_lora_state(
                config_path=config_path,
                output_root=str(output_root),
                gpu=gpu,
                function_call_id=_function_call_id(call),
                function_call_dashboard_url=_safe_call_dashboard_url(call),
                submitted_at_unix=time.time(),
            )
            state_path = _write_detached_temporal_lora_state(output_root, state)
    print("Submitted detached Temporal LoRA signal run.")
    print("Saved detached state to {0}".format(state_path))
    print(
        "Collect with: arch -arm64 ./.venv/bin/python -m "
        "temporal_finance.modal_temporal_lora_signal collect --state {0}".format(
            state_path
        )
    )


def _collect_detached(state_path: str) -> None:
    path = Path(state_path)
    state = _load_detached_temporal_lora_state(path)
    call = modal.FunctionCall.from_id(str(state["function_call_id"]))
    payload = call.get()
    paths = _collect_temporal_lora_artifacts(
        state=state,
        payload=payload,
        state_path=path,
    )
    print("Saved Temporal LoRA signal artifacts to {0}".format(state["output_root"]))
    print(json.dumps({"outputs": {k: str(v) for k, v in paths.items()}}, indent=2))
    print(json.dumps(payload.get("decision", {}), indent=2, sort_keys=True))


def build_modal_temporal_lora_config(
    config: TemporalLoraSignalConfig,
    remote_cache_dir: str = DEFAULT_KRONOS_MODAL_CACHE_DIR,
    remote_output_dir: str = DEFAULT_TEMPORAL_LORA_REMOTE_OUTPUT_DIR,
) -> TemporalLoraSignalConfig:
    tickers = config.tickers
    universe_file = config.universe_file
    if tickers is None and universe_file and Path(universe_file).exists():
        tickers = resolve_tickers(config)
        universe_file = None
    return replace(
        config,
        data_dir=_map_local_kronos_data_dir(config.data_dir),
        output_dir=remote_output_dir,
        kronos_cache_dir=remote_cache_dir,
        device="gpu",
        tickers=tickers,
        universe_file=universe_file,
    )


def _map_local_kronos_data_dir(local_data_dir: str) -> str:
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
    return str(PurePosixPath(DEFAULT_KRONOS_MODAL_DATA_ROOT) / "combined_by_ticker")


def _runner_cls_for_gpu(gpu: str):
    if gpu != DEFAULT_TEMPORAL_LORA_MODAL_GPU:
        return RemoteTemporalLoraSignalRunner.with_options(gpu=gpu)
    return RemoteTemporalLoraSignalRunner


def _function_call_id(call) -> str:
    call_id = getattr(call, "object_id", None) or getattr(
        call,
        "function_call_id",
        None,
    )
    if not call_id:
        raise RuntimeError("Detached Modal call did not expose a function call id.")
    return str(call_id)


def _safe_call_dashboard_url(call):
    try:
        return call.get_dashboard_url()
    except Exception:
        return None


def _build_detached_temporal_lora_state(
    config_path: str,
    output_root: str,
    gpu: str,
    function_call_id: str,
    function_call_dashboard_url,
    submitted_at_unix: float,
) -> Dict[str, object]:
    return {
        "version": 1,
        "status": "submitted",
        "config_path": config_path,
        "output_root": output_root,
        "gpu": gpu,
        "function_call_id": function_call_id,
        "function_call_dashboard_url": function_call_dashboard_url,
        "submitted_at_unix": submitted_at_unix,
    }


def _write_detached_temporal_lora_state(output_root: Path, state: dict) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "detached_state.json"
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _load_detached_temporal_lora_state(path: Path) -> dict:
    state = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "version",
        "config_path",
        "output_root",
        "gpu",
        "function_call_id",
        "submitted_at_unix",
    }
    missing = sorted(required - set(state))
    if missing:
        raise SystemExit(
            "Detached Temporal LoRA state is missing keys: {0}".format(
                ", ".join(missing)
            )
        )
    if int(state["version"]) != 1:
        raise SystemExit(
            "Unsupported detached Temporal LoRA state version: {0}".format(
                state["version"]
            )
        )
    return state


def _collect_temporal_lora_artifacts(
    state: dict,
    payload: dict,
    state_path: Path | None = None,
    collected_at_unix: float | None = None,
) -> Dict[str, Path]:
    collected_at = time.time() if collected_at_unix is None else collected_at_unix
    paths = write_temporal_lora_artifacts(
        state["output_root"],
        payload["artifacts"],
    )
    if state_path is not None:
        updated = dict(state)
        updated["status"] = "collected"
        updated["collected_at_unix"] = collected_at
        updated["runtime_seconds"] = collected_at - float(state["submitted_at_unix"])
        updated["decision"] = payload.get("decision", {})
        updated["outputs"] = {key: str(value) for key, value in paths.items()}
        state_path.write_text(
            json.dumps(updated, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return paths


if __name__ == "__main__":
    main()
