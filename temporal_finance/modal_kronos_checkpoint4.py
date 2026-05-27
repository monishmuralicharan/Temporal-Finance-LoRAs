"""Modal orchestration for the remote checkpoint 4 Kronos baseline."""

import argparse
import json
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import modal

from temporal_finance.kronos_benchmark import (
    build_benchmark_config,
    build_failed_benchmark_record,
    load_benchmark_references,
    load_kronos_benchmark_manifest,
    load_promoted_run_names,
    materialize_remote_benchmark_artifacts,
    selected_calibration_modes,
    write_benchmark_outputs,
)
from temporal_finance.kronos_config import (
    KronosCheckpoint4Config,
    load_kronos_checkpoint4_config,
)
from temporal_finance.kronos_experiments import (
    build_experiment_config,
    build_experiment_record,
    flatten_kronos_output_evaluation_metrics,
    kronos_experiment_definitions_for_suite,
    select_best_kronos_experiment,
    should_skip_kronos_experiment,
    write_kronos_experiment_outputs,
)
from temporal_finance.kronos_output_evaluation import run_kronos_output_evaluation
from temporal_finance.kronos_modal_utils import (
    DEFAULT_KRONOS_MODAL_APP_NAME,
    DEFAULT_KRONOS_MODAL_CACHE_DIR,
    DEFAULT_KRONOS_MODAL_CACHE_VOLUME_NAME,
    DEFAULT_KRONOS_MODAL_GPU,
    KRONOS_REPO_PATH,
    KRONOS_REPO_URL,
    KRONOS_UPSTREAM_COMMIT,
    build_kronos_modal_run_config,
)
from temporal_finance.modal_utils import serialize_run_artifacts, write_local_artifacts

KRONOS_CACHE_VOLUME = modal.Volume.from_name(
    DEFAULT_KRONOS_MODAL_CACHE_VOLUME_NAME, create_if_missing=True
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
    .add_local_python_source("temporal_finance")
)
DOWNLOAD_IMAGE = IMAGE
APP = modal.App(DEFAULT_KRONOS_MODAL_APP_NAME, image=IMAGE)


@APP.cls(
    gpu=DEFAULT_KRONOS_MODAL_GPU,
    volumes={Path(DEFAULT_KRONOS_MODAL_CACHE_DIR): KRONOS_CACHE_VOLUME},
    timeout=7200,
    scaledown_window=300,
)
class RemoteKronosCheckpoint4Runner:
    config_json: str = modal.parameter()

    @modal.enter()
    def load(self):
        from temporal_finance.kronos import KronosForecaster

        os.environ["HF_HOME"] = DEFAULT_KRONOS_MODAL_CACHE_DIR
        payload = json.loads(self.config_json)
        self._base_config = KronosCheckpoint4Config.from_dict(payload)
        self._base_config.validate()
        self._forecaster = KronosForecaster(self._base_config)
        KRONOS_CACHE_VOLUME.commit()

    @modal.method()
    def run(self):
        from temporal_finance.kronos_pipeline import run_kronos_checkpoint4

        run_config = replace(
            self._base_config,
            output_dir=tempfile.mkdtemp(prefix="kronos-checkpoint4-"),
        )
        result = run_kronos_checkpoint4(
            run_config,
            forecaster=self._forecaster,
            progress_callback=lambda message: print(message, flush=True),
        )
        return serialize_run_artifacts(result)


@APP.function(
    image=DOWNLOAD_IMAGE,
    volumes={Path(DEFAULT_KRONOS_MODAL_CACHE_DIR): KRONOS_CACHE_VOLUME},
    timeout=7200,
    scaledown_window=60,
)
def download_kronos_models(config_json: str) -> str:
    from model import Kronos, KronosTokenizer

    os.environ["HF_HOME"] = DEFAULT_KRONOS_MODAL_CACHE_DIR
    config = KronosCheckpoint4Config.from_dict(json.loads(config_json))
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
        description="Manage Kronos models on Modal or run checkpoint 4 remotely."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download-models",
        help="Download the Kronos model/tokenizer directly into a Modal Volume.",
    )
    download_parser.add_argument(
        "--config",
        default="configs/kronos_checkpoint4.yaml",
        help="Path to the YAML config containing model_id and tokenizer_id.",
    )

    run_parser = subparsers.add_parser(
        "run", help="Run the checkpoint 4 Kronos baseline remotely on Modal."
    )
    run_parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config for the Kronos checkpoint 4 run.",
    )
    run_parser.add_argument(
        "--gpu",
        default=DEFAULT_KRONOS_MODAL_GPU,
        help="Modal GPU type to request, for example T4 or L4.",
    )

    experiments_parser = subparsers.add_parser(
        "run-experiments",
        help="Run the ordered Kronos fix experiments remotely on Modal.",
    )
    experiments_parser.add_argument(
        "--config",
        default="configs/kronos_checkpoint4_greedy_smoke.yaml",
        help="Base Kronos YAML config. Experiment overrides are applied in order.",
    )
    experiments_parser.add_argument(
        "--output-dir",
        default="outputs/kronos_experiments",
        help="Directory for per-experiment outputs and the summary report.",
    )
    experiments_parser.add_argument(
        "--gpu",
        default=DEFAULT_KRONOS_MODAL_GPU,
        help="Modal GPU type to request, for example T4 or L4.",
    )
    experiments_parser.add_argument(
        "--suite",
        choices=["fix", "paper-proximal"],
        default="fix",
        help="Experiment suite to run.",
    )
    experiments_parser.add_argument(
        "--include-sample20",
        action="store_true",
        help="Include the higher-cost 20-sample paper-proximal variant.",
    )
    benchmark_parser = subparsers.add_parser(
        "run-benchmark",
        help="Run a manifest-driven Kronos benchmark remotely on Modal.",
    )
    benchmark_parser.add_argument(
        "--manifest",
        required=True,
        help="YAML or CSV benchmark manifest.",
    )
    benchmark_parser.add_argument(
        "--gpu",
        default=DEFAULT_KRONOS_MODAL_GPU,
        help="Modal GPU type to request, for example T4 or L4.",
    )
    benchmark_parser.add_argument(
        "--only-promoted-from",
        default="",
        help="Optional benchmark_summary.csv or promoted_run_names.txt to filter runs.",
    )

    args = parser.parse_args()

    if args.command == "download-models":
        _download_models(args.config)
        return
    if args.command == "run":
        _run_remote(args.config, args.gpu)
        return
    if args.command == "run-experiments":
        _run_experiments_remote(
            args.config,
            args.output_dir,
            args.gpu,
            args.suite,
            args.include_sample20,
        )
        return
    if args.command == "run-benchmark":
        _run_benchmark_remote(
            args.manifest,
            args.gpu,
            args.only_promoted_from,
        )
        return
    parser.error("Unknown command: {0}".format(args.command))


def _download_models(config_path: str) -> None:
    config = load_kronos_checkpoint4_config(config_path)
    modal_config = build_kronos_modal_run_config(config)
    config_json = json.dumps(modal_config.to_dict(), sort_keys=True)
    with modal.enable_output():
        with APP.run():
            message = download_kronos_models.remote(config_json)
    print(message)


def _run_remote(config_path: str, gpu: str) -> None:
    base_config = load_kronos_checkpoint4_config(config_path)
    remote_config = build_kronos_modal_run_config(base_config)
    config_json = json.dumps(remote_config.to_dict(), sort_keys=True)

    runner_cls = RemoteKronosCheckpoint4Runner
    if gpu != DEFAULT_KRONOS_MODAL_GPU:
        runner_cls = RemoteKronosCheckpoint4Runner.with_options(gpu=gpu)

    with modal.enable_output():
        with APP.run():
            runner = runner_cls(config_json=config_json)
            artifacts = runner.run.remote()

    predictions_path, metrics_path, per_ticker_metrics_path = write_local_artifacts(
        output_dir=base_config.resolved_output_dir,
        predictions_csv=artifacts["predictions_csv"],
        metrics_json=artifacts["metrics_json"],
        per_ticker_metrics_csv=artifacts["per_ticker_metrics_csv"],
    )
    metrics = json.loads(artifacts["metrics_json"])

    print("Saved predictions to {0}".format(predictions_path))
    print("Saved metrics to {0}".format(metrics_path))
    print("Saved per-ticker metrics to {0}".format(per_ticker_metrics_path))
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _run_experiments_remote(
    config_path: str,
    output_dir: str,
    gpu: str,
    suite: str,
    include_sample20: bool,
) -> None:
    base_config = load_kronos_checkpoint4_config(config_path)
    output_root = Path(output_dir).resolve()
    records = []

    runner_cls = RemoteKronosCheckpoint4Runner
    if gpu != DEFAULT_KRONOS_MODAL_GPU:
        runner_cls = RemoteKronosCheckpoint4Runner.with_options(gpu=gpu)

    definitions = kronos_experiment_definitions_for_suite(
        suite,
        include_sample20=include_sample20,
    )

    with modal.enable_output():
        with APP.run():
            for definition in definitions:
                experiment_config = build_experiment_config(
                    base_config,
                    definition,
                    output_root,
                )
                skip_reason = should_skip_kronos_experiment(definition, records)
                if skip_reason:
                    records.append(
                        build_experiment_record(
                            definition=definition,
                            config=experiment_config,
                            status="skipped",
                            error=skip_reason,
                        )
                    )
                    continue

                try:
                    remote_config = build_kronos_modal_run_config(
                        experiment_config
                    )
                    config_json = json.dumps(
                        remote_config.to_dict(), sort_keys=True
                    )
                    runner = runner_cls(config_json=config_json)
                    artifacts = runner.run.remote()
                    predictions_path, _, _ = write_local_artifacts(
                        output_dir=experiment_config.resolved_output_dir,
                        predictions_csv=artifacts["predictions_csv"],
                        metrics_json=artifacts["metrics_json"],
                        per_ticker_metrics_csv=artifacts[
                            "per_ticker_metrics_csv"
                        ],
                    )
                    evaluation = run_kronos_output_evaluation(
                        predictions_path=str(predictions_path),
                        output_dir=str(
                            experiment_config.resolved_output_dir
                            / "evaluation"
                        ),
                    )
                    metrics = json.loads(artifacts["metrics_json"])
                    metrics.update(
                        flatten_kronos_output_evaluation_metrics(
                            evaluation.metrics
                        )
                    )
                    current_best = select_best_kronos_experiment(records)
                    records.append(
                        build_experiment_record(
                            definition=definition,
                            config=experiment_config,
                            metrics=metrics,
                            reference_metrics=(
                                current_best.metrics
                                if current_best is not None
                                else None
                            ),
                        )
                    )
                except Exception as exc:
                    records.append(
                        build_experiment_record(
                            definition=definition,
                            config=experiment_config,
                            status="failed",
                            error=str(exc),
                        )
                    )

    paths = write_kronos_experiment_outputs(records, output_root)
    print("Saved Kronos experiment summary to {0}".format(paths["summary_csv"]))
    print(
        "Saved Kronos benchmark summary to {0}".format(
            paths["benchmark_summary_csv"]
        )
    )
    if "best_config" in paths:
        print("Saved best config to {0}".format(paths["best_config"]))
    print(json.dumps({"outputs": {k: str(v) for k, v in paths.items()}}, indent=2))


def _run_benchmark_remote(
    manifest_path: str,
    gpu: str,
    only_promoted_from: str = "",
) -> None:
    manifest = load_kronos_benchmark_manifest(manifest_path)
    base_config = load_kronos_checkpoint4_config(manifest.base_config_path)
    output_root = Path(manifest.output_dir).resolve()
    references = load_benchmark_references(manifest.references)
    only_run_names = (
        load_promoted_run_names(only_promoted_from)
        if only_promoted_from
        else None
    )
    records = []

    runner_cls = RemoteKronosCheckpoint4Runner
    if gpu != DEFAULT_KRONOS_MODAL_GPU:
        runner_cls = RemoteKronosCheckpoint4Runner.with_options(gpu=gpu)

    with modal.enable_output():
        with APP.run():
            for experiment in manifest.experiments:
                modes = selected_calibration_modes(experiment, only_run_names)
                if not modes:
                    continue
                experiment_config = build_benchmark_config(
                    base_config,
                    experiment,
                    output_root,
                )
                try:
                    remote_config = build_kronos_modal_run_config(
                        experiment_config
                    )
                    config_json = json.dumps(
                        remote_config.to_dict(), sort_keys=True
                    )
                    print(
                        "Running benchmark experiment {0} ({1})".format(
                            experiment.name,
                            ", ".join(modes),
                        ),
                        flush=True,
                    )
                    start = time.monotonic()
                    runner = runner_cls(config_json=config_json)
                    artifacts = runner.run.remote()
                    runtime_seconds = time.monotonic() - start
                    records.extend(
                        materialize_remote_benchmark_artifacts(
                            manifest=manifest,
                            experiment=experiment,
                            config=experiment_config,
                            predictions_csv=artifacts["predictions_csv"],
                            metrics_json=artifacts["metrics_json"],
                            output_root=output_root,
                            references=references,
                            runtime_seconds=runtime_seconds,
                            modal_gpu=gpu,
                            only_calibration_modes=modes,
                        )
                    )
                except Exception as exc:
                    records.append(
                        build_failed_benchmark_record(
                            manifest=manifest,
                            experiment=experiment,
                            config=experiment_config,
                            output_root=output_root,
                            error=str(exc),
                        )
                    )

    paths = write_benchmark_outputs(records, output_root)
    print(
        "Saved Kronos benchmark summary to {0}".format(
            paths["benchmark_summary_csv"]
        )
    )
    print(json.dumps({"outputs": {k: str(v) for k, v in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
