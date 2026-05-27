"""CLI entrypoint for ordered local Kronos fix experiments."""

import argparse
import json
from pathlib import Path

from temporal_finance.kronos_config import load_kronos_checkpoint4_config
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
from temporal_finance.kronos_pipeline import run_kronos_checkpoint4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ordered Kronos fix experiments locally."
    )
    parser.add_argument(
        "--config",
        default="configs/kronos_checkpoint4_greedy_smoke.yaml",
        help="Base Kronos YAML config. Experiment overrides are applied in order.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/kronos_experiments",
        help="Directory for per-experiment outputs and the summary report.",
    )
    parser.add_argument(
        "--suite",
        choices=["fix", "paper-proximal"],
        default="fix",
        help="Experiment suite to run.",
    )
    parser.add_argument(
        "--include-sample20",
        action="store_true",
        help="Include the higher-cost 20-sample paper-proximal variant.",
    )
    args = parser.parse_args()

    base_config = load_kronos_checkpoint4_config(args.config)
    output_root = Path(args.output_dir).resolve()
    records = []

    definitions = kronos_experiment_definitions_for_suite(
        args.suite,
        include_sample20=args.include_sample20,
    )

    for definition in definitions:
        config = build_experiment_config(base_config, definition, output_root)
        skip_reason = should_skip_kronos_experiment(definition, records)
        if skip_reason:
            records.append(
                build_experiment_record(
                    definition=definition,
                    config=config,
                    status="skipped",
                    error=skip_reason,
                )
            )
            continue

        try:
            result = run_kronos_checkpoint4(config)
            evaluation = run_kronos_output_evaluation(
                predictions_path=str(result.predictions_path),
                output_dir=str(config.resolved_output_dir / "evaluation"),
            )
            metrics = dict(result.metrics)
            metrics.update(
                flatten_kronos_output_evaluation_metrics(evaluation.metrics)
            )
            current_best = select_best_kronos_experiment(records)
            records.append(
                build_experiment_record(
                    definition=definition,
                    config=config,
                    metrics=metrics,
                    reference_metrics=(
                        current_best.metrics if current_best is not None else None
                    ),
                )
            )
        except Exception as exc:
            records.append(
                build_experiment_record(
                    definition=definition,
                    config=config,
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


if __name__ == "__main__":
    main()
