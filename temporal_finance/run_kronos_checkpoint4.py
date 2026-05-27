"""CLI entrypoint for the checkpoint 4 Kronos baseline."""

import argparse
import json

from temporal_finance.kronos_config import load_kronos_checkpoint4_config
from temporal_finance.kronos_pipeline import run_kronos_checkpoint4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the checkpoint 4 Kronos baseline."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config for the Kronos checkpoint 4 run.",
    )
    args = parser.parse_args()

    config = load_kronos_checkpoint4_config(args.config)
    result = run_kronos_checkpoint4(config)

    print("Saved predictions to {0}".format(result.predictions_path))
    print("Saved metrics to {0}".format(result.metrics_path))
    print("Saved per-ticker metrics to {0}".format(result.per_ticker_metrics_path))
    print(json.dumps(result.metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
