"""CLI entrypoint for quick Kronos output grading."""

import argparse
import json

from temporal_finance.kronos_output_evaluation import run_kronos_output_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a Kronos-style predictions.csv file."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to a Kronos-style predictions.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for evaluation metrics.",
    )
    parser.add_argument(
        "--min-assets-per-date",
        type=int,
        default=2,
        help="Minimum assets needed for daily IC/RankIC.",
    )
    parser.add_argument(
        "--group-by",
        default="",
        help="Optional comma-separated columns for per-group metrics.",
    )
    args = parser.parse_args()

    group_columns = [
        column.strip()
        for column in args.group_by.split(",")
        if column.strip()
    ]
    result = run_kronos_output_evaluation(
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        min_assets_per_date=args.min_assets_per_date,
        group_columns=group_columns or None,
    )

    print("Saved metrics to {0}".format(result.metrics_path))
    if result.daily_metrics_path is not None:
        print(
            "Saved daily cross-section metrics to {0}".format(
                result.daily_metrics_path
            )
        )
    if result.group_metrics_path is not None:
        print("Saved group metrics to {0}".format(result.group_metrics_path))
    print(json.dumps(result.metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
