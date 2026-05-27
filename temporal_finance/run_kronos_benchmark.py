"""CLI entrypoint for manifest-driven Kronos benchmark sweeps."""

import argparse
import json
from pathlib import Path

from temporal_finance.kronos_benchmark import (
    load_promoted_run_names,
    run_benchmark_manifest_local,
    write_benchmark_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a manifest-driven Kronos benchmark locally."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="YAML or CSV benchmark manifest.",
    )
    parser.add_argument(
        "--only-promoted-from",
        default="",
        help="Optional benchmark_summary.csv or promoted_run_names.txt to filter runs.",
    )
    args = parser.parse_args()

    only_run_names = None
    if args.only_promoted_from:
        only_run_names = load_promoted_run_names(args.only_promoted_from)

    records = run_benchmark_manifest_local(
        manifest_path=args.manifest,
        only_run_names=only_run_names,
    )
    output_root = Path(records[0].output_dir).parent if records else Path(".")
    paths = write_benchmark_outputs(records, output_root)
    print(
        json.dumps(
            {
                "record_count": len(records),
                "outputs": {key: str(value) for key, value in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
