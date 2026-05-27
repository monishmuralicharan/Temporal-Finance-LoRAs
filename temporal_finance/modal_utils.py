"""Pure helpers for Modal artifact transfer."""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from temporal_finance.kronos_pipeline import KronosCheckpoint4RunResult


def serialize_run_artifacts(result: "KronosCheckpoint4RunResult") -> Dict[str, str]:
    return {
        "predictions_csv": result.predictions_path.read_text(encoding="utf-8"),
        "metrics_json": result.metrics_path.read_text(encoding="utf-8"),
        "per_ticker_metrics_csv": result.per_ticker_metrics_path.read_text(
            encoding="utf-8"
        ),
    }


def write_local_artifacts(
    output_dir: Path,
    predictions_csv: str,
    metrics_json: str,
    per_ticker_metrics_csv: str = "",
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    per_ticker_metrics_path = output_dir / "per_ticker_metrics.csv"
    predictions_path.write_text(predictions_csv, encoding="utf-8")
    metrics_path.write_text(metrics_json, encoding="utf-8")
    per_ticker_metrics_path.write_text(per_ticker_metrics_csv, encoding="utf-8")
    return predictions_path, metrics_path, per_ticker_metrics_path
