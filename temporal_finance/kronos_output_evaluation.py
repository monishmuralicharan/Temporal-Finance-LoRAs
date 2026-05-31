"""Generic evaluation for Kronos-style prediction outputs."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from temporal_finance.evaluation import compute_metrics, return_column_names


@dataclass
class KronosOutputEvaluationResult:
    metrics_path: Path
    daily_metrics_path: Optional[Path]
    group_metrics_path: Optional[Path]
    metrics: Dict[str, object]
    daily_metrics: pd.DataFrame
    group_metrics: pd.DataFrame


def evaluate_kronos_output(
    predictions: pd.DataFrame,
    min_assets_per_date: int = 2,
    group_columns: Optional[List[str]] = None,
) -> Dict[str, object]:
    frame = _normalize_predictions(predictions)
    metrics = {
        "row_count": int(len(frame)),
        "ticker_count": _unique_count(frame, "ticker"),
        "date_count": _unique_count(frame, "date"),
        "target_date_count": _unique_count(frame, "target_date"),
        "paired_series": _compute_paired_series_metrics(frame),
        "ohlc_validity": _compute_ohlc_validity(frame),
    }
    return_columns = _maybe_return_columns(frame)
    if return_columns is not None:
        pred_column, actual_column = return_columns
        return_frame = frame.dropna(subset=[pred_column, actual_column])
        if not return_frame.empty:
            metrics["return_metrics"] = compute_metrics(return_frame)
            metrics["daily_cross_section"] = _summarize_daily_metrics(
                compute_daily_cross_section_metrics(
                    return_frame,
                    min_assets_per_date=min_assets_per_date,
                )
            )
    if group_columns:
        group_metrics = compute_group_metrics(frame, group_columns)
        metrics["group_metrics"] = group_metrics.to_dict(orient="records")
    return metrics


def compute_daily_cross_section_metrics(
    predictions: pd.DataFrame,
    min_assets_per_date: int = 2,
) -> pd.DataFrame:
    required = {"ticker", "date"}
    if not required.issubset(predictions.columns):
        return pd.DataFrame()
    return_columns = _maybe_return_columns(predictions)
    if return_columns is None:
        return pd.DataFrame()
    pred_column, actual_column = return_columns
    rows = []
    for date, group in predictions.groupby("date", sort=True):
        group = group.dropna(subset=[pred_column, actual_column])
        if group["ticker"].nunique() < min_assets_per_date:
            continue
        pred = group[pred_column].to_numpy(dtype=float)
        actual = group[actual_column].to_numpy(dtype=float)
        rows.append(
            {
                "date": date,
                "asset_count": int(group["ticker"].nunique()),
                "ic": _safe_corr(pred, actual, method="pearson"),
                "rank_ic": _safe_corr(pred, actual, method="spearman"),
                "directional_accuracy": float(
                    np.mean(np.sign(pred) == np.sign(actual))
                ),
                "pred_positive_rate": float(np.mean(pred > 0)),
                "actual_positive_rate": float(np.mean(actual > 0)),
            }
        )
    return pd.DataFrame(rows)


def compute_group_metrics(
    predictions: pd.DataFrame,
    group_columns: List[str],
) -> pd.DataFrame:
    available = [column for column in group_columns if column in predictions.columns]
    if not available:
        return pd.DataFrame()
    rows = []
    for keys, group in predictions.groupby(available, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(available, keys))
        row["row_count"] = int(len(group))
        return_columns = _maybe_return_columns(group)
        if return_columns is not None:
            pred_column, actual_column = return_columns
            clean = group.dropna(subset=[pred_column, actual_column])
            if not clean.empty:
                group_return_metrics = compute_metrics(clean)
                row.update(
                    {
                        "return_mae": group_return_metrics["mae"],
                        "directional_accuracy": group_return_metrics[
                            "directional_accuracy"
                        ],
                        "pearson_ic": group_return_metrics["pearson_ic"],
                        "spearman_rank_ic": group_return_metrics[
                            "spearman_rank_ic"
                        ],
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def run_kronos_output_evaluation(
    predictions_path: str,
    output_dir: str,
    min_assets_per_date: int = 2,
    group_columns: Optional[List[str]] = None,
) -> KronosOutputEvaluationResult:
    predictions = pd.read_csv(predictions_path)
    output_path = Path(output_dir).resolve()
    metrics = evaluate_kronos_output(
        predictions,
        min_assets_per_date=min_assets_per_date,
        group_columns=group_columns,
    )
    daily_metrics = pd.DataFrame()
    if _maybe_return_columns(predictions) is not None:
        daily_metrics = compute_daily_cross_section_metrics(
            _normalize_predictions(predictions),
            min_assets_per_date=min_assets_per_date,
        )
    group_metrics = pd.DataFrame()
    if group_columns:
        group_metrics = compute_group_metrics(
            _normalize_predictions(predictions),
            group_columns,
        )
    paths = _write_outputs(output_path, metrics, daily_metrics, group_metrics)
    return KronosOutputEvaluationResult(
        metrics_path=paths["metrics"],
        daily_metrics_path=paths.get("daily_metrics"),
        group_metrics_path=paths.get("group_metrics"),
        metrics=metrics,
        daily_metrics=daily_metrics,
        group_metrics=group_metrics,
    )


def _maybe_return_columns(predictions: pd.DataFrame):
    try:
        return return_column_names(predictions)
    except KeyError:
        return None


def _normalize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    for column in frame.columns:
        if column.startswith("pred_") or column.startswith("actual_"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("date", "target_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d")
    return frame.replace([np.inf, -np.inf], np.nan)


def _compute_paired_series_metrics(frame: pd.DataFrame) -> Dict[str, object]:
    metrics = {}
    for series_name in _discover_paired_series(frame.columns):
        pred_column = "pred_{0}".format(series_name)
        actual_column = "actual_{0}".format(series_name)
        metrics[series_name] = _paired_metrics(
            frame[pred_column],
            frame[actual_column],
        )
    return metrics


def _discover_paired_series(columns: Iterable[str]) -> List[str]:
    column_set = set(columns)
    series_names = []
    for column in sorted(column_set):
        if not column.startswith("pred_"):
            continue
        series_name = column[len("pred_") :]
        if "actual_{0}".format(series_name) in column_set:
            series_names.append(series_name)
    return series_names


def _paired_metrics(pred: pd.Series, actual: pd.Series) -> Dict[str, object]:
    frame = pd.DataFrame({"pred": pred, "actual": actual}).dropna()
    if frame.empty:
        return {"sample_count": 0}
    pred_values = frame["pred"].to_numpy(dtype=float)
    actual_values = frame["actual"].to_numpy(dtype=float)
    error = pred_values - actual_values
    abs_actual = np.abs(actual_values)
    nonzero = abs_actual > 1e-12
    mape = None
    if np.any(nonzero):
        mape = float(np.mean(np.abs(error[nonzero]) / abs_actual[nonzero]))
    return {
        "sample_count": int(len(frame)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mape": mape,
        "normalized_mae": _safe_divide(
            float(np.mean(np.abs(error))), float(np.mean(abs_actual))
        ),
        "r2": _r2_score(pred_values, actual_values),
        "pearson_ic": _safe_corr(pred_values, actual_values, method="pearson"),
        "spearman_rank_ic": _safe_corr(
            pred_values, actual_values, method="spearman"
        ),
    }


def _compute_ohlc_validity(frame: pd.DataFrame) -> Dict[str, object]:
    rows = []
    suffixes = _discover_ohlc_suffixes(frame.columns)
    for suffix in suffixes:
        open_column = "pred_open{0}".format(suffix)
        high_column = "pred_high{0}".format(suffix)
        low_column = "pred_low{0}".format(suffix)
        close_column = "pred_close{0}".format(suffix)
        values = frame[[open_column, high_column, low_column, close_column]].dropna()
        if values.empty:
            continue
        high = values[high_column].to_numpy(dtype=float)
        low = values[low_column].to_numpy(dtype=float)
        open_ = values[open_column].to_numpy(dtype=float)
        close = values[close_column].to_numpy(dtype=float)
        invalid = (
            (high < np.maximum(open_, close))
            | (low > np.minimum(open_, close))
            | (low > high)
        )
        rows.append(
            {
                "suffix": suffix.lstrip("_") or "current",
                "sample_count": int(len(values)),
                "invalid_count": int(np.sum(invalid)),
                "invalid_rate": float(np.mean(invalid)),
                "negative_volume_count": _count_negative_volume(frame, suffix),
            }
        )
    total_samples = sum(row["sample_count"] for row in rows)
    total_invalid = sum(row["invalid_count"] for row in rows)
    total_negative_volume = sum(row["negative_volume_count"] for row in rows)
    return {
        "total_sample_count": int(total_samples),
        "total_invalid_count": int(total_invalid),
        "total_invalid_rate": _safe_divide(total_invalid, total_samples),
        "total_negative_volume_count": int(total_negative_volume),
        "by_step": rows,
    }


def _discover_ohlc_suffixes(columns: Iterable[str]) -> List[str]:
    column_set = set(columns)
    suffixes = []
    pattern = re.compile(r"^pred_open(?P<suffix>(?:_t\d+)?)$")
    for column in column_set:
        match = pattern.match(column)
        if not match:
            continue
        suffix = match.group("suffix")
        required = {
            "pred_open{0}".format(suffix),
            "pred_high{0}".format(suffix),
            "pred_low{0}".format(suffix),
            "pred_close{0}".format(suffix),
        }
        if required.issubset(column_set):
            suffixes.append(suffix)
    return sorted(suffixes)


def _count_negative_volume(frame: pd.DataFrame, suffix: str) -> int:
    volume_column = "pred_volume{0}".format(suffix)
    if volume_column not in frame.columns:
        return 0
    volume = pd.to_numeric(frame[volume_column], errors="coerce").dropna()
    if volume.empty:
        return 0
    return int(np.sum(volume.to_numpy(dtype=float) < 0))


def _summarize_daily_metrics(daily_metrics: pd.DataFrame) -> Dict[str, object]:
    if daily_metrics.empty:
        return {"window_count": 0}
    return {
        "window_count": int(len(daily_metrics)),
        "mean_ic": _mean_or_none(daily_metrics["ic"]),
        "mean_rank_ic": _mean_or_none(daily_metrics["rank_ic"]),
        "ic_positive_rate": _positive_rate_or_none(daily_metrics["ic"]),
        "rank_ic_positive_rate": _positive_rate_or_none(
            daily_metrics["rank_ic"]
        ),
        "mean_directional_accuracy": _mean_or_none(
            daily_metrics["directional_accuracy"]
        ),
    }


def _write_outputs(
    output_dir: Path,
    metrics: Dict[str, object],
    daily_metrics: pd.DataFrame,
    group_metrics: pd.DataFrame,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"metrics": output_dir / "metrics.json"}
    paths["metrics"].write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not daily_metrics.empty:
        paths["daily_metrics"] = output_dir / "daily_cross_section_metrics.csv"
        daily_metrics.to_csv(paths["daily_metrics"], index=False)
    if not group_metrics.empty:
        paths["group_metrics"] = output_dir / "group_metrics.csv"
        group_metrics.to_csv(paths["group_metrics"], index=False)
    return paths


def _safe_corr(
    pred: np.ndarray,
    actual: np.ndarray,
    method: str,
) -> Optional[float]:
    if len(pred) < 2:
        return None
    if np.std(pred) == 0 or np.std(actual) == 0:
        return None
    if method == "pearson":
        value = stats.pearsonr(pred, actual)[0]
    elif method == "spearman":
        value = stats.spearmanr(pred, actual)[0]
    else:
        raise ValueError("Unsupported correlation method: {0}".format(method))
    if np.isnan(value):
        return None
    return float(value)


def _r2_score(pred: np.ndarray, actual: np.ndarray) -> Optional[float]:
    if len(pred) < 2:
        return None
    total = float(np.sum(np.square(actual - np.mean(actual))))
    if total == 0:
        return None
    residual = float(np.sum(np.square(actual - pred)))
    return 1.0 - (residual / total)


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    value = numerator / denominator
    if not np.isfinite(value):
        return None
    return float(value)


def _mean_or_none(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _positive_rate_or_none(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(np.mean(values > 0))


def _unique_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].nunique())
