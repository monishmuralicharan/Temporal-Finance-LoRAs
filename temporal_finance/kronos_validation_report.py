"""Validation reporting for the selected Kronos base-model candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from temporal_finance.evaluation import compute_metrics, return_column_names
from temporal_finance.kronos_output_evaluation import (
    compute_daily_cross_section_metrics,
)


CANDIDATE_CALIBRATION = "vol_rescale_mean_center"
REQUIRED_OFFICIAL_BENCHMARKS = {
    "dataset_common100_context512_s10_stride5",
    "dataset_common100_context512_s10_recent_daily",
}
NAIVE_BASELINE_NAMES = (
    "zero_return",
    "prior_5d_momentum",
    "prior_20d_momentum_scaled_to_5d",
    "rolling_20d_mean_daily_return_scaled_to_5d",
)

ACCEPTANCE_THRESHOLDS = {
    "min_return_ic": 0.05,
    "min_return_rank_ic": 0.05,
    "min_directional_accuracy": 0.525,
    "min_daily_ic_positive_rate": 0.52,
    "min_daily_rank_ic_positive_rate": 0.52,
    "min_pred_abs_return_multiple": 0.5,
    "max_pred_abs_return_multiple": 2.0,
    "max_positive_rate_gap": 0.10,
    "max_invalid_ohlc_rate": 0.03,
    "min_rank_ic_gap_to_best_naive": 0.02,
}


def run_validation_report(
    benchmark_dirs: Sequence[str],
    local_data_dir: str,
    output_dir: str,
    candidate_calibration: str = CANDIDATE_CALIBRATION,
    forecast_horizon: int = 5,
    min_assets_per_date: int = 25,
) -> Dict[str, Path]:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    data_root = Path(local_data_dir).resolve()
    history_cache: Dict[str, pd.DataFrame] = {}

    validation_rows = []
    baseline_rows = []
    period_rows = []

    for benchmark_dir in benchmark_dirs:
        benchmark_path = Path(benchmark_dir).resolve()
        benchmark_result = _load_candidate_run(
            benchmark_path,
            candidate_calibration=candidate_calibration,
        )
        if benchmark_result is None:
            validation_rows.append(
                _missing_candidate_row(benchmark_path, candidate_calibration)
            )
            continue

        summary_row, predictions = benchmark_result
        benchmark_name = benchmark_path.name
        kronos_model_name = str(summary_row["name"])
        predictions = _normalize_predictions(predictions)

        model_frames = {
            "kronos_candidate": predictions,
            **build_naive_baseline_predictions(
                predictions=predictions,
                local_data_dir=data_root,
                forecast_horizon=forecast_horizon,
                history_cache=history_cache,
            ),
        }

        model_metrics = {}
        for model_name, model_predictions in model_frames.items():
            metrics = _compute_report_metrics(
                model_predictions,
                min_assets_per_date=min_assets_per_date,
            )
            model_metrics[model_name] = metrics
            baseline_rows.append(
                {
                    "benchmark": benchmark_name,
                    "benchmark_dir": str(benchmark_path),
                    "model": model_name,
                    "run_name": kronos_model_name
                    if model_name == "kronos_candidate"
                    else model_name,
                    **metrics,
                }
            )
            period_rows.extend(
                _compute_period_rows(
                    benchmark_name=benchmark_name,
                    benchmark_path=benchmark_path,
                    model_name=model_name,
                    predictions=model_predictions,
                )
            )

        best_baseline_name, best_baseline_rank_ic = _best_naive_rank_ic(
            model_metrics
        )
        validation_row = _build_validation_row(
            benchmark_path=benchmark_path,
            summary_row=summary_row,
            kronos_metrics=model_metrics["kronos_candidate"],
            best_baseline_name=best_baseline_name,
            best_baseline_rank_ic=best_baseline_rank_ic,
        )
        validation_rows.append(validation_row)

    validation_frame = pd.DataFrame(validation_rows)
    baseline_frame = pd.DataFrame(baseline_rows)
    period_frame = pd.DataFrame(period_rows)
    decision = _build_base_model_decision(validation_frame)

    validation_path = output_path / "validation_summary.csv"
    baseline_path = output_path / "baseline_comparison.csv"
    period_path = output_path / "period_metrics.csv"
    decision_path = output_path / "base_model_decision.json"

    validation_frame.to_csv(validation_path, index=False)
    baseline_frame.to_csv(baseline_path, index=False)
    period_frame.to_csv(period_path, index=False)
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "validation_summary": validation_path,
        "baseline_comparison": baseline_path,
        "period_metrics": period_path,
        "base_model_decision": decision_path,
    }


def build_naive_baseline_predictions(
    predictions: pd.DataFrame,
    local_data_dir: Path,
    forecast_horizon: int = 5,
    history_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    cache = history_cache if history_cache is not None else {}
    baseline_returns = {name: [] for name in NAIVE_BASELINE_NAMES}

    for row in predictions.itertuples(index=False):
        ticker = str(getattr(row, "ticker"))
        origin_date = str(getattr(row, "date"))
        history = cache.get(ticker)
        if history is None:
            history = _load_price_history(local_data_dir, ticker)
            cache[ticker] = history
        baseline_returns["zero_return"].append(0.0)
        baseline_returns["prior_5d_momentum"].append(
            _prior_return(history, origin_date, lookback=forecast_horizon)
        )
        baseline_returns["prior_20d_momentum_scaled_to_5d"].append(
            _scaled_prior_return(
                history,
                origin_date,
                lookback=20,
                forecast_horizon=forecast_horizon,
            )
        )
        baseline_returns["rolling_20d_mean_daily_return_scaled_to_5d"].append(
            _rolling_mean_return(
                history,
                origin_date,
                lookback=20,
                forecast_horizon=forecast_horizon,
            )
        )

    frames = {}
    for name, values in baseline_returns.items():
        frame = predictions.copy()
        frame["pred_return"] = values
        frame["pred_5d_return"] = values
        pred_close = frame["last_context_close"] * (1.0 + frame["pred_return"])
        if "pred_close_target" in frame.columns:
            frame["pred_close_target"] = pred_close
        frame["pred_close_t5"] = pred_close
        frames[name] = frame
    return frames


def _load_candidate_run(
    benchmark_path: Path,
    candidate_calibration: str,
) -> Optional[Tuple[pd.Series, pd.DataFrame]]:
    summary_path = benchmark_path / "benchmark_summary.csv"
    if not summary_path.exists():
        return None
    summary = pd.read_csv(summary_path)
    candidates = summary.loc[
        (summary["status"].astype(str) == "completed")
        & (summary["calibration_mode"].astype(str) == candidate_calibration)
    ]
    if candidates.empty:
        return None
    row = candidates.iloc[0]
    predictions_path = Path(str(row["output_dir"])) / "predictions.csv"
    if not predictions_path.exists():
        return None
    return row, pd.read_csv(predictions_path)


def _normalize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    for column in ("date", "target_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d")
    numeric_columns = [
        "last_context_close",
        "pred_close_target",
        "actual_close_target",
        "pred_close_t5",
        "actual_close_t5",
        "pred_return",
        "actual_return",
        "pred_5d_return",
        "actual_5d_return",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _load_price_history(local_data_dir: Path, ticker: str) -> pd.DataFrame:
    path = local_data_dir / "{0}.csv".format(ticker)
    if not path.exists():
        return pd.DataFrame(columns=["date", "close", "daily_return"])
    frame = pd.read_csv(path, usecols=["timestamps", "close"])
    frame = frame.rename(columns={"timestamps": "date"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    frame["daily_return"] = frame["close"].pct_change()
    return frame.reset_index(drop=True)


def _prior_return(history: pd.DataFrame, origin_date: str, lookback: int) -> float:
    index = _history_index(history, origin_date)
    if index is None or index < lookback:
        return float("nan")
    start = float(history.iloc[index - lookback]["close"])
    end = float(history.iloc[index]["close"])
    if start == 0:
        return float("nan")
    return (end / start) - 1.0


def _scaled_prior_return(
    history: pd.DataFrame,
    origin_date: str,
    lookback: int,
    forecast_horizon: int,
) -> float:
    value = _prior_return(history, origin_date, lookback=lookback)
    if not np.isfinite(value):
        return float("nan")
    if value <= -1.0:
        return value * (forecast_horizon / float(lookback))
    return float((1.0 + value) ** (forecast_horizon / float(lookback)) - 1.0)


def _rolling_mean_return(
    history: pd.DataFrame,
    origin_date: str,
    lookback: int,
    forecast_horizon: int,
) -> float:
    index = _history_index(history, origin_date)
    if index is None or index < lookback:
        return float("nan")
    daily = history.iloc[index - lookback + 1 : index + 1]["daily_return"]
    daily = pd.to_numeric(daily, errors="coerce").dropna()
    if len(daily) < lookback:
        return float("nan")
    mean_daily = float(daily.mean())
    return float((1.0 + mean_daily) ** forecast_horizon - 1.0)


def _history_index(history: pd.DataFrame, origin_date: str) -> Optional[int]:
    if history.empty:
        return None
    matches = history.index[history["date"] == origin_date].tolist()
    if not matches:
        return None
    return int(matches[0])


def _compute_report_metrics(
    predictions: pd.DataFrame,
    min_assets_per_date: int,
) -> Dict[str, Optional[float]]:
    pred_column, actual_column = return_column_names(predictions)
    clean = predictions.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[pred_column, actual_column]
    )
    if clean.empty:
        return {
            "sample_count": 0,
            "pearson_ic": None,
            "spearman_rank_ic": None,
            "directional_accuracy": None,
            "mae": None,
            "mse": None,
            "pred_abs_return_mean": None,
            "actual_abs_return_mean": None,
            "pred_abs_return_multiple": None,
            "pred_positive_rate": None,
            "actual_positive_rate": None,
            "positive_rate_gap": None,
            "daily_window_count": 0,
            "daily_mean_ic": None,
            "daily_mean_rank_ic": None,
            "daily_ic_positive_rate": None,
            "daily_rank_ic_positive_rate": None,
        }
    metrics = compute_metrics(clean)
    pred = clean[pred_column].to_numpy(dtype=float)
    actual = clean[actual_column].to_numpy(dtype=float)
    pred_abs = float(np.mean(np.abs(pred)))
    actual_abs = float(np.mean(np.abs(actual)))
    pred_positive_rate = float(np.mean(pred > 0))
    actual_positive_rate = float(np.mean(actual > 0))
    metrics.update(
        {
            "pred_abs_return_mean": pred_abs,
            "actual_abs_return_mean": actual_abs,
            "pred_abs_return_multiple": _safe_divide(pred_abs, actual_abs),
            "pred_positive_rate": pred_positive_rate,
            "actual_positive_rate": actual_positive_rate,
            "positive_rate_gap": abs(pred_positive_rate - actual_positive_rate),
        }
    )
    daily = compute_daily_cross_section_metrics(
        clean,
        min_assets_per_date=min_assets_per_date,
    )
    metrics.update(_summarize_daily_metrics(daily))
    return metrics


def _summarize_daily_metrics(daily: pd.DataFrame) -> Dict[str, Optional[float]]:
    if daily.empty:
        return {
            "daily_window_count": 0,
            "daily_mean_ic": None,
            "daily_mean_rank_ic": None,
            "daily_ic_positive_rate": None,
            "daily_rank_ic_positive_rate": None,
        }
    ic = pd.to_numeric(daily["ic"], errors="coerce")
    rank_ic = pd.to_numeric(daily["rank_ic"], errors="coerce")
    return {
        "daily_window_count": int(len(daily)),
        "daily_mean_ic": _series_mean(ic),
        "daily_mean_rank_ic": _series_mean(rank_ic),
        "daily_ic_positive_rate": _positive_rate(ic),
        "daily_rank_ic_positive_rate": _positive_rate(rank_ic),
    }


def _compute_period_rows(
    benchmark_name: str,
    benchmark_path: Path,
    model_name: str,
    predictions: pd.DataFrame,
) -> List[Dict[str, object]]:
    pred_column, actual_column = return_column_names(predictions)
    frame = predictions.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[pred_column, actual_column, "date"]
    )
    if frame.empty:
        return []
    frame = frame.copy()
    dates = pd.to_datetime(frame["date"])
    frame["period"] = dates.dt.to_period("Q").astype(str)
    rows = []
    for period, group in frame.groupby("period", sort=True):
        metrics = _compute_report_metrics(group, min_assets_per_date=2)
        rows.append(
            {
                "benchmark": benchmark_name,
                "benchmark_dir": str(benchmark_path),
                "period": period,
                "model": model_name,
                **metrics,
            }
        )
    return rows


def _best_naive_rank_ic(
    model_metrics: Dict[str, Dict[str, Optional[float]]],
) -> Tuple[Optional[str], Optional[float]]:
    best_name = None
    best_value = None
    for name in NAIVE_BASELINE_NAMES:
        value = _as_float(model_metrics.get(name, {}).get("spearman_rank_ic"))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_name = name
            best_value = value
    return best_name, best_value


def _build_validation_row(
    benchmark_path: Path,
    summary_row: pd.Series,
    kronos_metrics: Dict[str, Optional[float]],
    best_baseline_name: Optional[str],
    best_baseline_rank_ic: Optional[float],
) -> Dict[str, object]:
    benchmark_name = benchmark_path.name
    rank_ic = _as_float(kronos_metrics.get("spearman_rank_ic"))
    rank_gap = (
        rank_ic - best_baseline_rank_ic
        if rank_ic is not None and best_baseline_rank_ic is not None
        else None
    )
    row = {
        "benchmark": benchmark_name,
        "benchmark_dir": str(benchmark_path),
        "run_name": str(summary_row["name"]),
        "stage": _series_get(summary_row, "stage"),
        "status": _series_get(summary_row, "status"),
        "calibration_mode": _series_get(summary_row, "calibration_mode"),
        "required_official": benchmark_name in REQUIRED_OFFICIAL_BENCHMARKS,
        "diagnostic": "mixed200" in benchmark_name,
        "passed_existing_gate": _parse_bool(_series_get(summary_row, "passed_gate")),
        "best_naive_baseline": best_baseline_name,
        "best_naive_rank_ic": best_baseline_rank_ic,
        "rank_ic_gap_to_best_naive": rank_gap,
        "paper_reference_return_ic": _as_float(
            _series_get(summary_row, "paper_reference_return_ic")
        ),
        "paper_reference_return_rank_ic": _as_float(
            _series_get(summary_row, "paper_reference_return_rank_ic")
        ),
        "return_ic_gap_to_paper": _as_float(
            _series_get(summary_row, "return_ic_gap_to_paper")
        ),
        "return_rank_ic_gap_to_paper": _as_float(
            _series_get(summary_row, "return_rank_ic_gap_to_paper")
        ),
        "output_total_invalid_ohlc_rate": _as_float(
            _series_get(summary_row, "output_total_invalid_ohlc_rate")
        ),
        **kronos_metrics,
    }
    reasons = _candidate_rejection_reasons(row)
    row["accepted"] = not reasons
    row["rejection_reasons"] = "; ".join(reasons)
    return row


def _missing_candidate_row(
    benchmark_path: Path,
    candidate_calibration: str,
) -> Dict[str, object]:
    return {
        "benchmark": benchmark_path.name,
        "benchmark_dir": str(benchmark_path),
        "run_name": "",
        "stage": "",
        "status": "missing",
        "calibration_mode": candidate_calibration,
        "required_official": benchmark_path.name in REQUIRED_OFFICIAL_BENCHMARKS,
        "diagnostic": "mixed200" in benchmark_path.name,
        "passed_existing_gate": False,
        "accepted": False,
        "rejection_reasons": "candidate predictions not found",
    }


def _candidate_rejection_reasons(row: Dict[str, object]) -> List[str]:
    thresholds = ACCEPTANCE_THRESHOLDS
    reasons = []
    if str(row.get("status")) != "completed":
        reasons.append("benchmark did not complete")
    if not _parse_bool(row.get("passed_existing_gate")):
        reasons.append("existing promotion gate did not pass")
    _require_min(reasons, row, "pearson_ic", thresholds["min_return_ic"])
    _require_min(
        reasons,
        row,
        "spearman_rank_ic",
        thresholds["min_return_rank_ic"],
    )
    _require_min(
        reasons,
        row,
        "directional_accuracy",
        thresholds["min_directional_accuracy"],
    )
    _require_min(
        reasons,
        row,
        "daily_ic_positive_rate",
        thresholds["min_daily_ic_positive_rate"],
    )
    _require_min(
        reasons,
        row,
        "daily_rank_ic_positive_rate",
        thresholds["min_daily_rank_ic_positive_rate"],
    )
    _require_between(
        reasons,
        row,
        "pred_abs_return_multiple",
        thresholds["min_pred_abs_return_multiple"],
        thresholds["max_pred_abs_return_multiple"],
    )
    _require_max(
        reasons,
        row,
        "positive_rate_gap",
        thresholds["max_positive_rate_gap"],
    )
    _require_max(
        reasons,
        row,
        "output_total_invalid_ohlc_rate",
        thresholds["max_invalid_ohlc_rate"],
        strict=True,
    )
    _require_min(
        reasons,
        row,
        "rank_ic_gap_to_best_naive",
        thresholds["min_rank_ic_gap_to_best_naive"],
    )
    return reasons


def _build_base_model_decision(validation_frame: pd.DataFrame) -> Dict[str, object]:
    required = set(REQUIRED_OFFICIAL_BENCHMARKS)
    if validation_frame.empty:
        return {
            "accepted": False,
            "decision": "reject",
            "rejection_reasons": ["no validation rows were produced"],
            "required_benchmarks": sorted(required),
        }
    present_required = set(
        validation_frame.loc[
            validation_frame["required_official"].astype(bool), "benchmark"
        ].astype(str)
    )
    missing_required = sorted(required - present_required)
    blocking_rows = validation_frame.loc[
        validation_frame["required_official"].astype(bool)
        & ~validation_frame["accepted"].astype(bool)
    ]
    reasons = []
    for name in missing_required:
        reasons.append("missing required benchmark: {0}".format(name))
    for row in blocking_rows.to_dict(orient="records"):
        reasons.append(
            "{0}: {1}".format(row["benchmark"], row.get("rejection_reasons") or "")
        )
    accepted = not reasons
    diagnostic_rows = validation_frame.loc[
        validation_frame["diagnostic"].astype(bool)
    ]
    mixed200_accepted = None
    if not diagnostic_rows.empty:
        mixed200_accepted = bool(diagnostic_rows["accepted"].all())
    return {
        "accepted": accepted,
        "decision": "accept" if accepted else "reject",
        "rejection_reasons": reasons,
        "required_benchmarks": sorted(required),
        "present_required_benchmarks": sorted(present_required),
        "candidate_calibration": CANDIDATE_CALIBRATION,
        "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
        "mixed200_diagnostic_accepted": mixed200_accepted,
    }


def _require_min(
    reasons: List[str],
    row: Dict[str, object],
    key: str,
    minimum: float,
) -> None:
    value = _as_float(row.get(key))
    if value is None or value < minimum:
        reasons.append("{0} is below {1}".format(key, minimum))


def _require_max(
    reasons: List[str],
    row: Dict[str, object],
    key: str,
    maximum: float,
    strict: bool = False,
) -> None:
    value = _as_float(row.get(key))
    if value is None or (value >= maximum if strict else value > maximum):
        comparator = "below" if strict else "at most"
        reasons.append("{0} must be {1} {2}".format(key, comparator, maximum))


def _require_between(
    reasons: List[str],
    row: Dict[str, object],
    key: str,
    minimum: float,
    maximum: float,
) -> None:
    value = _as_float(row.get(key))
    if value is None or value < minimum or value > maximum:
        reasons.append("{0} is outside [{1}, {2}]".format(key, minimum, maximum))


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0 or not np.isfinite(denominator):
        return None
    return float(numerator / denominator)


def _series_mean(series: pd.Series) -> Optional[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _positive_rate(series: pd.Series) -> Optional[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float((clean > 0).mean())


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _series_get(row: pd.Series, key: str) -> object:
    if key not in row.index:
        return None
    value = row[key]
    if pd.isna(value):
        return None
    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Kronos base-model validation reports."
    )
    parser.add_argument(
        "--benchmark-dir",
        action="append",
        required=True,
        help="Benchmark output directory containing benchmark_summary.csv.",
    )
    parser.add_argument(
        "--local-data-dir",
        required=True,
        help="Directory of per-ticker local Kronos CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where validation report artifacts should be written.",
    )
    parser.add_argument(
        "--candidate-calibration",
        default=CANDIDATE_CALIBRATION,
        help="Calibration mode to validate as the candidate base model.",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=5,
        help="Forecast horizon used for scaled naive baselines.",
    )
    parser.add_argument(
        "--min-assets-per-date",
        type=int,
        default=25,
        help="Minimum assets per date for daily cross-sectional metrics.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    paths = run_validation_report(
        benchmark_dirs=args.benchmark_dir,
        local_data_dir=args.local_data_dir,
        output_dir=args.output_dir,
        candidate_calibration=args.candidate_calibration,
        forecast_horizon=args.forecast_horizon,
        min_assets_per_date=args.min_assets_per_date,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
