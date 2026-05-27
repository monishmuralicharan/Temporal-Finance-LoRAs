"""Ordered Kronos experiment definitions and selection helpers."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from temporal_finance.kronos_config import KronosCheckpoint4Config

PRED_ABS_RETURN_MAX_MULTIPLE = 2.0
POSITIVE_RATE_MAX_GAP = 0.25
MAE_MAX_WORSENING = 0.10
MATERIAL_IC_IMPROVEMENT = 0.05
PAPER_REFERENCE_LABEL = "Kronos-small Table 2 return forecasting"
PAPER_REFERENCE_RETURN_IC = 0.0665
PAPER_REFERENCE_RETURN_RANK_IC = 0.0622


@dataclass(frozen=True)
class KronosExperimentDefinition:
    name: str
    description: str
    overrides: Dict[str, object]
    selectable: bool = True
    requires_calibrated_best: bool = False


@dataclass
class KronosExperimentRecord:
    name: str
    description: str
    status: str
    selectable: bool
    passed_guardrails: bool
    selected_best: bool
    rejection_reasons: List[str]
    output_dir: str
    metrics: Dict[str, object]
    config: Dict[str, object]
    error: Optional[str] = None

    def summary_row(self) -> Dict[str, object]:
        row = {
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "selectable": self.selectable,
            "passed_guardrails": self.passed_guardrails,
            "selected_best": self.selected_best,
            "rejection_reasons": "; ".join(self.rejection_reasons),
            "error": self.error,
            "output_dir": self.output_dir,
        }
        for key in (
            "model_id",
            "target_column",
            "use_adjusted_ohlc",
            "temperature",
            "top_k",
            "top_p",
            "kronos_sample_count",
            "sample_count",
            "eval_stride",
        ):
            value = self.metrics.get(key)
            if value is None:
                value = self.config.get(key)
            row[key] = value
        for key in (
            "directional_accuracy",
            "mae",
            "mse",
            "pearson_ic",
            "spearman_rank_ic",
            "pred_abs_return_mean",
            "actual_abs_return_mean",
            "pred_abs_return_multiple",
            "pred_positive_rate",
            "actual_positive_rate",
            "positive_rate_gap",
            "pred_close_mae",
            "pred_close_mape",
            "pred_actual_close_ratio_mean",
            "invalid_pred_ohlc_count",
            "negative_pred_volume_count",
            "daily_window_count",
            "daily_mean_ic",
            "daily_mean_rank_ic",
            "daily_ic_positive_rate",
            "daily_rank_ic_positive_rate",
            "daily_mean_directional_accuracy",
            "output_total_invalid_ohlc_rate",
        ):
            row[key] = self.metrics.get(key)
        return row


def ordered_kronos_experiment_definitions() -> List[KronosExperimentDefinition]:
    """Return the fixed experiment order from the Kronos fix plan."""

    return [
        KronosExperimentDefinition(
            name="baseline_greedy_adjusted",
            description="Kronos-base, adjusted OHLCV, greedy decoding.",
            overrides={
                "model_id": "NeoQuasar/Kronos-base",
                "target_column": "Adj Close",
                "use_adjusted_ohlc": True,
                "top_k": 1,
                "top_p": 1.0,
                "sample_count": 1,
            },
        ),
        KronosExperimentDefinition(
            name="raw_ohlcv_greedy",
            description="Kronos-base, raw OHLCV, greedy decoding; diagnostic-only.",
            overrides={
                "model_id": "NeoQuasar/Kronos-base",
                "target_column": "Close",
                "use_adjusted_ohlc": False,
                "top_k": 1,
                "top_p": 1.0,
                "sample_count": 1,
            },
            selectable=False,
        ),
        KronosExperimentDefinition(
            name="kronos_small_greedy_adjusted",
            description="Kronos-small, adjusted OHLCV, greedy decoding.",
            overrides={
                "model_id": "NeoQuasar/Kronos-small",
                "target_column": "Adj Close",
                "use_adjusted_ohlc": True,
                "top_k": 1,
                "top_p": 1.0,
                "sample_count": 1,
            },
        ),
        KronosExperimentDefinition(
            name="sampling_ensemble_adjusted",
            description="Kronos-base, adjusted OHLCV, 5-sample decoding.",
            overrides={
                "model_id": "NeoQuasar/Kronos-base",
                "target_column": "Adj Close",
                "use_adjusted_ohlc": True,
                "top_k": 5,
                "top_p": 0.9,
                "sample_count": 5,
            },
            requires_calibrated_best=True,
        ),
    ]


def paper_proximal_kronos_experiment_definitions(
    include_sample20: bool = False,
) -> List[KronosExperimentDefinition]:
    """Return public-model experiments closer to the Kronos paper setup."""

    definitions = [
        KronosExperimentDefinition(
            name="base_sample10_adjusted",
            description=(
                "Kronos-base, adjusted OHLCV, paper-proximal 10-sample "
                "nucleus decoding."
            ),
            overrides={
                "model_id": "NeoQuasar/Kronos-base",
                "target_column": "Adj Close",
                "use_adjusted_ohlc": True,
                "temperature": 0.6,
                "top_k": 0,
                "top_p": 0.9,
                "sample_count": 10,
            },
        ),
        KronosExperimentDefinition(
            name="small_sample10_adjusted",
            description=(
                "Kronos-small, adjusted OHLCV, paper-proximal 10-sample "
                "nucleus decoding."
            ),
            overrides={
                "model_id": "NeoQuasar/Kronos-small",
                "target_column": "Adj Close",
                "use_adjusted_ohlc": True,
                "temperature": 0.6,
                "top_k": 0,
                "top_p": 0.9,
                "sample_count": 10,
            },
        ),
    ]
    if include_sample20:
        definitions.append(
            KronosExperimentDefinition(
                name="base_sample20_adjusted",
                description=(
                    "Kronos-base, adjusted OHLCV, higher-cost 20-sample "
                    "nucleus decoding."
                ),
                overrides={
                    "model_id": "NeoQuasar/Kronos-base",
                    "target_column": "Adj Close",
                    "use_adjusted_ohlc": True,
                    "temperature": 0.6,
                    "top_k": 0,
                    "top_p": 0.9,
                    "sample_count": 20,
                },
            )
        )
    return definitions


def kronos_experiment_definitions_for_suite(
    suite: str,
    include_sample20: bool = False,
) -> List[KronosExperimentDefinition]:
    if suite == "fix":
        return ordered_kronos_experiment_definitions()
    if suite == "paper-proximal":
        return paper_proximal_kronos_experiment_definitions(
            include_sample20=include_sample20
        )
    raise ValueError("Unknown Kronos experiment suite: {0}".format(suite))


def build_experiment_config(
    base_config: KronosCheckpoint4Config,
    definition: KronosExperimentDefinition,
    output_root: Path,
) -> KronosCheckpoint4Config:
    payload = base_config.to_dict()
    payload.update(definition.overrides)
    payload["output_dir"] = str(output_root / definition.name)
    return KronosCheckpoint4Config.from_dict(payload)


def compute_kronos_prediction_diagnostics(
    predictions: pd.DataFrame,
) -> Dict[str, Optional[float]]:
    pred_return = predictions["pred_5d_return"].to_numpy(dtype=float)
    actual_return = predictions["actual_5d_return"].to_numpy(dtype=float)
    pred_close = predictions["pred_close_t5"].to_numpy(dtype=float)
    actual_close = predictions["actual_close_t5"].to_numpy(dtype=float)

    actual_abs_return_mean = float(np.mean(np.abs(actual_return)))
    pred_abs_return_mean = float(np.mean(np.abs(pred_return)))
    close_abs_error = np.abs(pred_close - actual_close)
    close_ratio = _safe_divide_array(pred_close, actual_close)
    close_mape = _safe_divide_array(close_abs_error, np.abs(actual_close))

    pred_positive_rate = float(np.mean(pred_return > 0))
    actual_positive_rate = float(np.mean(actual_return > 0))

    diagnostics = {
        "pred_abs_return_mean": pred_abs_return_mean,
        "actual_abs_return_mean": actual_abs_return_mean,
        "pred_abs_return_multiple": _safe_divide_scalar(
            pred_abs_return_mean, actual_abs_return_mean
        ),
        "pred_positive_rate": pred_positive_rate,
        "actual_positive_rate": actual_positive_rate,
        "positive_rate_gap": float(abs(pred_positive_rate - actual_positive_rate)),
        "pred_close_mae": float(np.mean(close_abs_error)),
        "pred_close_mape": _safe_nanmean(close_mape),
        "pred_actual_close_ratio_mean": _safe_nanmean(close_ratio),
        "pred_actual_close_ratio_std": _safe_nanstd(close_ratio),
        "invalid_pred_ohlc_count": _count_invalid_ohlc(predictions),
        "negative_pred_volume_count": _count_negative_volume(predictions),
    }
    return diagnostics


def build_experiment_record(
    definition: KronosExperimentDefinition,
    config: KronosCheckpoint4Config,
    metrics: Optional[Dict[str, object]] = None,
    status: str = "completed",
    error: Optional[str] = None,
    reference_metrics: Optional[Dict[str, object]] = None,
) -> KronosExperimentRecord:
    metrics = metrics or {}
    passed, reasons = evaluate_kronos_guardrails(metrics, reference_metrics)
    if not definition.selectable:
        reasons = reasons + ["diagnostic-only experiment"]
    if status != "completed":
        passed = False
        if error:
            reasons = reasons + [error]
    return KronosExperimentRecord(
        name=definition.name,
        description=definition.description,
        status=status,
        selectable=definition.selectable,
        passed_guardrails=passed,
        selected_best=False,
        rejection_reasons=reasons,
        output_dir=config.output_dir,
        metrics=metrics,
        config=config.to_dict(),
        error=error,
    )


def should_skip_kronos_experiment(
    definition: KronosExperimentDefinition,
    completed_records: Iterable[KronosExperimentRecord],
) -> Optional[str]:
    if (
        definition.requires_calibrated_best
        and select_best_kronos_experiment(completed_records) is None
    ):
        return "requires at least one calibrated selectable winner"
    return None


def evaluate_kronos_guardrails(
    metrics: Dict[str, object],
    reference_metrics: Optional[Dict[str, object]] = None,
) -> Tuple[bool, List[str]]:
    reasons = []
    pred_abs_multiple = _as_float(metrics.get("pred_abs_return_multiple"))
    positive_rate_gap = _as_float(metrics.get("positive_rate_gap"))
    mae = _as_float(metrics.get("mae"))

    if pred_abs_multiple is None:
        reasons.append("missing predicted/actual abs-return multiple")
    elif pred_abs_multiple > PRED_ABS_RETURN_MAX_MULTIPLE:
        reasons.append(
            "predicted abs return is {0:.2f}x actual".format(pred_abs_multiple)
        )

    if positive_rate_gap is None:
        reasons.append("missing positive-rate gap")
    elif positive_rate_gap > POSITIVE_RATE_MAX_GAP:
        reasons.append(
            "positive-rate gap is {0:.3f}".format(positive_rate_gap)
        )

    if reference_metrics is not None and mae is not None:
        reference_mae = _as_float(reference_metrics.get("mae"))
        ic = _as_float(metrics.get("pearson_ic")) or 0.0
        reference_ic = _as_float(reference_metrics.get("pearson_ic")) or 0.0
        worsened_mae = (
            reference_mae is not None
            and mae > reference_mae * (1.0 + MAE_MAX_WORSENING)
        )
        material_ic_gain = ic >= reference_ic + MATERIAL_IC_IMPROVEMENT
        if worsened_mae and not material_ic_gain:
            reasons.append("MAE worsened by more than 10% without material IC gain")

    return len(reasons) == 0, reasons


def select_best_kronos_experiment(
    records: Iterable[KronosExperimentRecord],
) -> Optional[KronosExperimentRecord]:
    candidates = [
        record
        for record in records
        if record.status == "completed"
        and record.selectable
        and record.passed_guardrails
    ]
    if not candidates:
        return None
    return sorted(candidates, key=_selection_key, reverse=True)[0]


def write_kronos_experiment_outputs(
    records: List[KronosExperimentRecord],
    output_root: Path,
) -> Dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    best = select_best_kronos_experiment(records)
    for record in records:
        record.selected_best = bool(best is not None and record.name == best.name)

    summary = pd.DataFrame([record.summary_row() for record in records])
    benchmark_summary = build_kronos_benchmark_summary(records)
    summary_path = output_root / "summary.csv"
    json_path = output_root / "summary.json"
    benchmark_summary_path = output_root / "benchmark_summary.csv"
    benchmark_json_path = output_root / "benchmark_summary.json"
    best_config_path = output_root / "best_config.yaml"

    summary.to_csv(summary_path, index=False)
    benchmark_summary.to_csv(benchmark_summary_path, index=False)
    payload = {
        "best_experiment": best.name if best is not None else None,
        "records": [_record_to_json(record) for record in records],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    benchmark_payload = {
        "paper_reference": {
            "label": PAPER_REFERENCE_LABEL,
            "return_ic": PAPER_REFERENCE_RETURN_IC,
            "return_rank_ic": PAPER_REFERENCE_RETURN_RANK_IC,
        },
        "records": benchmark_summary.to_dict(orient="records"),
    }
    benchmark_json_path.write_text(
        json.dumps(benchmark_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths = {
        "summary_csv": summary_path,
        "summary_json": json_path,
        "benchmark_summary_csv": benchmark_summary_path,
        "benchmark_summary_json": benchmark_json_path,
    }
    if best is not None:
        best_config_path.write_text(
            yaml.safe_dump(best.config, sort_keys=False),
            encoding="utf-8",
        )
        paths["best_config"] = best_config_path
    return paths


def build_kronos_benchmark_summary(
    records: List[KronosExperimentRecord],
) -> pd.DataFrame:
    rows = []
    for record in records:
        metrics = record.metrics
        return_ic = _as_float(metrics.get("pearson_ic"))
        return_rank_ic = _as_float(metrics.get("spearman_rank_ic"))
        rows.append(
            {
                "name": record.name,
                "status": record.status,
                "model_id": metrics.get("model_id") or record.config.get("model_id"),
                "eval_stride": metrics.get("eval_stride")
                or record.config.get("eval_stride"),
                "temperature": metrics.get("temperature")
                or record.config.get("temperature"),
                "top_k": metrics.get("top_k") or record.config.get("top_k"),
                "top_p": metrics.get("top_p") or record.config.get("top_p"),
                "sample_count": metrics.get("kronos_sample_count")
                or metrics.get("sample_count")
                or record.config.get("sample_count"),
                "return_ic": return_ic,
                "paper_reference_return_ic": PAPER_REFERENCE_RETURN_IC,
                "return_ic_gap_to_paper": _safe_difference(
                    return_ic,
                    PAPER_REFERENCE_RETURN_IC,
                ),
                "return_rank_ic": return_rank_ic,
                "paper_reference_return_rank_ic": PAPER_REFERENCE_RETURN_RANK_IC,
                "return_rank_ic_gap_to_paper": _safe_difference(
                    return_rank_ic,
                    PAPER_REFERENCE_RETURN_RANK_IC,
                ),
                "daily_window_count": metrics.get("daily_window_count"),
                "daily_mean_ic": metrics.get("daily_mean_ic"),
                "daily_mean_rank_ic": metrics.get("daily_mean_rank_ic"),
                "daily_ic_positive_rate": metrics.get(
                    "daily_ic_positive_rate"
                ),
                "daily_rank_ic_positive_rate": metrics.get(
                    "daily_rank_ic_positive_rate"
                ),
                "directional_accuracy": metrics.get("directional_accuracy"),
                "mae": metrics.get("mae"),
                "pred_abs_return_multiple": metrics.get(
                    "pred_abs_return_multiple"
                ),
                "positive_rate_gap": metrics.get("positive_rate_gap"),
                "invalid_pred_ohlc_count": metrics.get(
                    "invalid_pred_ohlc_count"
                ),
                "output_total_invalid_ohlc_rate": metrics.get(
                    "output_total_invalid_ohlc_rate"
                ),
                "passed_guardrails": record.passed_guardrails,
                "selected_best": record.selected_best,
                "output_dir": record.output_dir,
                "paper_reference_label": PAPER_REFERENCE_LABEL,
            }
        )
    return pd.DataFrame(rows)


def flatten_kronos_output_evaluation_metrics(
    metrics: Dict[str, object],
) -> Dict[str, object]:
    daily = _as_dict(metrics.get("daily_cross_section"))
    return_metrics = _as_dict(metrics.get("return_metrics"))
    ohlc_validity = _as_dict(metrics.get("ohlc_validity"))
    return {
        "output_row_count": metrics.get("row_count"),
        "output_ticker_count": metrics.get("ticker_count"),
        "daily_window_count": daily.get("window_count"),
        "daily_mean_ic": daily.get("mean_ic"),
        "daily_mean_rank_ic": daily.get("mean_rank_ic"),
        "daily_ic_positive_rate": daily.get("ic_positive_rate"),
        "daily_rank_ic_positive_rate": daily.get("rank_ic_positive_rate"),
        "daily_mean_directional_accuracy": daily.get(
            "mean_directional_accuracy"
        ),
        "output_return_mae": return_metrics.get("mae"),
        "output_return_ic": return_metrics.get("pearson_ic"),
        "output_return_rank_ic": return_metrics.get("spearman_rank_ic"),
        "output_total_invalid_ohlc_rate": ohlc_validity.get(
            "total_invalid_rate"
        ),
    }


def _selection_key(record: KronosExperimentRecord):
    metrics = record.metrics
    return (
        _as_float(metrics.get("directional_accuracy")) or float("-inf"),
        _as_float(metrics.get("pearson_ic")) or float("-inf"),
        -(_as_float(metrics.get("mae")) or float("inf")),
    )


def _record_to_json(record: KronosExperimentRecord) -> Dict[str, object]:
    payload = asdict(record)
    return payload


def _count_invalid_ohlc(predictions: pd.DataFrame) -> int:
    required = {"pred_open_t5", "pred_high_t5", "pred_low_t5", "pred_close_t5"}
    if not required.issubset(predictions.columns):
        return 0
    high = predictions["pred_high_t5"].to_numpy(dtype=float)
    low = predictions["pred_low_t5"].to_numpy(dtype=float)
    open_ = predictions["pred_open_t5"].to_numpy(dtype=float)
    close = predictions["pred_close_t5"].to_numpy(dtype=float)
    invalid = (
        (high < np.maximum(open_, close))
        | (low > np.minimum(open_, close))
        | (low > high)
    )
    return int(np.sum(invalid))


def _count_negative_volume(predictions: pd.DataFrame) -> int:
    if "pred_volume_t5" not in predictions.columns:
        return 0
    volume = predictions["pred_volume_t5"].to_numpy(dtype=float)
    return int(np.sum(volume < 0))


def _safe_divide_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    result[~np.isfinite(result)] = np.nan
    return result


def _safe_divide_scalar(
    numerator: float, denominator: float
) -> Optional[float]:
    if denominator == 0 or not np.isfinite(denominator):
        return None
    value = numerator / denominator
    if not np.isfinite(value):
        return None
    return float(value)


def _safe_difference(
    value: Optional[float],
    reference: float,
) -> Optional[float]:
    if value is None:
        return None
    return float(value - reference)


def _safe_nanmean(values: np.ndarray) -> Optional[float]:
    if np.all(np.isnan(values)):
        return None
    return float(np.nanmean(values))


def _safe_nanstd(values: np.ndarray) -> Optional[float]:
    if np.all(np.isnan(values)):
        return None
    return float(np.nanstd(values))


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _as_dict(value: object) -> Dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}
