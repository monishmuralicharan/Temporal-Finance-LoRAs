"""Manifest-driven Kronos base model benchmark utilities."""

import csv
import json
import time
from io import StringIO
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from temporal_finance.evaluation import compute_metrics
from temporal_finance.kronos_config import (
    KronosCheckpoint4Config,
    KronosConfigurationError,
    load_kronos_checkpoint4_config,
)
from temporal_finance.kronos_experiments import (
    PAPER_REFERENCE_LABEL,
    PAPER_REFERENCE_RETURN_IC,
    PAPER_REFERENCE_RETURN_RANK_IC,
    compute_kronos_prediction_diagnostics,
    flatten_kronos_output_evaluation_metrics,
)
from temporal_finance.kronos_output_evaluation import run_kronos_output_evaluation
from temporal_finance.kronos_pipeline import run_kronos_checkpoint4

CALIBRATION_NONE = "none"
CALIBRATION_VOL_RESCALE = "vol_rescale"
CALIBRATION_MEAN_CENTER = "mean_center"
CALIBRATION_VOL_RESCALE_MEAN_CENTER = "vol_rescale_mean_center"
CALIBRATION_MODES = {
    CALIBRATION_NONE,
    CALIBRATION_VOL_RESCALE,
    CALIBRATION_MEAN_CENTER,
    CALIBRATION_VOL_RESCALE_MEAN_CENTER,
}

DEFAULT_PROMOTION_THRESHOLDS = {
    "min_return_ic": 0.03,
    "min_return_rank_ic": 0.03,
    "min_daily_positive_rate": 0.52,
    "min_directional_accuracy": 0.525,
    "min_pred_abs_return_multiple": 0.75,
    "max_pred_abs_return_multiple": 1.50,
    "max_positive_rate_gap": 0.15,
    "max_invalid_ohlc_rate": 0.02,
    "ridge_rank_ic_tolerance": 0.03,
}


@dataclass(frozen=True)
class KronosBenchmarkExperiment:
    name: str
    description: str
    overrides: Dict[str, object]
    calibration_modes: List[str]
    selectable: bool = True


@dataclass(frozen=True)
class KronosBenchmarkManifest:
    name: str
    stage: str
    base_config_path: str
    output_dir: str
    experiments: List[KronosBenchmarkExperiment]
    min_assets_per_date: int = 2
    calibration_lookback: int = 20
    references: Optional[Dict[str, str]] = None
    promotion_thresholds: Optional[Dict[str, float]] = None
    enforce_ridge_gate: bool = False


@dataclass(frozen=True)
class PromotionGateResult:
    passed: bool
    reasons: List[str]


@dataclass
class KronosBenchmarkRunRecord:
    name: str
    base_experiment: str
    description: str
    stage: str
    status: str
    selectable: bool
    calibration_mode: str
    passed_gate: bool
    rejection_reasons: List[str]
    output_dir: str
    metrics: Dict[str, object]
    config: Dict[str, object]
    runtime_seconds: Optional[float] = None
    modal_gpu: Optional[str] = None
    error: Optional[str] = None

    def summary_row(self) -> Dict[str, object]:
        row = {
            "name": self.name,
            "base_experiment": self.base_experiment,
            "description": self.description,
            "stage": self.stage,
            "status": self.status,
            "selectable": self.selectable,
            "calibration_mode": self.calibration_mode,
            "passed_gate": self.passed_gate,
            "rejection_reasons": "; ".join(self.rejection_reasons),
            "error": self.error,
            "output_dir": self.output_dir,
            "runtime_seconds": self.runtime_seconds,
            "modal_gpu": self.modal_gpu,
            "paper_reference_label": PAPER_REFERENCE_LABEL,
        }
        for key in (
            "model_id",
            "target_column",
            "use_adjusted_ohlc",
            "context_length",
            "forecast_horizon",
            "eval_stride",
            "temperature",
            "top_k",
            "top_p",
            "kronos_sample_count",
            "sample_count",
            "batch_size",
        ):
            row[key] = self.metrics.get(key, self.config.get(key))
        for key in (
            "sample_count",
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
            "current_kronos_rank_ic",
            "ridge_rank_ic",
            "rank_ic_gap_to_current_kronos",
            "rank_ic_gap_to_ridge",
        ):
            row[key] = self.metrics.get(key)
        row["paper_reference_return_ic"] = PAPER_REFERENCE_RETURN_IC
        row["paper_reference_return_rank_ic"] = PAPER_REFERENCE_RETURN_RANK_IC
        row["return_ic_gap_to_paper"] = _safe_difference(
            _as_float(self.metrics.get("pearson_ic")),
            PAPER_REFERENCE_RETURN_IC,
        )
        row["return_rank_ic_gap_to_paper"] = _safe_difference(
            _as_float(self.metrics.get("spearman_rank_ic")),
            PAPER_REFERENCE_RETURN_RANK_IC,
        )
        return row


def load_kronos_benchmark_manifest(path: str) -> KronosBenchmarkManifest:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise KronosConfigurationError(
            "Benchmark manifest does not exist: {0}".format(manifest_path)
        )
    if manifest_path.suffix.lower() == ".csv":
        return _load_csv_manifest(manifest_path)
    return _load_yaml_manifest(manifest_path)


def _load_yaml_manifest(path: Path) -> KronosBenchmarkManifest:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise KronosConfigurationError("Benchmark manifest must be a YAML mapping.")
    experiments = [_parse_experiment(item) for item in raw.get("experiments", [])]
    if not experiments:
        raise KronosConfigurationError("Benchmark manifest must define experiments.")
    thresholds = raw.get("promotion_thresholds") or {}
    if thresholds and not isinstance(thresholds, dict):
        raise KronosConfigurationError("promotion_thresholds must be a mapping.")
    references = raw.get("references") or {}
    if references and not isinstance(references, dict):
        raise KronosConfigurationError("references must be a mapping.")
    return KronosBenchmarkManifest(
        name=str(raw.get("name") or path.stem),
        stage=str(raw.get("stage") or "smoke"),
        base_config_path=str(raw["base_config"]),
        output_dir=str(raw["output_dir"]),
        experiments=experiments,
        min_assets_per_date=int(raw.get("min_assets_per_date", 2)),
        calibration_lookback=int(raw.get("calibration_lookback", 20)),
        references=references or None,
        promotion_thresholds={str(k): float(v) for k, v in thresholds.items()}
        or None,
        enforce_ridge_gate=bool(raw.get("enforce_ridge_gate", False)),
    )


def _load_csv_manifest(path: Path) -> KronosBenchmarkManifest:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise KronosConfigurationError("CSV benchmark manifest has no rows.")
    first = rows[0]
    experiments = []
    for row in rows:
        overrides = {}
        for key in (
            "model_id",
            "target_column",
            "use_adjusted_ohlc",
            "temperature",
            "top_k",
            "top_p",
            "sample_count",
            "batch_size",
            "eval_stride",
        ):
            value = row.get(key)
            if value not in (None, ""):
                overrides[key] = _parse_csv_value(value)
        modes = _split_modes(row.get("calibration_modes") or CALIBRATION_NONE)
        experiments.append(
            KronosBenchmarkExperiment(
                name=str(row["name"]),
                description=str(row.get("description") or row["name"]),
                overrides=overrides,
                calibration_modes=modes,
                selectable=_parse_bool(row.get("selectable", "true")),
            )
        )
    return KronosBenchmarkManifest(
        name=str(first.get("manifest_name") or path.stem),
        stage=str(first.get("stage") or "smoke"),
        base_config_path=str(first["base_config"]),
        output_dir=str(first["output_dir"]),
        experiments=experiments,
        min_assets_per_date=int(first.get("min_assets_per_date") or 2),
        calibration_lookback=int(first.get("calibration_lookback") or 20),
        references=None,
        promotion_thresholds=None,
        enforce_ridge_gate=_parse_bool(first.get("enforce_ridge_gate", "false")),
    )


def _parse_experiment(raw: object) -> KronosBenchmarkExperiment:
    if not isinstance(raw, dict):
        raise KronosConfigurationError("Each benchmark experiment must be a mapping.")
    modes = _split_modes(raw.get("calibration_modes") or [CALIBRATION_NONE])
    unknown_modes = sorted(set(modes) - CALIBRATION_MODES)
    if unknown_modes:
        raise KronosConfigurationError(
            "Unknown calibration modes: {0}".format(", ".join(unknown_modes))
        )
    overrides = raw.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise KronosConfigurationError("Experiment overrides must be a mapping.")
    return KronosBenchmarkExperiment(
        name=str(raw["name"]),
        description=str(raw.get("description") or raw["name"]),
        overrides=dict(overrides),
        calibration_modes=modes,
        selectable=bool(raw.get("selectable", True)),
    )


def run_benchmark_manifest_local(
    manifest_path: str,
    forecaster: Optional[object] = None,
    histories: Optional[Dict[str, pd.DataFrame]] = None,
    only_run_names: Optional[Sequence[str]] = None,
) -> List[KronosBenchmarkRunRecord]:
    manifest = load_kronos_benchmark_manifest(manifest_path)
    base_config = load_kronos_checkpoint4_config(manifest.base_config_path)
    output_root = Path(manifest.output_dir).resolve()
    references = load_benchmark_references(manifest.references)
    records = []
    allowed = set(only_run_names or [])

    for experiment in manifest.experiments:
        modes = _select_calibration_modes(experiment, allowed)
        if not modes:
            continue
        config = build_benchmark_config(base_config, experiment, output_root)
        raw_output_dir = output_root / "_raw" / experiment.name
        raw_config = KronosCheckpoint4Config.from_dict(
            {**config.to_dict(), "output_dir": str(raw_output_dir)}
        )
        start = time.monotonic()
        try:
            result = run_kronos_checkpoint4(
                config=raw_config,
                forecaster=forecaster,
                histories=histories,
            )
            runtime_seconds = time.monotonic() - start
            for mode in modes:
                records.append(
                    _materialize_benchmark_run(
                        manifest=manifest,
                        experiment=experiment,
                        config=config,
                        raw_predictions=result.predictions,
                        raw_metrics=result.metrics,
                        calibration_mode=mode,
                        output_root=output_root,
                        references=references,
                        runtime_seconds=runtime_seconds,
                    )
                )
        except Exception as exc:
            records.append(
                _failed_record(
                    manifest=manifest,
                    experiment=experiment,
                    config=config,
                    output_root=output_root,
                    error=str(exc),
                )
            )
    write_benchmark_outputs(records, output_root)
    return records


def build_benchmark_config(
    base_config: KronosCheckpoint4Config,
    experiment: KronosBenchmarkExperiment,
    output_root: Path,
) -> KronosCheckpoint4Config:
    payload = base_config.to_dict()
    payload.update(experiment.overrides)
    payload["output_dir"] = str(output_root / experiment.name)
    return KronosCheckpoint4Config.from_dict(payload)


def selected_calibration_modes(
    experiment: KronosBenchmarkExperiment,
    only_run_names: Optional[Sequence[str]] = None,
) -> List[str]:
    return _select_calibration_modes(experiment, set(only_run_names or []))


def build_failed_benchmark_record(
    manifest: KronosBenchmarkManifest,
    experiment: KronosBenchmarkExperiment,
    config: KronosCheckpoint4Config,
    output_root: Path,
    error: str,
) -> KronosBenchmarkRunRecord:
    return _failed_record(
        manifest=manifest,
        experiment=experiment,
        config=config,
        output_root=output_root,
        error=error,
    )


def materialize_remote_benchmark_artifacts(
    manifest: KronosBenchmarkManifest,
    experiment: KronosBenchmarkExperiment,
    config: KronosCheckpoint4Config,
    predictions_csv: str,
    metrics_json: str,
    output_root: Path,
    references: Dict[str, Optional[float]],
    runtime_seconds: Optional[float] = None,
    modal_gpu: Optional[str] = None,
    only_calibration_modes: Optional[Sequence[str]] = None,
) -> List[KronosBenchmarkRunRecord]:
    raw_predictions = pd.read_csv(StringIO(predictions_csv))
    raw_metrics = json.loads(metrics_json)
    modes = list(only_calibration_modes or experiment.calibration_modes)
    return [
        _materialize_benchmark_run(
            manifest=manifest,
            experiment=experiment,
            config=config,
            raw_predictions=raw_predictions,
            raw_metrics=raw_metrics,
            calibration_mode=mode,
            output_root=output_root,
            references=references,
            runtime_seconds=runtime_seconds,
            modal_gpu=modal_gpu,
        )
        for mode in modes
    ]


def calibrate_kronos_predictions(
    predictions: pd.DataFrame,
    mode: str,
    lookback: int = 20,
) -> pd.DataFrame:
    if mode not in CALIBRATION_MODES:
        raise ValueError("Unknown calibration mode: {0}".format(mode))
    frame = predictions.copy()
    if mode == CALIBRATION_NONE:
        frame["calibration_mode"] = mode
        return frame
    pred = pd.to_numeric(frame["pred_5d_return"], errors="coerce")
    actual = pd.to_numeric(frame["actual_5d_return"], errors="coerce")
    calibrated = pred.copy()
    if mode in (CALIBRATION_MEAN_CENTER, CALIBRATION_VOL_RESCALE_MEAN_CENTER):
        bias = _rolling_prior_mean(pred - actual, frame, lookback).fillna(0.0)
        calibrated = calibrated - bias
    if mode in (CALIBRATION_VOL_RESCALE, CALIBRATION_VOL_RESCALE_MEAN_CENTER):
        actual_scale = _rolling_prior_mean(actual.abs(), frame, lookback)
        pred_scale = _rolling_prior_mean(calibrated.abs(), frame, lookback)
        factor = _safe_series_divide(actual_scale, pred_scale).clip(lower=0.25, upper=4.0)
        factor = factor.fillna(1.0)
        calibrated = calibrated * factor
    frame["pred_5d_return"] = calibrated
    frame["pred_close_t5"] = frame["last_context_close"] * (1.0 + calibrated)
    _rescale_predicted_ohlc(frame, predictions["pred_close_t5"], frame["pred_close_t5"])
    frame["calibration_mode"] = mode
    return frame


def score_and_write_benchmark_predictions(
    predictions: pd.DataFrame,
    output_dir: Path,
    config: KronosCheckpoint4Config,
    metadata: Dict[str, object],
    min_assets_per_date: int = 2,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    per_ticker_path = output_dir / "per_ticker_metrics.csv"
    predictions.to_csv(predictions_path, index=False)
    per_ticker = compute_per_ticker_metrics(predictions)
    per_ticker.to_csv(per_ticker_path, index=False)

    metrics = compute_metrics(predictions)
    metrics.update(compute_kronos_prediction_diagnostics(predictions))
    metrics.update(_config_metric_payload(config))
    metrics.update(metadata)
    evaluation = run_kronos_output_evaluation(
        predictions_path=str(predictions_path),
        output_dir=str(output_dir / "evaluation"),
        min_assets_per_date=min_assets_per_date,
    )
    metrics.update(flatten_kronos_output_evaluation_metrics(evaluation.metrics))
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def compute_per_ticker_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, group in predictions.groupby("ticker", sort=False):
        metrics = compute_metrics(group)
        metrics.update(compute_kronos_prediction_diagnostics(group))
        metrics["ticker"] = ticker
        rows.append(metrics)
    columns = [
        "ticker",
        "sample_count",
        "mse",
        "mae",
        "directional_accuracy",
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
        "pred_actual_close_ratio_std",
        "invalid_pred_ohlc_count",
        "negative_pred_volume_count",
    ]
    return pd.DataFrame(rows, columns=columns)


def evaluate_promotion_gate(
    metrics: Dict[str, object],
    selectable: bool = True,
    stage: str = "smoke",
    thresholds: Optional[Dict[str, float]] = None,
    references: Optional[Dict[str, Optional[float]]] = None,
    enforce_ridge_gate: bool = False,
) -> PromotionGateResult:
    thresholds = {**DEFAULT_PROMOTION_THRESHOLDS, **(thresholds or {})}
    references = references or {}
    reasons = []
    if not selectable:
        reasons.append("diagnostic-only experiment")
    _require_min(reasons, metrics, "pearson_ic", thresholds["min_return_ic"], "return IC")
    _require_min(
        reasons,
        metrics,
        "spearman_rank_ic",
        thresholds["min_return_rank_ic"],
        "return RankIC",
    )
    _require_min(
        reasons,
        metrics,
        "daily_ic_positive_rate",
        thresholds["min_daily_positive_rate"],
        "daily IC positive rate",
    )
    _require_min(
        reasons,
        metrics,
        "daily_rank_ic_positive_rate",
        thresholds["min_daily_positive_rate"],
        "daily RankIC positive rate",
    )
    _require_min(
        reasons,
        metrics,
        "directional_accuracy",
        thresholds["min_directional_accuracy"],
        "directional accuracy",
    )
    multiple = _as_float(metrics.get("pred_abs_return_multiple"))
    if multiple is None:
        reasons.append("missing predicted/actual abs-return multiple")
    elif not (
        thresholds["min_pred_abs_return_multiple"]
        <= multiple
        <= thresholds["max_pred_abs_return_multiple"]
    ):
        reasons.append("predicted/actual abs-return multiple is {0:.2f}".format(multiple))
    _require_max(
        reasons,
        metrics,
        "positive_rate_gap",
        thresholds["max_positive_rate_gap"],
        "positive-rate gap",
    )
    invalid_rate = metrics.get("output_total_invalid_ohlc_rate")
    if invalid_rate is None:
        invalid_count = _as_float(metrics.get("invalid_pred_ohlc_count"))
        sample_count = _as_float(metrics.get("sample_count"))
        invalid_rate = (
            invalid_count / sample_count
            if invalid_count is not None and sample_count
            else None
        )
    if invalid_rate is None:
        reasons.append("missing invalid OHLC rate")
    elif float(invalid_rate) > thresholds["max_invalid_ohlc_rate"]:
        reasons.append("invalid OHLC rate is {0:.3f}".format(float(invalid_rate)))

    current_rank_ic = references.get("current_kronos_rank_ic")
    rank_ic = _as_float(metrics.get("spearman_rank_ic"))
    if current_rank_ic is not None and rank_ic is not None:
        if rank_ic <= current_rank_ic:
            reasons.append("RankIC does not beat current Kronos baseline")
    ridge_rank_ic = references.get("ridge_rank_ic")
    if enforce_ridge_gate and ridge_rank_ic is not None and rank_ic is not None:
        min_ridge_competitive = ridge_rank_ic - thresholds["ridge_rank_ic_tolerance"]
        if rank_ic < min_ridge_competitive:
            reasons.append("RankIC is not competitive with Ridge")
    return PromotionGateResult(passed=not reasons, reasons=reasons)


def load_benchmark_references(
    references: Optional[Dict[str, str]],
) -> Dict[str, Optional[float]]:
    payload = {
        "current_kronos_rank_ic": None,
        "ridge_rank_ic": None,
        "ridge_return_ic": None,
    }
    if not references:
        return payload
    current_path = references.get("current_kronos_metrics")
    if current_path and Path(current_path).exists():
        current = json.loads(Path(current_path).read_text(encoding="utf-8"))
        payload["current_kronos_rank_ic"] = _as_float(
            current.get("spearman_rank_ic")
        )
    ridge_path = references.get("ridge_metrics")
    if ridge_path and Path(ridge_path).exists():
        ridge = json.loads(Path(ridge_path).read_text(encoding="utf-8"))
        point_metrics = ridge.get("models", {}).get("ridge", {}).get(
            "point_metrics", {}
        )
        payload["ridge_rank_ic"] = _as_float(point_metrics.get("spearman_rank_ic"))
        payload["ridge_return_ic"] = _as_float(point_metrics.get("pearson_ic"))
    return payload


def write_benchmark_outputs(
    records: List[KronosBenchmarkRunRecord],
    output_root: Path,
) -> Dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([record.summary_row() for record in records])
    summary_path = output_root / "benchmark_summary.csv"
    json_path = output_root / "benchmark_summary.json"
    promoted_path = output_root / "promoted_run_names.txt"
    summary.to_csv(summary_path, index=False)
    payload = {
        "paper_reference": {
            "label": PAPER_REFERENCE_LABEL,
            "return_ic": PAPER_REFERENCE_RETURN_IC,
            "return_rank_ic": PAPER_REFERENCE_RETURN_RANK_IC,
        },
        "records": [record.summary_row() for record in records],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    promoted = [record.name for record in records if record.passed_gate]
    promoted_path.write_text(
        "\n".join(promoted) + ("\n" if promoted else ""),
        encoding="utf-8",
    )
    return {
        "benchmark_summary_csv": summary_path,
        "benchmark_summary_json": json_path,
        "promoted_run_names": promoted_path,
    }


def load_promoted_run_names(path: str) -> List[str]:
    promoted_path = Path(path)
    if promoted_path.suffix.lower() == ".csv":
        frame = pd.read_csv(promoted_path)
        if "name" not in frame.columns or "passed_gate" not in frame.columns:
            raise KronosConfigurationError(
                "Promoted CSV must contain name and passed_gate columns."
            )
        return frame.loc[frame["passed_gate"].astype(bool), "name"].astype(str).tolist()
    return [
        line.strip()
        for line in promoted_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def benchmark_run_name(experiment_name: str, calibration_mode: str) -> str:
    if calibration_mode == CALIBRATION_NONE:
        return experiment_name
    return "{0}__{1}".format(experiment_name, calibration_mode)


def _materialize_benchmark_run(
    manifest: KronosBenchmarkManifest,
    experiment: KronosBenchmarkExperiment,
    config: KronosCheckpoint4Config,
    raw_predictions: pd.DataFrame,
    raw_metrics: Dict[str, object],
    calibration_mode: str,
    output_root: Path,
    references: Dict[str, Optional[float]],
    runtime_seconds: Optional[float] = None,
    modal_gpu: Optional[str] = None,
) -> KronosBenchmarkRunRecord:
    run_name = benchmark_run_name(experiment.name, calibration_mode)
    run_output_dir = output_root / run_name
    predictions = calibrate_kronos_predictions(
        raw_predictions,
        calibration_mode,
        lookback=manifest.calibration_lookback,
    )
    metadata = {
        "run_name": run_name,
        "base_experiment": experiment.name,
        "stage": manifest.stage,
        "calibration_mode": calibration_mode,
        "failed_tickers": raw_metrics.get("failed_tickers", []),
        "failed_ticker_count": raw_metrics.get("failed_ticker_count", 0),
        "runtime_seconds": runtime_seconds,
        "modal_gpu": modal_gpu,
    }
    metrics = score_and_write_benchmark_predictions(
        predictions=predictions,
        output_dir=run_output_dir,
        config=config,
        metadata=metadata,
        min_assets_per_date=manifest.min_assets_per_date,
    )
    _attach_reference_metrics(metrics, references)
    gate = evaluate_promotion_gate(
        metrics,
        selectable=experiment.selectable,
        stage=manifest.stage,
        thresholds=manifest.promotion_thresholds,
        references=references,
        enforce_ridge_gate=manifest.enforce_ridge_gate,
    )
    metrics_path = run_output_dir / "metrics.json"
    metrics.update(
        {
            "passed_gate": gate.passed,
            "rejection_reasons": gate.reasons,
        }
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return KronosBenchmarkRunRecord(
        name=run_name,
        base_experiment=experiment.name,
        description=experiment.description,
        stage=manifest.stage,
        status="completed",
        selectable=experiment.selectable,
        calibration_mode=calibration_mode,
        passed_gate=gate.passed,
        rejection_reasons=gate.reasons,
        output_dir=str(run_output_dir),
        metrics=metrics,
        config=config.to_dict(),
        runtime_seconds=runtime_seconds,
        modal_gpu=modal_gpu,
    )


def _failed_record(
    manifest: KronosBenchmarkManifest,
    experiment: KronosBenchmarkExperiment,
    config: KronosCheckpoint4Config,
    output_root: Path,
    error: str,
) -> KronosBenchmarkRunRecord:
    return KronosBenchmarkRunRecord(
        name=experiment.name,
        base_experiment=experiment.name,
        description=experiment.description,
        stage=manifest.stage,
        status="failed",
        selectable=experiment.selectable,
        calibration_mode=CALIBRATION_NONE,
        passed_gate=False,
        rejection_reasons=[error],
        output_dir=str(output_root / experiment.name),
        metrics={},
        config=config.to_dict(),
        error=error,
    )


def _config_metric_payload(config: KronosCheckpoint4Config) -> Dict[str, object]:
    return {
        "model": "Kronos",
        "model_id": config.model_id,
        "tokenizer_id": config.tokenizer_id,
        "ticker": config.ticker,
        "tickers": config.effective_tickers,
        "ticker_count": len(config.effective_tickers),
        "target_column": config.target_column,
        "context_length": config.context_length,
        "forecast_horizon": config.forecast_horizon,
        "eval_stride": config.eval_stride,
        "eval_start_date": config.eval_start_date,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "kronos_sample_count": config.sample_count,
        "use_adjusted_ohlc": config.use_adjusted_ohlc,
        "batch_size": config.batch_size,
    }


def _attach_reference_metrics(
    metrics: Dict[str, object],
    references: Dict[str, Optional[float]],
) -> None:
    metrics["current_kronos_rank_ic"] = references.get("current_kronos_rank_ic")
    metrics["ridge_rank_ic"] = references.get("ridge_rank_ic")
    metrics["ridge_return_ic"] = references.get("ridge_return_ic")
    rank_ic = _as_float(metrics.get("spearman_rank_ic"))
    metrics["rank_ic_gap_to_current_kronos"] = _safe_difference(
        rank_ic,
        references.get("current_kronos_rank_ic"),
    )
    metrics["rank_ic_gap_to_ridge"] = _safe_difference(
        rank_ic,
        references.get("ridge_rank_ic"),
    )


def _rolling_prior_mean(values: pd.Series, frame: pd.DataFrame, lookback: int) -> pd.Series:
    working = pd.DataFrame(
        {
            "ticker": frame["ticker"].astype(str),
            "date": pd.to_datetime(frame["date"]),
            "value": pd.to_numeric(values, errors="coerce"),
            "original_index": frame.index,
        }
    ).sort_values(["ticker", "date", "original_index"])
    working["prior_mean"] = (
        working.groupby("ticker")["value"]
        .transform(lambda series: series.shift(1).rolling(lookback, min_periods=3).mean())
    )
    return working.set_index("original_index")["prior_mean"].reindex(frame.index)


def _safe_series_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0.0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _rescale_predicted_ohlc(
    frame: pd.DataFrame,
    old_close: pd.Series,
    new_close: pd.Series,
) -> None:
    ratio = _safe_series_divide(new_close, pd.to_numeric(old_close, errors="coerce"))
    ratio = ratio.fillna(1.0)
    for column in ("pred_open_t5", "pred_high_t5", "pred_low_t5"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce") * ratio


def _select_calibration_modes(
    experiment: KronosBenchmarkExperiment,
    allowed_run_names: set,
) -> List[str]:
    if not allowed_run_names:
        return experiment.calibration_modes
    return [
        mode
        for mode in experiment.calibration_modes
        if benchmark_run_name(experiment.name, mode) in allowed_run_names
    ]


def _require_min(
    reasons: List[str],
    metrics: Dict[str, object],
    key: str,
    threshold: float,
    label: str,
) -> None:
    value = _as_float(metrics.get(key))
    if value is None:
        reasons.append("missing {0}".format(label))
    elif value <= threshold:
        reasons.append("{0} is {1:.4f}".format(label, value))


def _require_max(
    reasons: List[str],
    metrics: Dict[str, object],
    key: str,
    threshold: float,
    label: str,
) -> None:
    value = _as_float(metrics.get(key))
    if value is None:
        reasons.append("missing {0}".format(label))
    elif value > threshold:
        reasons.append("{0} is {1:.4f}".format(label, value))


def _safe_difference(
    value: Optional[float],
    reference: Optional[float],
) -> Optional[float]:
    if value is None or reference is None:
        return None
    return float(value - reference)


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


def _split_modes(value: object) -> List[str]:
    if isinstance(value, str):
        modes = [item.strip() for item in value.split(",")]
    else:
        modes = [str(item).strip() for item in value]
    modes = [mode for mode in modes if mode]
    return modes or [CALIBRATION_NONE]


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def _parse_csv_value(value: str) -> object:
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        if "." in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value
