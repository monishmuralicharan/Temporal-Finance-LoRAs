"""Aggressive frozen Kronos base-model search utilities."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import modal
import pandas as pd
import yaml
from modal.exception import TimeoutError as ModalTimeoutError

from temporal_finance.kronos_benchmark import load_kronos_benchmark_manifest
from temporal_finance.kronos_validation_report import (
    build_naive_baseline_predictions,
    _as_float,
    _compute_report_metrics,
)
from temporal_finance.modal_kronos_checkpoint4 import (
    _collect_detached_benchmark_artifacts,
    _run_benchmark_remote,
)

DEFAULT_SUITE_DIR = "outputs/kronos_base_search/aggressive_common100"
COMMON100_BASE_CONFIG = "configs/kronos_dataset_common100_context512_s10_smoke.yaml"
MIXED200_BASE_CONFIG = "configs/kronos_dataset_top200_context512_smoke.yaml"
LOCAL_DATA_DIR = "dataset/data/kronos/combined_by_ticker"
CANDIDATE_MODELS = [
    ("small", "NeoQuasar/Kronos-small"),
    ("base", "NeoQuasar/Kronos-base"),
]
HORIZONS = [1, 3, 5, 10]
DECODING_PRESETS = [
    {
        "name": "greedy",
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0,
        "sample_count": 1,
    },
    {
        "name": "t04_p08_s3",
        "temperature": 0.4,
        "top_k": 0,
        "top_p": 0.8,
        "sample_count": 3,
    },
    {
        "name": "t06_p09_s5",
        "temperature": 0.6,
        "top_k": 0,
        "top_p": 0.9,
        "sample_count": 5,
    },
    {
        "name": "t06_p09_s10",
        "temperature": 0.6,
        "top_k": 0,
        "top_p": 0.9,
        "sample_count": 10,
    },
]
CALIBRATION_MODES = ["none", "vol_rescale", "vol_rescale_mean_center"]
RELAXED_STAGE1_THRESHOLDS = {
    "min_return_ic": 0.03,
    "min_return_rank_ic": 0.03,
    "min_daily_positive_rate": 0.50,
    "min_directional_accuracy": 0.515,
    "min_pred_abs_return_multiple": 0.5,
    "max_pred_abs_return_multiple": 2.0,
    "max_positive_rate_gap": 0.20,
    "max_invalid_ohlc_rate": 0.03,
    "ridge_rank_ic_tolerance": 0.03,
}
STRICT_THRESHOLDS = {
    "min_return_ic": 0.05,
    "min_return_rank_ic": 0.05,
    "min_daily_positive_rate": 0.52,
    "min_directional_accuracy": 0.525,
    "min_pred_abs_return_multiple": 0.5,
    "max_pred_abs_return_multiple": 2.0,
    "max_positive_rate_gap": 0.10,
    "max_invalid_ohlc_rate": 0.03,
    "ridge_rank_ic_tolerance": 0.03,
}
NAIVE_RANK_IC_MARGIN = 0.02


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    model_label: str
    model_id: str
    forecast_horizon: int
    decoding_name: str
    temperature: float
    top_k: int
    top_p: float
    sample_count: int

    def overrides(self, eval_stride: int, eval_start_date: Optional[str] = None) -> Dict[str, object]:
        payload = {
            "model_id": self.model_id,
            "target_column": "close",
            "use_adjusted_ohlc": False,
            "forecast_horizon": self.forecast_horizon,
            "eval_stride": eval_stride,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "sample_count": self.sample_count,
            "batch_size": 8,
        }
        if eval_start_date:
            payload["eval_start_date"] = eval_start_date
        return payload


def build_candidate_grid() -> List[CandidateSpec]:
    candidates = []
    for model_label, model_id in CANDIDATE_MODELS:
        for horizon in HORIZONS:
            for preset in DECODING_PRESETS:
                name = "{0}_h{1}_{2}".format(model_label, horizon, preset["name"])
                candidates.append(
                    CandidateSpec(
                        name=name,
                        model_label=model_label,
                        model_id=model_id,
                        forecast_horizon=horizon,
                        decoding_name=str(preset["name"]),
                        temperature=float(preset["temperature"]),
                        top_k=int(preset["top_k"]),
                        top_p=float(preset["top_p"]),
                        sample_count=int(preset["sample_count"]),
                    )
                )
    return candidates


def generate_search_suite(output_dir: str = DEFAULT_SUITE_DIR) -> Dict[str, Path]:
    suite_dir = Path(output_dir).resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidate_grid()
    grid_path = suite_dir / "candidate_grid.csv"
    pd.DataFrame([asdict(candidate) for candidate in candidates]).to_csv(
        grid_path, index=False
    )
    state = {
        "version": 1,
        "suite_dir": str(suite_dir),
        "local_data_dir": LOCAL_DATA_DIR,
        "generated_at_unix": time.time(),
        "stages": {"stage1": []},
    }
    for candidate in candidates:
        manifest_path = _write_candidate_manifest(
            suite_dir=suite_dir,
            stage="stage1",
            candidate=candidate,
            base_config=COMMON100_BASE_CONFIG,
            eval_stride=21,
            calibration_modes=CALIBRATION_MODES,
            thresholds=RELAXED_STAGE1_THRESHOLDS,
        )
        state["stages"]["stage1"].append(
            _suite_item(candidate, manifest_path, suite_dir / "stage1" / candidate.name)
        )
    state_path = suite_dir / "suite_state.json"
    _write_json(state_path, state)
    return {"candidate_grid": grid_path, "suite_state": state_path}


def prepare_stage_from_previous(
    suite_state_path: str,
    from_stage: str,
    to_stage: str,
) -> Dict[str, Path]:
    state_path = Path(suite_state_path).resolve()
    state = _load_json(state_path)
    suite_dir = Path(state["suite_dir"])
    rows = _read_stage_summary(state, from_stage)
    if rows.empty:
        raise SystemExit("No completed summary rows found for {0}.".format(from_stage))
    if from_stage == "stage1" and to_stage == "stage2":
        promoted = _filter_rows(rows, RELAXED_STAGE1_THRESHOLDS)
        eval_stride = 5
        eval_start_date = None
        base_config = COMMON100_BASE_CONFIG
        thresholds = STRICT_THRESHOLDS
    elif from_stage == "stage2" and to_stage == "recent_daily":
        promoted = _filter_rows(rows, STRICT_THRESHOLDS, require_naive_gap=True)
        eval_stride = 1
        eval_start_date = "2025-10-01"
        base_config = COMMON100_BASE_CONFIG
        thresholds = STRICT_THRESHOLDS
    else:
        raise SystemExit("Unsupported transition: {0} -> {1}".format(from_stage, to_stage))
    stage_items = _write_promoted_stage(
        state=state,
        suite_dir=suite_dir,
        rows=promoted,
        stage=to_stage,
        base_config=base_config,
        eval_stride=eval_stride,
        eval_start_date=eval_start_date,
        thresholds=thresholds,
    )
    state.setdefault("stages", {})[to_stage] = stage_items
    _write_json(state_path, state)
    return {"suite_state": state_path, "stage_summary": _write_stage_summary(state, to_stage)}


def prepare_mixed200_diagnostic(suite_state_path: str) -> Dict[str, Path]:
    state_path = Path(suite_state_path).resolve()
    state = _load_json(state_path)
    suite_dir = Path(state["suite_dir"])
    source_stage = "recent_daily" if "recent_daily" in state.get("stages", {}) else "stage2"
    rows = _read_stage_summary(state, source_stage)
    promoted = _filter_rows(rows, STRICT_THRESHOLDS, require_naive_gap=True)
    best = select_best_candidate(promoted)
    if best is None:
        raise SystemExit("No accepted candidate is available for mixed200 diagnostic.")
    stage_items = _write_promoted_stage(
        state=state,
        suite_dir=suite_dir,
        rows=pd.DataFrame([best]),
        stage="mixed200",
        base_config=MIXED200_BASE_CONFIG,
        eval_stride=21,
        eval_start_date=None,
        thresholds=STRICT_THRESHOLDS,
    )
    state.setdefault("stages", {})["mixed200"] = stage_items
    _write_json(state_path, state)
    return {"suite_state": state_path, "stage_summary": _write_stage_summary(state, "mixed200")}


def submit_stage(suite_state_path: str, stage: str, gpu: str = "L4", limit: Optional[int] = None) -> Dict[str, object]:
    state_path = Path(suite_state_path).resolve()
    state = _load_json(state_path)
    submitted = 0
    for item in state.get("stages", {}).get(stage, []):
        if limit is not None and submitted >= limit:
            break
        if item.get("status") not in ("generated", "submit_failed"):
            continue
        _run_benchmark_remote(str(item["manifest_path"]), gpu=gpu, detach=True)
        detached_state_path = Path(item["output_dir"]) / "detached_state.json"
        detached_state = _load_json(detached_state_path)
        item.update(
            {
                "status": "submitted",
                "gpu": gpu,
                "detached_state_path": str(detached_state_path),
                "function_call_id": detached_state.get("function_call_id"),
                "function_call_dashboard_url": detached_state.get(
                    "function_call_dashboard_url"
                ),
                "submitted_at_unix": detached_state.get("submitted_at_unix"),
            }
        )
        submitted += 1
        _write_json(state_path, state)
    return {"submitted": submitted, "suite_state": str(state_path)}


def collect_stage(suite_state_path: str, stage: str, timeout_seconds: float = 1.0) -> Dict[str, object]:
    state_path = Path(suite_state_path).resolve()
    state = _load_json(state_path)
    collected = 0
    still_running = 0
    failed = 0
    for item in state.get("stages", {}).get(stage, []):
        if item.get("status") == "collected":
            continue
        call_id = item.get("function_call_id")
        detached_state_path = item.get("detached_state_path")
        if not call_id or not detached_state_path:
            continue
        try:
            call = modal.FunctionCall.from_id(str(call_id))
            artifacts = call.get(timeout=timeout_seconds)
        except ModalTimeoutError:
            item["status"] = "submitted"
            still_running += 1
            continue
        except TimeoutError:
            item["status"] = "submitted"
            still_running += 1
            continue
        except Exception as exc:
            item["status"] = "collect_failed"
            item["error"] = str(exc)
            failed += 1
            continue
        detached_state = _load_json(Path(detached_state_path))
        paths = _collect_detached_benchmark_artifacts(
            state=detached_state,
            artifacts=artifacts,
            state_path=Path(detached_state_path),
        )
        item.update(
            {
                "status": "collected",
                "outputs": {key: str(value) for key, value in paths.items()},
                "collected_at_unix": time.time(),
            }
        )
        collected += 1
        _write_json(state_path, state)
    summary_path = _write_stage_summary(state, stage)
    _write_json(state_path, state)
    return {
        "collected": collected,
        "still_running": still_running,
        "failed": failed,
        "stage_summary": str(summary_path),
        "suite_state": str(state_path),
    }


def write_selected_base_model(suite_state_path: str, stage: str = "recent_daily") -> Path:
    state_path = Path(suite_state_path).resolve()
    state = _load_json(state_path)
    suite_dir = Path(state["suite_dir"])
    rows = _read_stage_summary(state, stage)
    if rows.empty and stage == "recent_daily":
        stage = "stage2"
        rows = _read_stage_summary(state, stage)
    selected = select_best_candidate(_filter_rows(rows, STRICT_THRESHOLDS, require_naive_gap=True))
    output_path = suite_dir / "selected_base_model.json"
    if selected is None:
        payload = {
            "accepted": False,
            "selection_stage": stage,
            "reason": "No candidate passed strict validation.",
            "ranked_candidates": _rank_rows(rows).head(20).to_dict(orient="records"),
        }
    else:
        payload = {"accepted": True, "selection_stage": stage, **selected}
    _write_json(output_path, payload)
    return output_path


def select_best_candidate(rows: pd.DataFrame) -> Optional[Dict[str, object]]:
    if rows.empty:
        return None
    ranked = _rank_rows(rows)
    if ranked.empty:
        return None
    return ranked.iloc[0].to_dict()


def _rank_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    frame = rows.copy()
    for column in (
        "rank_ic_gap_to_best_naive",
        "daily_rank_ic_positive_rate",
        "directional_accuracy",
        "spearman_rank_ic",
        "pred_abs_return_multiple",
    ):
        if column not in frame.columns:
            frame[column] = None
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["calibration_distortion"] = (frame["pred_abs_return_multiple"] - 1.0).abs()
    return frame.sort_values(
        [
            "rank_ic_gap_to_best_naive",
            "daily_rank_ic_positive_rate",
            "directional_accuracy",
            "spearman_rank_ic",
            "calibration_distortion",
        ],
        ascending=[False, False, False, False, True],
    )


def _write_candidate_manifest(
    suite_dir: Path,
    stage: str,
    candidate: CandidateSpec,
    base_config: str,
    eval_stride: int,
    calibration_modes: Sequence[str],
    thresholds: Dict[str, float],
    eval_start_date: Optional[str] = None,
) -> Path:
    stage_dir = suite_dir / stage / candidate.name
    manifest_dir = suite_dir / "manifests" / stage
    manifest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "kronos_base_search_{0}_{1}".format(stage, candidate.name),
        "stage": stage,
        "base_config": base_config,
        "output_dir": str(stage_dir),
        "enforce_ridge_gate": False,
        "references": {},
        "min_assets_per_date": 25,
        "calibration_lookback": 20,
        "promotion_thresholds": thresholds,
        "experiments": [
            {
                "name": candidate.name,
                "description": _candidate_description(candidate, stage, eval_stride),
                "overrides": candidate.overrides(
                    eval_stride=eval_stride,
                    eval_start_date=eval_start_date,
                ),
                "calibration_modes": list(calibration_modes),
            }
        ],
    }
    manifest_path = manifest_dir / "{0}.yaml".format(candidate.name)
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path


def _write_promoted_stage(
    state: Dict[str, object],
    suite_dir: Path,
    rows: pd.DataFrame,
    stage: str,
    base_config: str,
    eval_stride: int,
    eval_start_date: Optional[str],
    thresholds: Dict[str, float],
) -> List[Dict[str, object]]:
    candidate_grid = _candidate_grid_by_name(Path(state["suite_dir"]) / "candidate_grid.csv")
    items = []
    if rows.empty:
        return items
    for base_experiment, group in rows.groupby("base_experiment", sort=True):
        candidate = candidate_grid[str(base_experiment)]
        modes = sorted(set(group["calibration_mode"].astype(str)))
        manifest_path = _write_candidate_manifest(
            suite_dir=suite_dir,
            stage=stage,
            candidate=candidate,
            base_config=base_config,
            eval_stride=eval_stride,
            calibration_modes=modes,
            thresholds=thresholds,
            eval_start_date=eval_start_date,
        )
        items.append(_suite_item(candidate, manifest_path, suite_dir / stage / candidate.name))
    return items


def _suite_item(candidate: CandidateSpec, manifest_path: Path, output_dir: Path) -> Dict[str, object]:
    return {
        "candidate_name": candidate.name,
        "model_label": candidate.model_label,
        "model_id": candidate.model_id,
        "forecast_horizon": candidate.forecast_horizon,
        "decoding_name": candidate.decoding_name,
        "sample_count": candidate.sample_count,
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "status": "generated",
    }


def _read_stage_summary(state: Dict[str, object], stage: str) -> pd.DataFrame:
    rows = []
    for item in state.get("stages", {}).get(stage, []):
        summary_path = Path(item["output_dir"]) / "benchmark_summary.csv"
        if summary_path.exists():
            frame = pd.read_csv(summary_path)
            rows.append(_add_naive_rank_gap(frame, local_data_dir=state["local_data_dir"]))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _write_stage_summary(state: Dict[str, object], stage: str) -> Path:
    suite_dir = Path(state["suite_dir"])
    frame = _read_stage_summary(state, stage)
    output_name = {
        "stage1": "stage1_summary.csv",
        "stage2": "stage2_validation_summary.csv",
        "recent_daily": "recent_daily_validation_summary.csv",
        "mixed200": "mixed200_diagnostic_summary.csv",
    }.get(stage, "{0}_summary.csv".format(stage))
    path = suite_dir / output_name
    frame.to_csv(path, index=False)
    return path


def _add_naive_rank_gap(summary: pd.DataFrame, local_data_dir: str) -> pd.DataFrame:
    frame = summary.copy()
    gaps = []
    best_names = []
    best_values = []
    for _, row in frame.iterrows():
        output_dir = Path(str(row.get("output_dir", "")))
        predictions_path = output_dir / "predictions.csv"
        if not predictions_path.exists() or str(row.get("status")) != "completed":
            gaps.append(None)
            best_names.append(None)
            best_values.append(None)
            continue
        predictions = pd.read_csv(predictions_path)
        horizon = _row_forecast_horizon(row, predictions)
        baselines = build_naive_baseline_predictions(
            predictions=predictions,
            local_data_dir=Path(local_data_dir),
            forecast_horizon=horizon,
        )
        best_name = None
        best_rank_ic = None
        for name, baseline_predictions in baselines.items():
            metrics = _compute_report_metrics(baseline_predictions, min_assets_per_date=25)
            rank_ic = _as_float(metrics.get("spearman_rank_ic"))
            if rank_ic is not None and (best_rank_ic is None or rank_ic > best_rank_ic):
                best_name = name
                best_rank_ic = rank_ic
        row_rank_ic = _as_float(row.get("spearman_rank_ic"))
        gaps.append(row_rank_ic - best_rank_ic if row_rank_ic is not None and best_rank_ic is not None else None)
        best_names.append(best_name)
        best_values.append(best_rank_ic)
    frame["best_naive_baseline"] = best_names
    frame["best_naive_rank_ic"] = best_values
    frame["rank_ic_gap_to_best_naive"] = gaps
    return frame


def _row_forecast_horizon(row: pd.Series, predictions: pd.DataFrame) -> int:
    value = _as_float(row.get("forecast_horizon"))
    if value is not None:
        return int(value)
    if "forecast_horizon" in predictions.columns and not predictions.empty:
        value = _as_float(predictions["forecast_horizon"].iloc[0])
        if value is not None:
            return int(value)
    return 5


def _filter_rows(
    rows: pd.DataFrame,
    thresholds: Dict[str, float],
    require_naive_gap: bool = False,
) -> pd.DataFrame:
    if rows.empty:
        return rows
    frame = rows.copy()
    mask = frame["status"].astype(str).eq("completed")
    mask &= _numeric(frame, "pearson_ic") >= thresholds["min_return_ic"]
    mask &= _numeric(frame, "spearman_rank_ic") >= thresholds["min_return_rank_ic"]
    mask &= _numeric(frame, "directional_accuracy") >= thresholds["min_directional_accuracy"]
    mask &= _numeric(frame, "daily_ic_positive_rate") >= thresholds["min_daily_positive_rate"]
    mask &= _numeric(frame, "daily_rank_ic_positive_rate") >= thresholds["min_daily_positive_rate"]
    mask &= _numeric(frame, "pred_abs_return_multiple") >= thresholds["min_pred_abs_return_multiple"]
    mask &= _numeric(frame, "pred_abs_return_multiple") <= thresholds["max_pred_abs_return_multiple"]
    mask &= _numeric(frame, "positive_rate_gap") <= thresholds["max_positive_rate_gap"]
    mask &= _numeric(frame, "output_total_invalid_ohlc_rate") < thresholds["max_invalid_ohlc_rate"]
    if require_naive_gap:
        mask &= _numeric(frame, "rank_ic_gap_to_best_naive") >= NAIVE_RANK_IC_MARGIN
    return frame.loc[mask].copy()


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([float("nan")] * len(frame), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _candidate_grid_by_name(path: Path) -> Dict[str, CandidateSpec]:
    frame = pd.read_csv(path)
    result = {}
    for row in frame.to_dict(orient="records"):
        candidate = CandidateSpec(
            name=str(row["name"]),
            model_label=str(row["model_label"]),
            model_id=str(row["model_id"]),
            forecast_horizon=int(row["forecast_horizon"]),
            decoding_name=str(row["decoding_name"]),
            temperature=float(row["temperature"]),
            top_k=int(row["top_k"]),
            top_p=float(row["top_p"]),
            sample_count=int(row["sample_count"]),
        )
        result[candidate.name] = candidate
    return result


def _candidate_description(candidate: CandidateSpec, stage: str, eval_stride: int) -> str:
    return (
        "{0} horizon-{1} {2} Kronos search candidate for {3} "
        "with eval_stride {4}."
    ).format(
        candidate.model_id,
        candidate.forecast_horizon,
        candidate.decoding_name,
        stage,
        eval_stride,
    )


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _json_safe(payload)
    path.write_text(
        json.dumps(safe_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the aggressive Kronos base search.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--output-dir", default=DEFAULT_SUITE_DIR)
    submit = subparsers.add_parser("submit-stage")
    submit.add_argument("--state", required=True)
    submit.add_argument("--stage", required=True)
    submit.add_argument("--gpu", default="L4")
    submit.add_argument("--limit", type=int, default=None)
    collect = subparsers.add_parser("collect-stage")
    collect.add_argument("--state", required=True)
    collect.add_argument("--stage", required=True)
    collect.add_argument("--timeout-seconds", type=float, default=1.0)
    prepare = subparsers.add_parser("prepare-next")
    prepare.add_argument("--state", required=True)
    prepare.add_argument("--from-stage", required=True)
    prepare.add_argument("--to-stage", required=True)
    mixed = subparsers.add_parser("prepare-mixed200")
    mixed.add_argument("--state", required=True)
    select = subparsers.add_parser("select-base")
    select.add_argument("--state", required=True)
    select.add_argument("--stage", default="recent_daily")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.command == "generate":
        result = generate_search_suite(args.output_dir)
    elif args.command == "submit-stage":
        result = submit_stage(args.state, args.stage, gpu=args.gpu, limit=args.limit)
    elif args.command == "collect-stage":
        result = collect_stage(args.state, args.stage, timeout_seconds=args.timeout_seconds)
    elif args.command == "prepare-next":
        result = prepare_stage_from_previous(args.state, args.from_stage, args.to_stage)
    elif args.command == "prepare-mixed200":
        result = prepare_mixed200_diagnostic(args.state)
    elif args.command == "select-base":
        result = {"selected_base_model": str(write_selected_base_model(args.state, args.stage))}
    else:
        raise SystemExit("Unknown command: {0}".format(args.command))
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
