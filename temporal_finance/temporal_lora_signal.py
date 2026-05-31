"""Temporal LoRA signal-run orchestration, scoring, and decision logic."""

from __future__ import annotations

import base64
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from temporal_finance.evaluation import compute_metrics
from temporal_finance.kronos import KRONOS_COLUMNS
from temporal_finance.kronos_benchmark import calibrate_kronos_predictions
from temporal_finance.kronos_experiments import compute_kronos_prediction_diagnostics
from temporal_finance.kronos_output_evaluation import compute_daily_cross_section_metrics
from temporal_finance.temporal_lora_config import (
    TemporalLoraAdapterConfig,
    TemporalLoraSignalConfig,
)
from temporal_finance.temporal_lora_data import (
    TemporalLoraExample,
    build_naive_predictions,
    build_normalized_sequence,
    build_prediction_row_from_forecast,
    build_training_examples_for_window,
    build_validation_examples_for_window,
    build_window_plans,
    load_signal_histories,
    summarize_examples,
    temporal_lora_window_plans_to_frame,
    time_features,
)


MODEL_VARIANTS = [
    "frozen_base",
    "slow",
    "medium",
    "fast",
    "weighted_ensemble",
    "equal_ensemble",
    "stale_ensemble",
]


@dataclass
class TemporalLoraSignalRunResult:
    output_dir: Path
    artifacts: Dict[str, Path]
    decision: Dict[str, object]


class KronosTemporalLoraEngine:
    """Actual Kronos-small engine used by the Modal signal runner."""

    def __init__(self, config: TemporalLoraSignalConfig):
        self.config = config
        self._load_kronos()

    def _load_kronos(self) -> None:
        try:
            import torch
            from model import Kronos, KronosPredictor, KronosTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Kronos is not importable. Use the Modal runner or add the "
                "upstream Kronos repo to PYTHONPATH."
            ) from exc

        from temporal_finance.kronos import _resolve_device
        from temporal_finance.temporal_lora import inject_lora_adapters

        self.torch = torch
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)
        kwargs = {}
        if self.config.kronos_cache_dir:
            kwargs["cache_dir"] = self.config.kronos_cache_dir
        self.tokenizer = KronosTokenizer.from_pretrained(
            self.config.tokenizer_id,
            **kwargs,
        )
        self.model = Kronos.from_pretrained(self.config.model_id, **kwargs)
        self.device = _resolve_device(self.config.device, torch)
        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

        structural = self._structural_adapter_config()
        self.injection_report = inject_lora_adapters(
            self.model,
            rank=structural.rank,
            alpha=structural.alpha,
            dropout=structural.dropout,
            target_prefixes=self.config.target_module_prefixes,
            exclude_keywords=self.config.exclude_module_keywords,
        )
        self.model = self.model.to(self.device)
        self.predictor = KronosPredictor(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_context=self.config.context_length,
            clip=self.config.clip,
        )

    def _structural_adapter_config(self) -> TemporalLoraAdapterConfig:
        first = next(iter(self.config.adapters.values()))
        for adapter in self.config.adapters.values():
            if (
                adapter.rank != first.rank
                or float(adapter.alpha) != float(first.alpha)
                or float(adapter.dropout) != float(first.dropout)
            ):
                raise RuntimeError(
                    "The first signal runner expects slow/medium/fast adapters "
                    "to share rank, alpha, and dropout so one LoRA-injected model "
                    "can swap adapter states."
                )
        return first

    def train_adapter(
        self,
        adapter_name: str,
        examples: List[TemporalLoraExample],
        histories: Dict[str, pd.DataFrame],
        adapter_config: TemporalLoraAdapterConfig,
        output_path: Path,
    ) -> Dict[str, object]:
        from temporal_finance.temporal_lora import (
            lora_parameters,
            mark_only_lora_trainable,
            reset_lora_parameters,
            save_lora_state,
        )

        if not examples:
            raise RuntimeError("No training examples for adapter {0}.".format(adapter_name))
        reset_lora_parameters(self.model)
        mark_only_lora_trainable(self.model)
        trainable = list(lora_parameters(self.model))
        if not trainable:
            raise RuntimeError("No trainable LoRA parameters were found.")

        torch = self.torch
        seed_offsets = {"slow": 101, "medium": 211, "fast": 307}
        rng = random.Random(
            self.config.random_seed + seed_offsets.get(adapter_name, 997)
        )
        shuffled = list(examples)
        rng.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * 0.1)) if len(shuffled) >= 10 else 0
        validation = shuffled[:val_count]
        train = shuffled[val_count:] or shuffled

        optimizer = torch.optim.AdamW(
            trainable,
            lr=adapter_config.learning_rate,
            weight_decay=adapter_config.weight_decay,
        )
        best_loss = math.inf
        best_state = None
        stale_epochs = 0
        epoch_rows = []
        started = time.monotonic()
        for epoch in range(1, adapter_config.epochs + 1):
            rng.shuffle(train)
            self.model.train()
            train_losses = []
            for batch in _chunked(train, adapter_config.batch_size):
                tensors = self._training_batch_to_tensors(batch, histories)
                optimizer.zero_grad(set_to_none=True)
                loss = self._token_loss(tensors)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            val_loss = None
            if validation:
                val_loss = self._evaluate_token_loss(
                    validation,
                    histories,
                    adapter_config.batch_size,
                )
                score = val_loss
            else:
                score = float(np.mean(train_losses)) if train_losses else math.inf
            if score < best_loss:
                best_loss = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                    if ".lora_a." in key or ".lora_b." in key
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(train_losses)) if train_losses else None,
                    "validation_loss": val_loss,
                }
            )
            if validation and stale_epochs > adapter_config.patience:
                break

        if best_state is not None:
            self.model_state_update(best_state)
        metadata = {
            "adapter_name": adapter_name,
            "example_count": len(examples),
            "train_example_count": len(train),
            "validation_example_count": len(validation),
            "best_loss": None if math.isinf(best_loss) else best_loss,
            "epochs_ran": len(epoch_rows),
            "runtime_seconds": time.monotonic() - started,
            "epoch_metrics": epoch_rows,
        }
        save_lora_state(self.model, output_path, metadata=metadata)
        metadata["adapter_path"] = str(output_path)
        return metadata

    def model_state_update(self, state: Dict[str, object]) -> None:
        current = self.model.state_dict()
        for key, value in state.items():
            if key in current:
                current[key] = value.to(current[key].device)
        self.model.load_state_dict(current, strict=False)

    def predict_examples(
        self,
        examples: List[TemporalLoraExample],
        histories: Dict[str, pd.DataFrame],
        variant: str,
        adapter_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        from temporal_finance.temporal_lora import load_lora_state, reset_lora_parameters

        if adapter_path is None:
            reset_lora_parameters(self.model)
        else:
            load_lora_state(self.model, adapter_path, strict=True)
        self.model.eval()

        rows = []
        batch_size = max(1, min(64, self.config.sample_count * 4))
        for batch in _chunked(examples, batch_size):
            contexts = []
            x_timestamps = []
            y_timestamps = []
            for example in batch:
                history = histories[example.ticker]
                context = history.iloc[
                    example.context_start_index : example.context_end_index + 1
                ][KRONOS_COLUMNS]
                contexts.append(context)
                context_index = pd.DatetimeIndex(context.index)
                target_index = pd.DatetimeIndex(
                    [history.index[example.target_index]]
                )
                x_timestamps.append(pd.Series(context_index))
                y_timestamps.append(pd.Series(target_index))
            forecasts = self.predictor.predict_batch(
                df_list=contexts,
                x_timestamp_list=x_timestamps,
                y_timestamp_list=y_timestamps,
                pred_len=1,
                T=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                sample_count=self.config.sample_count,
                verbose=False,
            )
            for example, forecast in zip(batch, forecasts):
                rows.append(
                    build_prediction_row_from_forecast(
                        example=example,
                        history=histories[example.ticker],
                        forecast=forecast,
                        variant=variant,
                    )
                )
        return pd.DataFrame(rows)

    def _training_batch_to_tensors(
        self,
        batch: List[TemporalLoraExample],
        histories: Dict[str, pd.DataFrame],
    ) -> Dict[str, object]:
        arrays = []
        stamps = []
        for example in batch:
            normalized, _, _, _ = build_normalized_sequence(
                histories[example.ticker],
                example,
                clip=self.config.clip,
            )
            full_index = histories[example.ticker].index[
                example.context_start_index : example.target_index + 1
            ]
            arrays.append(normalized)
            stamps.append(time_features(full_index))
        torch = self.torch
        x = torch.from_numpy(np.stack(arrays, axis=0)).to(self.device)
        stamp = torch.from_numpy(np.stack(stamps, axis=0)).to(self.device)
        with torch.no_grad():
            s1_tokens, s2_tokens = self.tokenizer.encode(x, half=True)
        return {
            "s1_input": s1_tokens[:, :-1].long(),
            "s2_input": s2_tokens[:, :-1].long(),
            "s1_target": s1_tokens[:, 1:].long(),
            "s2_target": s2_tokens[:, 1:].long(),
            "stamp": stamp[:, :-1, :].float(),
        }

    def _token_loss(self, tensors: Dict[str, object]):
        torch = self.torch
        s1_logits, s2_logits = self.model(
            tensors["s1_input"],
            tensors["s2_input"],
            stamp=tensors["stamp"],
            use_teacher_forcing=True,
            s1_targets=tensors["s1_target"],
        )
        s1_target = tensors["s1_target"]
        s2_target = tensors["s2_target"]
        return torch.nn.functional.cross_entropy(
            s1_logits.reshape(-1, s1_logits.shape[-1]),
            s1_target.reshape(-1),
        ) + torch.nn.functional.cross_entropy(
            s2_logits.reshape(-1, s2_logits.shape[-1]),
            s2_target.reshape(-1),
        )

    def _evaluate_token_loss(
        self,
        examples: List[TemporalLoraExample],
        histories: Dict[str, pd.DataFrame],
        batch_size: int,
    ) -> float:
        self.model.eval()
        losses = []
        with self.torch.no_grad():
            for batch in _chunked(examples, batch_size):
                losses.append(
                    float(self._token_loss(self._training_batch_to_tensors(batch, histories)).cpu())
                )
        return float(np.mean(losses)) if losses else math.inf


def run_temporal_lora_signal(
    config: TemporalLoraSignalConfig,
    engine: Optional[object] = None,
    progress_callback: Optional[object] = None,
) -> TemporalLoraSignalRunResult:
    config.validate()
    output_dir = config.resolved_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    _emit(progress_callback, "Loading Temporal LoRA histories.")
    histories = load_signal_histories(config)
    plans = build_window_plans(config, histories)
    temporal_lora_window_plans_to_frame(plans).to_csv(
        output_dir / "window_manifest.csv",
        index=False,
    )
    if engine is None:
        _emit(progress_callback, "Loading Kronos Temporal LoRA engine.")
        engine = KronosTemporalLoraEngine(config)

    predictions_by_variant: Dict[str, List[pd.DataFrame]] = {
        variant: [] for variant in MODEL_VARIANTS
    }
    naive_predictions_by_variant: Dict[str, List[pd.DataFrame]] = {}
    all_examples: List[TemporalLoraExample] = []
    adapter_registry: Dict[str, object] = {
        "model_id": config.model_id,
        "tokenizer_id": config.tokenizer_id,
        "target_module_prefixes": config.target_module_prefixes,
        "adapters": [],
    }
    injection_report = getattr(engine, "injection_report", None)
    if injection_report is not None:
        adapter_registry["injection_report"] = {
            "target_names": list(injection_report.target_names),
            "target_count": int(injection_report.target_count),
            "trainable_parameter_count": int(
                injection_report.trainable_parameter_count
            ),
            "total_parameter_count": int(injection_report.total_parameter_count),
        }

    stale_adapter_paths = None
    completed_windows = 0
    for plan in plans:
        _emit(progress_callback, "Starting Temporal LoRA window {0}.".format(plan.window_id))
        validation_examples = build_validation_examples_for_window(
            histories=histories,
            plan=plan,
            context_length=config.context_length,
            stride=config.eval_stride,
        )
        all_examples.extend(validation_examples)
        adapter_paths = {}
        for adapter_name in ("slow", "medium", "fast"):
            adapter_config = config.adapters[adapter_name]
            examples = build_training_examples_for_window(
                histories=histories,
                plan=plan,
                adapter_name=adapter_name,
                context_length=config.context_length,
                stride=adapter_config.train_stride,
                max_examples=adapter_config.max_examples,
            )
            all_examples.extend(examples)
            path = output_dir / "adapters" / plan.window_id / f"{adapter_name}.pt"
            metadata = engine.train_adapter(
                adapter_name=adapter_name,
                examples=examples,
                histories=histories,
                adapter_config=adapter_config,
                output_path=path,
            )
            adapter_paths[adapter_name] = path
            adapter_registry["adapters"].append(
                {
                    "window_id": plan.window_id,
                    "adapter_name": adapter_name,
                    **metadata,
                }
            )
        if stale_adapter_paths is None:
            stale_adapter_paths = dict(adapter_paths)

        raw_window_frames = {}
        raw_window_frames["frozen_base"] = engine.predict_examples(
            validation_examples,
            histories,
            variant="frozen_base",
            adapter_path=None,
        )
        window_frames = {
            "frozen_base": _calibrate_variant(
                raw_window_frames["frozen_base"],
                config,
            )
        }
        for adapter_name in ("slow", "medium", "fast"):
            raw_window_frames[adapter_name] = engine.predict_examples(
                validation_examples,
                histories,
                variant=adapter_name,
                adapter_path=adapter_paths[adapter_name],
            )
            window_frames[adapter_name] = _calibrate_variant(
                raw_window_frames[adapter_name],
                config,
            )
        window_frames["weighted_ensemble"] = _calibrate_variant(
            ensemble_predictions(
                {
                    name: raw_window_frames[name]
                    for name in ("slow", "medium", "fast")
                },
                weights=config.ensemble_weights,
                variant="weighted_ensemble",
            ),
            config,
        )
        window_frames["equal_ensemble"] = _calibrate_variant(
            ensemble_predictions(
                {
                    name: raw_window_frames[name]
                    for name in ("slow", "medium", "fast")
                },
                weights={"slow": 1.0, "medium": 1.0, "fast": 1.0},
                variant="equal_ensemble",
            ),
            config,
        )
        stale_frames = {}
        for adapter_name in ("slow", "medium", "fast"):
            stale_frames[adapter_name] = engine.predict_examples(
                validation_examples,
                histories,
                variant="stale_{0}".format(adapter_name),
                adapter_path=stale_adapter_paths[adapter_name],
            )
        window_frames["stale_ensemble"] = _calibrate_variant(
            ensemble_predictions(
                stale_frames,
                weights=config.ensemble_weights,
                variant="stale_ensemble",
            ),
            config,
        )

        for variant, frame in window_frames.items():
            predictions_by_variant[variant].append(frame)

        for variant, frame in build_naive_predictions(
            validation_examples,
            histories,
        ).items():
            frame = frame.copy()
            frame["calibration_mode"] = "none"
            naive_predictions_by_variant.setdefault(variant, []).append(frame)

        completed_windows += 1
        elapsed = time.monotonic() - started
        if elapsed * config.modal_gpu_seconds_price >= config.budget_cap_usd:
            _emit(
                progress_callback,
                "Budget cap reached after window {0}; writing partial outputs.".format(
                    plan.window_id
                ),
            )
            break

    examples_summary = summarize_examples(all_examples)
    examples_summary.to_csv(output_dir / "training_examples_summary.csv", index=False)
    _write_json(output_dir / "adapter_registry.json", adapter_registry)

    prediction_outputs: Dict[str, pd.DataFrame] = {}
    for variant, frames in predictions_by_variant.items():
        prediction_outputs[variant] = _concat_frames(frames)
    for variant, frames in naive_predictions_by_variant.items():
        prediction_outputs[variant] = _concat_frames(frames)

    artifacts: Dict[str, Path] = {
        "window_manifest": output_dir / "window_manifest.csv",
        "training_examples_summary": output_dir / "training_examples_summary.csv",
        "adapter_registry": output_dir / "adapter_registry.json",
    }
    for variant, frame in prediction_outputs.items():
        path = output_dir / "predictions_{0}.csv".format(variant)
        frame.to_csv(path, index=False)
        artifacts["predictions_{0}".format(variant)] = path

    metrics_by_window, daily_metrics = build_metrics_tables(
        prediction_outputs,
        min_assets_per_date=config.min_assets_per_date,
    )
    metrics_by_window_path = output_dir / "metrics_by_window.csv"
    daily_metrics_path = output_dir / "daily_cross_section_metrics.csv"
    metrics_by_window.to_csv(metrics_by_window_path, index=False)
    daily_metrics.to_csv(daily_metrics_path, index=False)
    artifacts["metrics_by_window"] = metrics_by_window_path
    artifacts["daily_cross_section_metrics"] = daily_metrics_path

    decision = decide_temporal_lora_signal(metrics_by_window)
    signal_decision_path = output_dir / "signal_decision.json"
    _write_json(signal_decision_path, decision)
    artifacts["signal_decision"] = signal_decision_path

    run_state = {
        "status": "completed" if completed_windows == len(plans) else "partial",
        "completed_windows": completed_windows,
        "requested_windows": len(plans),
        "runtime_seconds": time.monotonic() - started,
        "estimated_modal_l4_cost_usd": (
            (time.monotonic() - started) * config.modal_gpu_seconds_price
        ),
        "budget_cap_usd": config.budget_cap_usd,
    }
    run_state_path = output_dir / "run_state.json"
    _write_json(run_state_path, run_state)
    artifacts["run_state"] = run_state_path
    return TemporalLoraSignalRunResult(
        output_dir=output_dir,
        artifacts=artifacts,
        decision=decision,
    )


def ensemble_predictions(
    frames_by_adapter: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    variant: str,
) -> pd.DataFrame:
    if not frames_by_adapter:
        return pd.DataFrame()
    normalized_weights = _normalize_weights(weights)
    keys = ["ticker", "window_id", "date", "target_date"]
    ordered = {
        name: frame.sort_values(keys).reset_index(drop=True)
        for name, frame in frames_by_adapter.items()
        if name in normalized_weights
    }
    if not ordered:
        raise ValueError("No adapter frames matched ensemble weights.")
    first = next(iter(ordered.values())).copy()
    for name, frame in ordered.items():
        if len(frame) != len(first):
            raise ValueError(
                "Adapter frame {0} has {1} rows, expected {2}.".format(
                    name,
                    len(frame),
                    len(first),
                )
            )
    result = first.copy()
    pred_return = sum(
        ordered[name]["pred_return"].astype(float) * normalized_weights[name]
        for name in ordered
    )
    result["variant"] = variant
    result["pred_return"] = pred_return
    result["pred_5d_return"] = pred_return
    result["pred_close_target"] = result["last_context_close"].astype(float) * (
        1.0 + pred_return
    )
    result["pred_close_t5"] = result["pred_close_target"]
    for base_name, columns in {
        "open": ("pred_open_target", "pred_open_t5"),
        "high": ("pred_high_target", "pred_high_t5"),
        "low": ("pred_low_target", "pred_low_t5"),
        "volume": ("pred_volume_target", "pred_volume_t5"),
    }.items():
        combined = sum(
            ordered[name][columns[0]].astype(float) * normalized_weights[name]
            for name in ordered
        )
        result[columns[0]] = combined
        result[columns[1]] = combined
    high = np.maximum.reduce(
        [
            result["pred_high_target"].astype(float).to_numpy(),
            result["pred_open_target"].astype(float).to_numpy(),
            result["pred_close_target"].astype(float).to_numpy(),
        ]
    )
    low = np.minimum.reduce(
        [
            result["pred_low_target"].astype(float).to_numpy(),
            result["pred_open_target"].astype(float).to_numpy(),
            result["pred_close_target"].astype(float).to_numpy(),
        ]
    )
    result["pred_high_target"] = high
    result["pred_high_t5"] = high
    result["pred_low_target"] = low
    result["pred_low_t5"] = low
    return result


def build_metrics_tables(
    predictions_by_variant: Dict[str, pd.DataFrame],
    min_assets_per_date: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    daily_rows = []
    for variant, frame in predictions_by_variant.items():
        if frame.empty:
            continue
        for window_id, group in _window_groups(frame):
            metrics = _score_predictions(group, min_assets_per_date)
            metrics_rows.append(
                {
                    "window_id": window_id,
                    "variant": variant,
                    **metrics,
                }
            )
            daily = compute_daily_cross_section_metrics(
                group,
                min_assets_per_date=min_assets_per_date,
            )
            if not daily.empty:
                daily = daily.copy()
                daily["window_id"] = window_id
                daily["variant"] = variant
                daily_rows.append(daily)
    return (
        pd.DataFrame(metrics_rows),
        _concat_frames(daily_rows),
    )


def decide_temporal_lora_signal(metrics_by_window: pd.DataFrame) -> Dict[str, object]:
    if metrics_by_window.empty:
        return {
            "accepted": False,
            "decision": "fail",
            "reason": "No metrics were produced.",
            "recommended_next_step": "Fix the signal-run pipeline before spending on GPU.",
        }
    pooled = metrics_by_window[metrics_by_window["window_id"] == "pooled"]
    weighted = _metric_row(pooled, "weighted_ensemble")
    frozen = _metric_row(pooled, "frozen_base")
    if weighted is None or frozen is None:
        return {
            "accepted": False,
            "decision": "fail",
            "reason": "Missing pooled weighted_ensemble or frozen_base metrics.",
            "recommended_next_step": "Inspect prediction artifacts and rerun the failed variant.",
        }
    weighted_rank = _rank_metric(weighted)
    frozen_rank = _rank_metric(frozen)
    rank_ic_lift = _safe_subtract(weighted_rank, frozen_rank)
    directional_lift = _safe_subtract(
        _as_float(weighted.get("directional_accuracy")),
        _as_float(frozen.get("directional_accuracy")),
    )
    windows_improved = _count_windows_improved(metrics_by_window)
    naive_rows = pooled[pooled["variant"].astype(str).str.startswith("naive_")]
    best_naive_variant = None
    best_naive_rank = None
    if not naive_rows.empty:
        naive_ranks = naive_rows.apply(lambda row: _rank_metric(row), axis=1)
        if naive_ranks.notna().any():
            idx = naive_ranks.astype(float).idxmax()
            best_naive_variant = str(naive_rows.loc[idx, "variant"])
            best_naive_rank = float(naive_ranks.loc[idx])
    best_naive_gap = _safe_subtract(weighted_rank, best_naive_rank)
    pred_abs_multiple = _as_float(weighted.get("pred_abs_return_multiple"))
    invalid_rate = _invalid_ohlc_rate(weighted)

    gates = {
        "rank_ic_lift_gte_0_015": rank_ic_lift is not None and rank_ic_lift >= 0.015,
        "directional_lift_gte_0_01": (
            directional_lift is not None and directional_lift >= 0.01
        ),
        "two_of_three_windows_improved": windows_improved >= 2,
        "beats_best_naive_rank_ic": best_naive_gap is not None and best_naive_gap > 0,
        "return_scale_in_range": (
            pred_abs_multiple is not None and 0.5 <= pred_abs_multiple <= 2.0
        ),
        "invalid_ohlc_lt_0_03": invalid_rate is not None and invalid_rate < 0.03,
    }
    stale = _metric_row(pooled, "stale_ensemble")
    stale_rank = _rank_metric(stale) if stale is not None else None
    stale_worse_than_rolling = (
        stale_rank is not None and weighted_rank is not None and stale_rank < weighted_rank
    )
    all_required = all(gates.values())
    if all_required and stale_worse_than_rolling:
        decision = "strong_pass"
        accepted = True
        reason = (
            "Weighted Temporal LoRA ensemble passed all gates and rolling adapters "
            "beat stale adapters."
        )
        next_step = "Scale to 200 assets and six or more rolling windows."
    elif gates["rank_ic_lift_gte_0_015"] and windows_improved >= 2:
        decision = "weak_pass"
        accepted = True
        reason = "RankIC lift appears, but one or more secondary gates are weak."
        next_step = "Run a cheap shuffled-label sanity check before scaling."
    else:
        decision = "fail"
        accepted = False
        reason = "No consistent Temporal LoRA lift over the frozen base."
        next_step = "Do not scale; revisit data, objective, or base model."
    return {
        "accepted": accepted,
        "decision": decision,
        "reason": reason,
        "primary_rank_ic_lift": rank_ic_lift,
        "directional_lift": directional_lift,
        "windows_improved_count": windows_improved,
        "best_naive_baseline": best_naive_variant,
        "best_naive_rank_ic_gap": best_naive_gap,
        "weighted_rank_ic": weighted_rank,
        "frozen_base_rank_ic": frozen_rank,
        "stale_ensemble_rank_ic": stale_rank,
        "stale_worse_than_rolling": stale_worse_than_rolling,
        "pred_abs_return_multiple": pred_abs_multiple,
        "invalid_ohlc_rate": invalid_rate,
        "gates": gates,
        "recommended_next_step": next_step,
    }


def serialize_temporal_lora_artifacts(output_dir: str | Path) -> Dict[str, object]:
    root = Path(output_dir)
    artifacts: Dict[str, object] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if path.suffix.lower() == ".pt":
            artifacts[relative] = {
                "encoding": "base64",
                "contents": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        elif path.suffix.lower() in {".csv", ".json", ".txt", ".yaml", ".yml"}:
            artifacts[relative] = path.read_text(encoding="utf-8")
    return artifacts


def write_temporal_lora_artifacts(
    output_dir: str | Path,
    artifacts: Dict[str, object],
) -> Dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for relative, contents in artifacts.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(contents, dict) and contents.get("encoding") == "base64":
            path.write_bytes(base64.b64decode(str(contents["contents"])))
        else:
            path.write_text(str(contents), encoding="utf-8")
        paths[relative] = path
    return paths


def _calibrate_variant(
    frame: pd.DataFrame,
    config: TemporalLoraSignalConfig,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    return calibrate_kronos_predictions(
        frame,
        config.calibration_mode,
        lookback=config.calibration_lookback,
    )


def _score_predictions(
    predictions: pd.DataFrame,
    min_assets_per_date: int,
) -> Dict[str, object]:
    clean = predictions.dropna(subset=["pred_return", "actual_return"])
    if clean.empty:
        return {"sample_count": 0}
    metrics = compute_metrics(clean)
    metrics.update(compute_kronos_prediction_diagnostics(clean))
    daily = compute_daily_cross_section_metrics(
        clean,
        min_assets_per_date=min_assets_per_date,
    )
    metrics.update(_summarize_daily_metrics(daily))
    sample_count = metrics.get("sample_count") or 0
    invalid_count = metrics.get("invalid_pred_ohlc_count") or 0
    metrics["invalid_ohlc_rate"] = (
        float(invalid_count) / float(sample_count) if sample_count else None
    )
    return metrics


def _summarize_daily_metrics(daily: pd.DataFrame) -> Dict[str, object]:
    if daily.empty:
        return {
            "daily_window_count": 0,
            "daily_mean_ic": None,
            "daily_mean_rank_ic": None,
            "daily_ic_positive_rate": None,
            "daily_rank_ic_positive_rate": None,
            "daily_mean_directional_accuracy": None,
        }
    return {
        "daily_window_count": int(len(daily)),
        "daily_mean_ic": _safe_mean(daily["ic"]),
        "daily_mean_rank_ic": _safe_mean(daily["rank_ic"]),
        "daily_ic_positive_rate": _positive_rate(daily["ic"]),
        "daily_rank_ic_positive_rate": _positive_rate(daily["rank_ic"]),
        "daily_mean_directional_accuracy": _safe_mean(daily["directional_accuracy"]),
    }


def _window_groups(frame: pd.DataFrame):
    for window_id, group in frame.groupby("window_id", sort=True):
        yield window_id, group
    yield "pooled", frame


def _concat_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(float(value) for value in weights.values())
    if total <= 0:
        raise ValueError("Ensemble weights must have positive total weight.")
    return {str(key): float(value) / total for key, value in weights.items()}


def _metric_row(frame: pd.DataFrame, variant: str):
    rows = frame[frame["variant"] == variant]
    if rows.empty:
        return None
    return rows.iloc[0]


def _rank_metric(row) -> Optional[float]:
    if row is None:
        return None
    value = _as_float(row.get("daily_mean_rank_ic"))
    if value is not None:
        return value
    return _as_float(row.get("spearman_rank_ic"))


def _count_windows_improved(metrics_by_window: pd.DataFrame) -> int:
    count = 0
    for window_id, group in metrics_by_window.groupby("window_id", sort=True):
        if window_id == "pooled":
            continue
        weighted = _metric_row(group, "weighted_ensemble")
        frozen = _metric_row(group, "frozen_base")
        if weighted is None or frozen is None:
            continue
        weighted_rank = _rank_metric(weighted)
        frozen_rank = _rank_metric(frozen)
        if weighted_rank is not None and frozen_rank is not None and weighted_rank > frozen_rank:
            count += 1
    return count


def _invalid_ohlc_rate(row) -> Optional[float]:
    value = _as_float(row.get("invalid_ohlc_rate"))
    if value is not None:
        return value
    invalid = _as_float(row.get("invalid_pred_ohlc_count"))
    sample_count = _as_float(row.get("sample_count"))
    if invalid is None or not sample_count:
        return None
    return invalid / sample_count


def _safe_subtract(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return float(left - right)


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _safe_mean(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _positive_rate(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float((values > 0).mean())


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _chunked(items: List[object], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _emit(progress_callback: Optional[object], message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)
