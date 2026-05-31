# Temporal LoRA Signal Run

## Purpose

This document defines the first low-cost signal run for Temporal LoRAs. The goal is not to produce a research-grade result yet. The goal is to answer one practical question:

Can fixed Slow, Medium, and Fast LoRA adapters add measurable forward-window signal over the current frozen Kronos engineering base?

If the answer is yes, the project is worth scaling into a research-grade Temporal LoRA effort. If the answer is no, we should stop or change the data/objective before spending on a larger run.

## Current Starting Point

The current base is an engineering base, not a validated alpha base.

- Model family: `NeoQuasar/Kronos-small`
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-base`
- Data source: `dataset/data/kronos/combined_by_ticker`
- Data shape: daily Kronos-format OHLCVA files with `timestamps, open, high, low, close, volume, amount`
- Dataset coverage: 300 ticker files, mostly `2021-05-28` through `2026-05-27`
- Context length: `512`
- Forecast horizon: `1`

The frozen-base inference control should use the best engineering inference settings from the base search:

- `model_id: NeoQuasar/Kronos-small`
- `forecast_horizon: 1`
- `temperature: 0.6`
- `top_k: 0`
- `top_p: 0.9`
- `sample_count: 10`
- calibration mode: `vol_rescale_mean_center`

Important: the LoRA attaches to the raw Kronos-small predictor model. The `vol_rescale_mean_center` step remains a post-processing calibration used during evaluation. We are not training the calibration as part of LoRA.

## What We Are Testing

We will train three separate LoRA adapters on the same frozen Kronos-small model:

- Slow LoRA: broad history adapter for stable structure.
- Medium LoRA: recent-month adapter for current regime.
- Fast LoRA: recent-week adapter for short-term shifts.

There is no regime-aware router in this run. We combine adapters with fixed static weights and compare the ensemble against the frozen base and naive baselines.

The fixed primary ensemble is:

- Slow: `0.25`
- Medium: `0.45`
- Fast: `0.30`

We also report:

- Slow only
- Medium only
- Fast only
- Equal-weight ensemble: `0.3333 / 0.3333 / 0.3333`
- Stale ensemble diagnostic: first trained adapters applied to later windows without updating

Weights must not be tuned on the forward validation window.

## Training Data And Windows

Use 100 liquid/common-stock assets for the first run. Prefer the same Common100 universe used in the aggressive base search so the frozen-base comparison is aligned with existing artifacts.

Use three rolling forward windows:

| Window | Slow Train | Medium Train | Fast Train | Validation |
| --- | --- | --- | --- | --- |
| W1 | `2021-05-28` to `2024-12-31` | trailing 63 trading days before validation | trailing 20 trading days before validation | `2025-01-02` to `2025-02-28` |
| W2 | `2021-05-28` to `2025-02-28` | trailing 63 trading days before validation | trailing 20 trading days before validation | `2025-03-03` to `2025-04-30` |
| W3 | `2021-05-28` to `2025-04-30` | trailing 63 trading days before validation | trailing 20 trading days before validation | `2025-05-01` to `2025-06-30` |

The first signal run intentionally stays early in 2025 because the dataset only starts in 2021 and the 512-day context consumes roughly two trading years. Later 2025 and 2026 windows are better reserved for a confirmation run after the first signal test passes.

## Training Examples

Each example is:

- Input: 512 daily K-line rows ending at context date `t`
- Target: next-day token pair for date `t + 1`
- Metadata: ticker, context end date, target date, split id, adapter scale

The tokenizer is frozen. Convert each K-line sequence into Kronos hierarchical token IDs using the existing Kronos tokenizer path.

Training objective:

- Teacher-forced next-token cross entropy for `s1` and `s2`
- Loss = `s1_cross_entropy + s2_cross_entropy`
- No direct return/ranking loss in the first signal run

This keeps the first experiment close to Kronos' native autoregressive training path and avoids adding a custom financial objective before we know LoRA adaptation has any signal.

## LoRA Implementation

Keep the Kronos tokenizer and base predictor weights frozen.

Add LoRA adapters to the predictor model only:

- Primary target: linear projections inside Kronos `transformer` blocks
- Secondary target if implementation is simple: linear layers in `dep_layer`
- Exclude: tokenizer, embeddings, temporal embedding, final token heads, and post-processing calibration

Default adapter config:

- rank `r = 8`
- alpha `16`
- dropout `0.05`
- train bias: `false`
- precision: fp16 or bf16 on Modal GPU
- optimizer: AdamW
- learning rate: `1e-4`
- epochs: `3`
- batch size: largest stable batch on L4, target `32-64`
- early stop: validation token loss patience `1`

If the exact Kronos module names differ during implementation, use a module discovery pass and select all `torch.nn.Linear` modules under the predictor model's transformer blocks, while still excluding tokenizer, embeddings, time embedding, and heads.

## Static Ensemble Inference

For each validation window, produce predictions for:

- frozen base
- Slow only
- Medium only
- Fast only
- fixed weighted ensemble
- equal weighted ensemble
- stale weighted ensemble
- naive baselines

Preferred ensemble method for the first run:

1. Run inference separately for each adapter.
2. Convert each prediction to predicted close and predicted return.
3. Combine predicted returns using the fixed weights.
4. Reconstruct ensemble predicted close from last context close and ensemble return.
5. Apply the same `vol_rescale_mean_center` calibration used for the frozen base.

This is cheaper and easier to debug than trying to merge adapter weights before inference. It also makes each adapter's contribution visible.

## Required Controls

Required low-cost controls:

- Frozen Kronos-small engineering base on the identical asset/date windows.
- Existing no-leakage naive baselines.
- Stale temporal ensemble: train the W1 Slow/Medium/Fast adapters once and apply them to W2 and W3 without updating.

Deferred control:

- Shuffled-label LoRA.

Do not run shuffled-label LoRA in the first pass unless the Temporal LoRA ensemble shows promising lift. If the first pass is promising, run one cheap shuffled sanity check on one window and 25-50 assets. The point of the shuffled control is to catch false positives from leakage, calibration artifacts, or adapter overfitting. It is not required before the first go/no-go read.

## Metrics

Primary metric:

- Cross-sectional Spearman RankIC of predicted return vs actual return by date, averaged over validation dates.

Secondary metrics:

- Pearson IC
- directional accuracy
- daily RankIC positive rate
- daily IC positive rate
- RankIC gap to best naive baseline
- predicted/actual absolute return multiple
- positive-rate gap
- invalid OHLC rate

Report metrics by:

- window
- adapter variant
- pooled all-window aggregate
- per-date daily metrics

## Signal Decision

Move forward to a research-grade effort only if the primary fixed weighted ensemble meets all required gates:

- Mean RankIC lift over frozen base is at least `+0.015`.
- Directional accuracy lift over frozen base is at least `+1 percentage point`.
- At least 2 of 3 rolling windows improve RankIC over frozen base.
- The ensemble beats the best naive baseline on RankIC in the pooled result.
- Predicted/actual absolute return multiple stays between `0.5` and `2.0`.
- Invalid OHLC rate stays below `3%`.

Interpretation:

- Strong pass: all gates pass, and stale ensemble is worse than rolling ensemble. Scale to 200 assets and 6+ windows.
- Weak pass: RankIC lift appears but directional or naive-baseline gap is weak. Run one shuffled-label sanity check before scaling.
- Fail: no consistent RankIC lift or lift appears only in one window. Do not scale; revisit data/objective/base model.

## Expected Outputs

Write outputs under:

`outputs/temporal_lora_signal_run/`

Expected files:

- `window_manifest.csv`
- `training_examples_summary.csv`
- `adapter_registry.json`
- `predictions_frozen_base.csv`
- `predictions_slow.csv`
- `predictions_medium.csv`
- `predictions_fast.csv`
- `predictions_weighted_ensemble.csv`
- `predictions_equal_ensemble.csv`
- `predictions_stale_ensemble.csv`
- `metrics_by_window.csv`
- `daily_cross_section_metrics.csv`
- `signal_decision.json`

`signal_decision.json` should contain:

- accepted: boolean
- decision: `strong_pass`, `weak_pass`, or `fail`
- reason
- primary RankIC lift
- directional lift
- windows improved count
- best naive baseline and gap
- recommended next step

## Cost Estimate

Modal currently lists L4 at `$0.000222/sec`, about `$0.80/hour`. It lists T4 at about `$0.59/hour`, A10 at about `$1.10/hour`, and H100 at about `$3.95/hour`. Modal Starter also includes `$30/month` free compute credits. Region selection can add `1.5x-1.75x`, and non-preemptible execution costs `3x`. Source: https://modal.com/pricing

Use L4 by default. The first run should avoid region selection and non-preemptible execution.

Low-cost first pass estimate:

- implementation/debug GPU time: `$5-$15`
- train Slow/Medium/Fast adapters across 3 windows: `$10-$25`
- frozen/adapter/ensemble inference and scoring: `$5-$15`
- expected first-pass budget: `$20-$55`
- hard cap before reassessing: `$75`

If the first pass shows promising lift, add the deferred shuffled-label sanity check:

- extra estimated cost: `$5-$15`

If the run approaches `$75` before producing a full W1-W3 result, stop and inspect logs before spending more.

## Why This Is Enough

This signal run does not prove the method is publishable. It only tests whether the core Temporal LoRA idea has measurable forward-window lift using the current data and engineering base.

That is the correct first milestone because the frozen base is weak. We should not spend on a large Temporal LoRA system until a small, controlled run shows that temporal adaptation can improve out-of-sample RankIC over the same frozen model and naive baselines.
