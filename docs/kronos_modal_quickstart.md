# Kronos Modal Quickstart

This runs the checkpoint 4 Kronos baseline on Modal. The upstream Kronos repo is
installed in the Modal image, and the Hugging Face tokenizer/model files are
cached in a Modal Volume instead of on your laptop.

## What It Uses

- Upstream repo: `shiyu-coder/Kronos`
- Pinned commit: `67b630e67f6a18c9e9be918d9b4337c960db1e9a`
- Default model: `NeoQuasar/Kronos-small` in `configs/kronos_checkpoint4_best.yaml`
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-base`
- Cache Volume: `temporal-finance-kronos-cache`

Kronos expects K-line inputs. The repo wrapper converts Yahoo Finance OHLCV into
adjusted K-lines by scaling `Open`, `High`, and `Low` with
`Adj Close / Close`, then uses adjusted close as the `close` channel.

## 1. Authenticate With Modal

```bash
modal setup
```

## 2. Download Model Files Into Modal

```bash
arch -arm64 ./.venv/bin/python -m temporal_finance.modal_kronos_checkpoint4 download-models --config configs/kronos_checkpoint4_best.yaml
```

This warms the Modal cache. The `run` command will also download missing files
automatically, but explicit download makes the first real run easier to debug.

## 3. Run Remotely

For the current recommended smoke benchmark, use the best config:

```bash
arch -arm64 ./.venv/bin/python -m temporal_finance.modal_kronos_checkpoint4 run --config configs/kronos_checkpoint4_best.yaml --gpu T4
```

The smoke config uses the full ticker/date range but evaluates every 21st
rolling window. Set `eval_stride: 1` for a full daily rolling run, which is
much slower.

Optional GPU override examples: `--gpu T4` or `--gpu L4`.

## 4. Check Outputs

The remote run writes these local files under the config's `output_dir`:

- `predictions.csv`
- `metrics.json`
- `per_ticker_metrics.csv`

Use the generic grader for quick continual-learning diagnostics:

```bash
arch -arm64 ./.venv/bin/python -m temporal_finance.run_kronos_output_evaluation --predictions outputs/kronos_checkpoint4_best/predictions.csv --output-dir outputs/kronos_checkpoint4_best/evaluation
```

## 5. Run Ordered Fix Experiments

To test the Kronos fixes in order on Modal T4 and keep a deterministic summary:

```bash
arch -arm64 ./.venv/bin/python -m temporal_finance.modal_kronos_checkpoint4 run-experiments --config configs/kronos_checkpoint4_greedy_smoke.yaml --output-dir outputs/kronos_experiments --gpu T4
```

This runs adjusted greedy, raw-OHLCV diagnostic, Kronos-small greedy, and a
sampling ensemble only if a calibrated winner exists first. It writes:

- `outputs/kronos_experiments/summary.csv`
- `outputs/kronos_experiments/summary.json`
- `outputs/kronos_experiments/best_config.yaml` when a selectable variant passes
  the guardrails

The current best smoke config is also checked in as
`configs/kronos_checkpoint4_best.yaml`.

## 6. Run Paper-Proximal Public-Model Benchmarks

Use the 10-symbol paper-proximal smoke before scaling to the broader stride-21 config:

```bash
arch -arm64 ./.venv/bin/python -m temporal_finance.modal_kronos_checkpoint4 run-experiments --config configs/kronos_paper_proximal_smoke.yaml --output-dir outputs/kronos_paper_proximal/smoke --gpu T4 --suite paper-proximal
```

The suite compares public Kronos-base and Kronos-small with adjusted OHLCV,
`temperature: 0.6`, `top_p: 0.9`, `top_k: 0`, and `sample_count: 10`. It
writes `benchmark_summary.csv` with gaps to the paper's Kronos-small return
IC/RankIC reference (`0.0665` / `0.0622`).

## 7. Run Staged Base Selection

The base-selection runner expands manifest experiments, runs raw Kronos forecasts
once per experiment, materializes calibration variants locally, and writes a
promotion summary. Start with the 10-symbol smoke manifest:

```bash
arch -arm64 ./.venv/bin/python -m temporal_finance.modal_kronos_checkpoint4 run-benchmark --manifest configs/kronos_base_selection_smoke.yaml --gpu T4
```

Scale only promoted candidates to the broader stride benchmark:

```bash
arch -arm64 ./.venv/bin/python -m temporal_finance.modal_kronos_checkpoint4 run-benchmark --manifest configs/kronos_base_selection_broad_stride.yaml --gpu T4 --only-promoted-from outputs/kronos_base_selection/smoke/promoted_run_names.txt
```

Use `configs/kronos_base_selection_full_daily.yaml` only after a broad-stride
candidate remains calibrated and beats the current Kronos baseline.
