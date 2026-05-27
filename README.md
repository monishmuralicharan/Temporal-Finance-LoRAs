# Temporal Finance LoRAs

Kronos-focused scaffold for testing temporal LoRA adaptation on financial
OHLCV/K-line forecasting. The current repo surface is intentionally narrow:
run Kronos on Modal, pull prediction artifacts locally, and grade the output.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
arch -arm64 ./.venv/bin/python -m pip install -r requirements.txt
modal setup
```

## Run Kronos

Download/cache the Kronos model files in Modal:

```bash
arch -arm64 ./.venv/bin/python -m temporal_finance.modal_kronos_checkpoint4 download-models --config configs/kronos_checkpoint4_best.yaml
```

Run the current best smoke config and write outputs locally:

```bash
arch -arm64 ./.venv/bin/python -m temporal_finance.modal_kronos_checkpoint4 run --config configs/kronos_checkpoint4_best.yaml --gpu T4
```

Expected output files under the config's `output_dir`:

- `predictions.csv`
- `metrics.json`
- `per_ticker_metrics.csv`

Optional: run the ordered Kronos variant sweep:

```bash
arch -arm64 ./.venv/bin/python -m temporal_finance.modal_kronos_checkpoint4 run-experiments --config configs/kronos_checkpoint4_greedy_smoke.yaml --output-dir outputs/kronos_experiments --gpu T4
```

Run the paper-proximal public-model benchmark suite:

```bash
arch -arm64 ./.venv/bin/python -m temporal_finance.modal_kronos_checkpoint4 run-experiments --config configs/kronos_paper_proximal_smoke.yaml --output-dir outputs/kronos_paper_proximal/smoke --gpu T4 --suite paper-proximal
```

This writes `summary.csv`, `benchmark_summary.csv`, and per-run evaluation
metrics under `outputs/kronos_paper_proximal/smoke`. Scale to
`configs/kronos_paper_proximal_stride21.yaml` after the smoke run completes.

## Base Model Selection Sweep

Run the staged manifest-driven benchmark on Modal:

```bash
arch -arm64 ./.venv/bin/python -m temporal_finance.modal_kronos_checkpoint4 run-benchmark --manifest configs/kronos_base_selection_smoke.yaml --gpu T4
```

If the smoke run promotes candidates, scale only those names to the broader
stride benchmark:

```bash
arch -arm64 ./.venv/bin/python -m temporal_finance.modal_kronos_checkpoint4 run-benchmark --manifest configs/kronos_base_selection_broad_stride.yaml --gpu T4 --only-promoted-from outputs/kronos_base_selection/smoke/promoted_run_names.txt
```

The full daily manifest is reserved for promoted broad-stride candidates:

```bash
arch -arm64 ./.venv/bin/python -m temporal_finance.modal_kronos_checkpoint4 run-benchmark --manifest configs/kronos_base_selection_full_daily.yaml --gpu T4 --only-promoted-from outputs/kronos_base_selection/broad_stride/promoted_run_names.txt
```

Each stage writes `benchmark_summary.csv`, `benchmark_summary.json`, and
`promoted_run_names.txt` under `outputs/kronos_base_selection/<stage>`.

## Grade An Output

```bash
arch -arm64 ./.venv/bin/python -m temporal_finance.run_kronos_output_evaluation --predictions outputs/kronos_checkpoint4_best/predictions.csv --output-dir outputs/kronos_checkpoint4_best/evaluation
```

The grader computes whatever is available in the CSV, including return metrics,
close/OHLC paired errors, daily IC/RankIC, and OHLC validity checks.

## Continual Learning Metrics

Primary metrics should track whether temporal LoRA mixtures improve future
K-line forecasting without discarding useful longer-horizon structure:

- OHLC/close path normalized MAE and MAPE.
- Rolling future-window adaptation improvement versus frozen Kronos.
- Temporal-mixture gain versus recent-only LoRA and all-history LoRA.
- Retained performance across older regimes after adding recent adapters.

Secondary probes:

- Price/return IC and RankIC.
- Directional accuracy.
- Volatility or high-low range error when those columns are available.
- OHLC validity: predicted high/low consistency and non-negative volume.

## Local Checks

```bash
arch -arm64 ./.venv/bin/python -m pytest
git diff --check
```
