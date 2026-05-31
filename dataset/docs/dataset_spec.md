# Dataset Specification

## Phase overview

| Phase | Calendar | Purpose |
|-------|----------|---------|
| Historical (Kronos-style pretrain / baselines) | 2015–2024 daily | Optional; matches original checkpoint spec |
| **Continual-learning phase** | **2025-01-02 → 2026-04-30** | LoRA adaptation data (this repo’s default) |
| First labeled sample | **≥ 2025-04-07** | Matches checkpoint 2; requires feature warmup |

## Universe

- **Assets**: Top **100–500** liquid U.S. common stocks (default **200** in `config/data_config.yaml`).
- **Selection**: Average dollar volume over a recent window; stored in `data/universe/top_liquid_200.txt`.
- **Market proxy**: `SPY` for `market_return`.

## Raw fields (per ticker, per trading day)

| Field | Source |
|-------|--------|
| `open`, `high`, `low`, `close` | Massive daily aggregates (`adjusted=true`) |
| `volume` | Same |
| `close` is split-adjusted | API default |

## Derived features (computed causally)

| Feature | Definition |
|---------|------------|
| `daily_return` | `close[t] / close[t-1] - 1` |
| `log_return` | `log(close[t] / close[t-1])` |
| `return_5d` | `close[t] / close[t-5] - 1` (past only; **not** the target) |
| `vol_20d` | Std of `log_return` over past 20 days |
| `vol_60d` | Std of `log_return` over past 60 days |
| `volume_zscore` | `(volume - mean_60) / std_60` |
| `ma_distance_20` | `close / SMA_20(close) - 1` |
| `drawdown` | `close / rolling_max_60(close) - 1` |
| `market_return` | `SPY` daily log return aligned by date |

## Target

- **`y`**: Forward **5-trading-day** log return from label date `t`:
  - `y = log(close[t+5] / close[t])`
- Label at `t` uses only data **≤ t** in `X`; prices at `t+1…t+5` appear only in `y`.

## Model tensors

| Tensor | Shape | Description |
|--------|-------|-------------|
| `X` | `[N, 120, F]` | Lookback window; `F` = raw + derived feature count |
| `router_features` | `[N, R]` | Regime vector (vol, drawdown, trend, market vol, …) |
| `y` | `[N, 1]` | Forward 5-day log return |
| `metadata` | struct | `asset_id`, `date` (label date), `split_id` |

## Splits

- **Scheme**: Rolling monthly blocks (`train` 2 mo → `val` 1 mo → `test` 1 mo), advancing forward.
- **Normalization**: Fit z-score on **train split only** per feature; apply to val/test.
- **Leakage checks**: No future rows in `X`; target uses strictly future closes; scaler fit only on train.

## Warmup requirement

Before the first label date you need enough history for:

- 120-day lookback
- 60-day rolling stats
- 5-day forward target

→ Fetch at least **~200 trading days** before `first_label_date` (configured in `history_buffer_trading_days`).
