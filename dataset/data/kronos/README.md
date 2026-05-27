# Kronos-ready daily data (U.S. equities)

Per-ticker CSVs for [Kronos `KronosPredictor`](https://github.com/shiyu-coder/Kronos).

## Layout

```
by_ticker/
  AAPL.csv
  MSFT.csv
  ...
manifest.csv          # row counts and date ranges per ticker
```

## Columns

| Column | Description |
|--------|-------------|
| `timestamps` | Trading date (daily bar) |
| `open`, `high`, `low`, `close` | Split-adjusted OHLC |
| `volume` | Share volume |
| `amount` | `close × volume` (dollar turnover proxy) |

## Default window

**2025-01-02 → 2026-04-30** (configured in `config/data_config.yaml` → `export`)

## Usage with Kronos

```python
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/kronos/by_ticker/AAPL.csv", parse_dates=["timestamps"])
x_df = df[["open", "high", "low", "close", "volume", "amount"]]
x_timestamp = df["timestamps"]
```

## Regenerate

```bash
python scripts/01_fetch_bars.py --exact-range
python scripts/export_kronos_csv.py
```
