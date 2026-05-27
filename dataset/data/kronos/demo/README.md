# Kronos demo dataset (5 stocks + S&P 500)

**Stocks:** AAPL, MSFT, NVDA, GOOGL, AMZN  
**S&P 500 proxy:** `SPY_S&P500.csv` (SPY ETF)

**Window:** 2025-01-02 → 2026-04-30 (daily bars)

## Files

| File | Contents |
|------|----------|
| `AAPL.csv` … `AMZN.csv` | One symbol per file |
| `SPY_S&P500.csv` | S&P 500 (SPY) |
| `all_tickers_combined.csv` | All six in one sheet (`ticker` column first) |
| `manifest.csv` | Row counts and date ranges |

## CSV columns (matches Kronos `predict` inputs)

| Column | Role in Kronos |
|--------|----------------|
| `x_timestamp` | → `x_timestamp` Series (historical bar dates) |
| `open`, `high`, `low`, `close` | → `df` OHLC (required) |
| `volume` | → `df` (optional) |
| `amount` | → `df` (optional; `close × volume`, or 0 if no volume) |

## Load like Kronos

```python
import pandas as pd

df = pd.read_csv("data/kronos/demo/AAPL.csv", parse_dates=["x_timestamp"])
x_df = df[["open", "high", "low", "close", "volume", "amount"]]
x_timestamp = df["x_timestamp"]

# Future dates for prediction (example: next 5 trading days after last bar)
y_timestamp = pd.bdate_range(x_timestamp.iloc[-1] + pd.Timedelta(days=1), periods=5)
```

## Regenerate

```bash
python scripts/export_kronos_demo.py --fetch
```
