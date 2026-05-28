#!/usr/bin/env bash
# Top 300 liquid stocks + SPY, daily OHLCV 2014 → today → all_tickers_combined.csv
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "==> Build universe (top 300 + SPY by dollar volume)"
python scripts/build_universe_top300.py

echo "==> Fetch + export combined CSV (resumes skipped parquets)"
python scripts/export_kronos_combined.py --fetch --start 2014-01-02

echo "Done: data/kronos/combined/all_tickers_combined.csv"
