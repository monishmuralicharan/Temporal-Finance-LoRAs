#!/usr/bin/env bash
# Fetch Jan 2025 – Apr 2026 daily bars and export Kronos-ready CSVs.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
# Prefer S3 bulk (no REST rate limits); fall back to REST for gaps
python scripts/ingest_s3_to_bars.py || python scripts/01_fetch_bars.py --exact-range
python scripts/01_fetch_bars.py --exact-range
python scripts/export_kronos_csv.py
