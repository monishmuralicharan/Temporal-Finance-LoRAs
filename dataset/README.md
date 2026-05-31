# Continual Learning on Kronos with Multi-Timescale LoRA

Research codebase for adapting [Kronos](https://arxiv.org/abs/2508.02739) with slow / medium / fast LoRA experts and a regime-aware router, using U.S. equity daily data from [Massive.com](https://massive.com).

## Checkpoints covered

| Checkpoint | Output |
|------------|--------|
| 1 | [`docs/research_plan.md`](docs/research_plan.md) |
| 2 | [`docs/dataset_spec.md`](docs/dataset_spec.md) |
| 3 | Data pipeline → `data/processed/{X,router_features,y,metadata}.*` |

## Quick start

### 1. Credentials (do not commit)

```bash
cp .env.example .env
# Paste from Massive dashboard → API + Flat Files (S3)
```

| Variable | Dashboard tab |
|----------|----------------|
| `MASSIVE_API_KEY` | Accessing the API |
| `MASSIVE_S3_ACCESS_KEY_ID` | Accessing Flat Files (S3) |
| `MASSIVE_S3_SECRET_ACCESS_KEY` | Flat Files secret |
| `MASSIVE_S3_ENDPOINT` | `https://files.massive.com` |
| `MASSIVE_S3_BUCKET` | `flatfiles` |

**Security:** Keys in screenshots/chat should be rotated if this repo is shared.

### 2. Environment

```bash
cd /Users/parth/Cursor_Projects/Continual_Learning_Research
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download bars (Jan 2025 → Apr 2026 + warmup)

Uses REST [`Custom Bars`](https://massive.com/docs/rest/stocks/aggregates/custom-bars):

```bash
python scripts/01_fetch_bars.py
# Or a quick test:
python scripts/01_fetch_bars.py --tickers AAPL MSFT SPY --start 2024-06-01 --end 2026-04-30
```

Optional bulk path (faster, no per-ticker rate limits) if your plan includes flat files:

```bash
python scripts/01_fetch_bars_s3.py --start 2025-01-02 --end 2026-04-30
```

### 4. Build tensors

```bash
python scripts/02_build_dataset.py
python scripts/03_verify_batch.py --index 0
```

### 5. Load in Python

```python
from pathlib import Path
from src.data.samples import load_dataset

bundle = load_dataset(Path("data/processed"))
# bundle.X.shape          -> (N, 120, F)
# bundle.router_features  -> (N, R)
# bundle.y                -> (N, 1)
```

## Configuration

Edit [`config/data_config.yaml`](config/data_config.yaml):

- `fetch.start_date` / `fetch.end_date` — download window (default 2025-01-02 → 2026-04-30)
- `splits.first_label_date` — first prediction label (default **2025-04-07**)
- `universe` — ticker list in `data/universe/top_liquid_200.txt`

## Massive API reference

- REST quickstart: https://massive.com/docs/rest/quickstart  
- Stock daily bars: `GET /v2/aggs/ticker/{TICKER}/range/1/day/{from}/{to}`  
- Flat file day aggregates: https://massive.com/docs/flat-files/stocks/day-aggregates  

## Project layout

```
config/data_config.yaml
docs/research_plan.md
docs/dataset_spec.md
data/universe/top_liquid_200.txt
scripts/01_fetch_bars.py
scripts/02_build_dataset.py
scripts/03_verify_batch.py
src/data/massive_rest.py
src/data/massive_s3.py
src/data/features.py
src/data/samples.py
```
