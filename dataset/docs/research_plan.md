# Research Plan: Time-Scale LoRA Experts on Kronos

## Problem

Financial foundation models like [Kronos](https://arxiv.org/abs/2508.02739) must adapt to new market regimes, but naive fine-tuning on recent data causes **catastrophic forgetting** of patterns learned during pre-training.

## Research Question

**Can time-scale-specialized LoRA experts (slow / medium / fast) with a regime-aware router reduce forgetting while improving future return forecasting?**

## Core Method

1. **Base model**: Kronos (pre-trained K-line foundation model).
2. **Adapters**: Three LoRA experts tuned for different temporal horizons:
   - **Slow LoRA** — long-horizon structure (months / regimes)
   - **Medium LoRA** — swing dynamics (weeks)
   - **Fast LoRA** — short-horizon noise and microstructure (days)
3. **Router**: Regime-aware gating over experts using `router_features` (volatility, drawdown, trend, market state).
4. **Continual learning**: Sequential monthly (or quarterly) updates with replay / regularization to measure forgetting.

## Prediction Task

| Component | Specification |
|-----------|---------------|
| Input | Past **120 trading days** of per-asset features |
| Output | **Future 5-day log return** (close[t+5] / close[t] − 1, or log equivalent) |
| Router input | Low-dimensional regime vector per sample |

## Success Criteria (one paragraph)

We pre-train or load Kronos, attach slow/medium/fast LoRA experts and a regime router, and train continually on rolling calendar splits from U.S. liquid equities (Jan 2025–Apr 2026 for the adaptation phase; 2015–2024 for historical pretrain/frozen evaluation if needed). Each sample uses 120 days of OHLCV-derived features to predict the next 5-day return without leakage. We compare against (a) frozen Kronos, (b) single LoRA fine-tune, and (c) full fine-tune, measuring **forward forecast error** (MAE / RankIC) and **backward retention** on held-out past regimes. Success means multi-expert routing beats single-adapter fine-tuning on future months while retaining more performance on older validation windows—demonstrating reduced catastrophic forgetting.

## References

- Kronos paper: https://arxiv.org/abs/2508.02739
- Market data: [Massive.com REST](https://massive.com/docs/rest/quickstart) and [flat files](https://massive.com/docs/flat-files/stocks/day-aggregates)
