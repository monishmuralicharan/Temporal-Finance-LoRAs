# Thoughts

## Decision Log

### 2026-05-19
- Base model for checkpoint 4 is FinCast v1 from the public upstream repo and checkpoint release.
- Checkpoint 4 scope is inference plus evaluation only; LoRA work starts after the baseline runs cleanly.
- Bootstrap on one ticker first, but keep the pipeline shaped so multi-ticker evaluation can be added later.
- Start with daily univariate `Adj Close` forecasting rather than OHLCV covariates; expand inputs only after FinCast integration is stable.
- Use `yfinance` for the first local data path and keep the loader interface simple enough to swap to CSV or paper datasets later.
- Run inference locally by default; reserve Modal for heavier future fine-tuning or continual-adaptation jobs.
- Depend on a pinned external `external/FinCast-fts` checkout and a local FinCast weights path instead of copying upstream code into this repo.
- The current wrapper was implemented against upstream FinCast commit `a96c73a38e5106f3b36e9844cc44425043167dd8`.
- Evaluate the checkpoint 4 baseline on 5-day return metrics by converting the day-5 close forecast into a return from the last context close.
- Remote checkpoint 4 runs now use Modal with a fixed FinCast checkpoint Volume, uploaded once, and return final artifacts to the local machine instead of persisting outputs remotely.

### 2026-05-20
- Modal integration targets Modal 1.x APIs: local project code is included via `Image.add_local_python_source`, and container idle behavior uses `scaledown_window`.
- Prefer direct Hugging Face-to-Modal checkpoint download for `v1.pth` so the 4 GB file does not need to transit or persist on the local machine.
- Use `128` context steps for the FinCast checkpoint 4 baseline because FinCast's decoder patches inputs in length-`32` chunks; the original `120`-day research target is not shape-compatible with this checkpoint without padding.
- Expand checkpoint 4 from one ticker to a starter multi-ticker univariate basket, reporting aggregate and per-ticker metrics while skipping and recording individual ticker failures.
- Add a separate leakage-safe multivariate supervised baseline with OHLCV-derived past-only features, Ridge regression, and a zero-return comparator before attempting FinCast covariate experiments.

### 2026-05-21
- Add TimesFM 2.5 as the next base-model baseline using the same multi-ticker `Adj Close` windows and 5-day return scoring as FinCast, keeping covariate/XReg experiments separate to avoid mixing model comparison with future-covariate leakage questions.
- Remote TimesFM runs use a Modal image built from upstream `google-research/timesfm` commit `d720daa6786539c2566a44464fbda1019c0a82c0` and cache Hugging Face model files in Modal Volume `temporal-finance-timesfm-cache` to avoid local checkpoint/runtime storage.
- Add a TimesFM XReg covariate experiment that uses lagged OHLCV/market features and static forecast-origin features; do not pass actual future OHLCV as dynamic horizon covariates.
- Initial TimesFM XReg-first config over-amplified return forecasts; prefer residual XReg (`timesfm + xreg`) with stronger ridge and no static origin covariates if XReg is used, but treat it as exploratory because Ridge still wins on return metrics.
- Add Kronos as the finance-specific foundation baseline, using upstream `shiyu-coder/Kronos` commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a`, `NeoQuasar/Kronos-base`, and `NeoQuasar/Kronos-Tokenizer-base`; cache model files in Modal Volume `temporal-finance-kronos-cache`.
- Full rolling Kronos evaluation is materially slower than TimesFM because Kronos uses autoregressive K-line token sampling; add `eval_stride` and smoke configs for first-pass metrics before spending Modal time on the full daily-window run.

### 2026-05-23
- Compare Kronos fixes with a fixed ordered experiment runner: adjusted greedy first, raw OHLCV as diagnostic-only, Kronos-small greedy, then sampling only after a calibrated winner exists.
- Select Kronos changes by directional accuracy with guardrails on return magnitude and positive-rate calibration so over-amplified zero-shot forecasts do not get promoted.
- Ordered Kronos smoke results picked `NeoQuasar/Kronos-small` with adjusted OHLCV and greedy decoding: directional accuracy `0.525`, MAE `0.0528`, predicted/actual abs-return multiple `0.995`; raw OHLCV and sampled Kronos-base stayed over-amplified and were rejected.
- Add paper-style cross-sectional evaluation for Kronos versus Ridge using matched dates, daily IC/RankIC, and top-k long returns. On the current 60-date smoke comparison, Ridge still wins with mean daily RankIC `0.0790` versus Kronos `0.0048` and top-3 mean return `1.29%` versus `0.87%`.
- Narrow the repo's public runnable surface to Kronos only. Keep the generic output grader focused on continual-learning diagnostics: path/close errors, IC/RankIC probes, and OHLC validity rather than FinCast/TimesFM/Ridge benchmark commands.

### 2026-05-25
- Add a separate paper-proximal Kronos benchmark suite instead of changing the existing greedy smoke suite: use public Kronos-base/small models, adjusted OHLCV, `temperature: 0.6`, `top_p: 0.9`, `top_k: 0`, and `sample_count: 10`, then compare return IC/RankIC against the paper's `0.0665`/`0.0622` Kronos-small reference.
- First 10-symbol paper-proximal Kronos-base smoke was over-amplified and worse than the greedy small baseline: return IC/RankIC `-0.0821`/`-0.0722`, directional accuracy `0.373`, MAE `0.1554`, predicted/actual abs-return multiple `3.84`; do not promote this sampled base config without a calibration fix.
- Implement staged base-model selection as the default path before LoRA: smoke first, promote by balanced alpha gates, then broader stride and full daily manifests only for survivors.

### 2026-05-26
- Base-selection smoke rejected public `NeoQuasar/Kronos-base` variants again; the credible public-model candidates are `NeoQuasar/Kronos-small` sampling variants with adjusted OHLCV and postprocessing calibration.
- Broad-stride validation over the 42-symbol basket selected `small_t06_p09_s10__vol_rescale_mean_center` as the best available ranking base candidate: pooled return IC/RankIC `0.1027`/`0.0844`, directional accuracy `0.535`, predicted/actual abs-return multiple `1.16`, invalid OHLC rate `0.002`, beating the same-basket greedy small baseline `0.0298`/`0.0313`.
- If optimizing pooled IC instead of ranking, `small_t06_p09_s10__vol_rescale` is the score leader at IC/RankIC `0.1297`/`0.0776`, but its directional accuracy and daily RankIC positive rate are weaker than the selected mean-centered candidate.
- Do not call the selected candidate a clean pass yet: it failed the strict daily consistency gate with daily IC positive rate `0.5000` and daily RankIC positive rate `0.5167`, with 2022 driving the main regime weakness. Treat it as the current base candidate for iteration, not a final LoRA starting point.
- A targeted regular `NeoQuasar/Kronos-base` sample-10 smoke using adjusted OHLCV, `temperature: 0.6`, `top_p: 0.9`, and the same calibration family was rejected: best variants still had negative return IC/RankIC, with `base_t06_p09_s10__vol_rescale_mean_center` at `-0.0674`/`-0.0867` and directional accuracy `0.435`.
