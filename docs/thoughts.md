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
### 2026-05-27
- Fetched and inspected `origin/dataset`; it contains Massive daily adjusted OHLCVA exports under `dataset/data/kronos/by_ticker` plus manifests/scripts, not a Kronos upstream branch.
- Added a `local_kronos_csv` data source and Modal dataset mount so Kronos can run directly on dataset-branch CSVs with `timestamps,open,high,low,close,volume,amount`.
- Dataset-branch smoke with 31 tickers, 120-day context, 5-day horizon, stride 21, and sample-10 decoding did not rescue `NeoQuasar/Kronos-base`: best base variant was `base_t06_p09_s10__vol_rescale` with IC/RankIC `-0.0464`/`-0.0729` and directional accuracy `0.458`.
- The same 31-ticker dataset smoke also did not produce a promoted `Kronos-small` candidate: `small_t06_p09_s10__vol_rescale` reached IC/RankIC `0.0056`/`-0.0045` with directional accuracy `0.542`. Treat the dataset branch as useful pipeline data, but too short and too recent to validate a paper-like base model.

### 2026-05-28

- Refetched `origin/dataset` at `cf50cdd` (`Add top-300 + SPY combined Kronos CSV (2021-2026)`). The branch now includes `dataset/data/kronos/combined/all_tickers_combined.csv` with 300 tickers and 359,137 rows, while the old `by_ticker` folder remains the 31-name 2025-2026 demo set.
- Split the combined branch CSV into `dataset/data/kronos/combined_by_ticker` for the existing Kronos per-ticker benchmark path, and taught the Modal config mapper to preserve dataset subdirectories such as `combined_by_ticker` instead of always using `by_ticker`.
- A top-200 Kronos-base sample-10, 512-context run on T4 was canceled by the local Modal client before completion, so it is not usable. A bounded top-100 sample-3 retry completed: 1,700 prediction rows, 100 tickers, 17 monthly-stride dates, 2025-01-02 through 2026-05-07, 5-day horizon.
- Best completed top-100 calibrated Kronos-base row was `vol_rescale_mean_center`: return IC `0.1415`, RankIC `0.1079`, predicted/actual abs-return multiple `1.54`, invalid OHLC rate `2.24%`; it still failed promotion because directional accuracy was `0.4888`. This supports the data/context-mismatch hypothesis for IC/RankIC, but not a final base-model promotion gate.

- Added a common-stock top-100 512-context Kronos-base sample-10 benchmark config using the `top_liquid_200.txt` intersection with the combined dataset, plus detached Modal submit/collect support so longer sample-10 runs are not lost if the local client disconnects.

- Submitted the common-stock top-100 sample-10 benchmark as a detached Modal L4 call (`fc-01KSREXEZ6YSFBT013AY3M4G26`); collection state is under `outputs/kronos_base_selection/dataset_common100_context512_s10/detached_state.json`.

- Collected the detached common-stock top-100 sample-10 L4 run. `base_t06_p09_s10_context512_common100__vol_rescale_mean_center` passed the promotion gate with return IC `0.1052`, RankIC `0.1126`, directional accuracy `0.5335`, daily IC/RankIC positive rates `0.5294`/`0.5294`, predicted/actual abs-return multiple `1.57`, positive-rate gap `0.045`, and invalid OHLC rate `0.12%` over 1,700 predictions.
### 2026-05-29

- Validate Kronos-base as the LoRA starting candidate under our protocol rather than requiring direct paper replication. The balanced validation path is common100 stride-5, common100 recent daily from 2025-10-01, and mixed200 stride-21 as a diagnostic-only universe stress check; acceptance requires calibrated Kronos-base to pass stricter stability gates and beat aligned no-leakage naive baselines.
- Submitted the common100 stride-5 Kronos-base validation pass as detached Modal call `fc-01KSRNNDCCEWADQVY1104DBTZQ`. Local collection was intentionally stopped while the detached run remained in flight; collect later from `outputs/kronos_base_selection/dataset_common100_context512_s10_stride5/detached_state.json`.
- Collected the common100 stride-5 validation pass. The calibrated `vol_rescale_mean_center` candidate failed promotion over 6,900 predictions: IC `0.0218`, RankIC `0.0325`, directional accuracy `0.5106`, daily IC/RankIC positive rates `0.4638`/`0.4928`, despite acceptable return magnitude and invalid OHLC rate. Stop before recent-daily and mixed200 validation; do not use this Kronos-base setup as the official LoRA base.
- Implemented the aggressive Kronos-only base search scaffold: horizon-safe prediction columns, generic return metrics, a 32-candidate common100 Stage 1 grid across Kronos-small/base, horizons 1/3/5/10, and four decoding presets, plus suite state generation/submit/collect/select commands under `temporal_finance.kronos_base_search`.
- Submitted all 32 Stage 1 aggressive Kronos base-search candidates to Modal as detached L4 calls. First collection passes materialized 21 raw candidates / 63 calibrated rows; current leader is `small_h3_t06_p09_s10__vol_rescale_mean_center` with IC `0.1978`, RankIC `0.1342`, directional accuracy `0.5435`, daily RankIC positive rate `0.7647`, and RankIC gap to best naive `0.1036`. Hold Stage 2 until the remaining 11 detached calls are collected.
- Final Stage 1 collection materialized 29/32 raw candidates. The last three long-running base sampling calls (`base_h5_t06_p09_s10`, `base_h10_t06_p09_s5`, `base_h10_t06_p09_s10`) were canceled after the leading h3 candidates were already clear; suite state records them as canceled.
- Aggressive Stage 2 stride-5 validation collected 17/17 promoted raw candidates. The Stage 1 h3 leader did not hold up (`small_h3_t06_p09_s10__vol_rescale_mean_center`: IC `0.0176`, RankIC `-0.0034`, directional `0.4914`), and only `small_h1_t06_p09_s5__vol_rescale_mean_center` plus `small_h1_t06_p09_s10__vol_rescale_mean_center` passed strict gates.
- Recent-daily validation from 2025-10-01 rejected both Stage 2 survivors: sample-10 IC/RankIC `0.0114`/`-0.0020`, directional `0.5020`; sample-5 IC/RankIC `0.0065`/`-0.0039`, directional `0.4971`. `selected_base_model.json` is intentionally `accepted: false`; do not use this aggressive search suite as the LoRA base.
- Define the first Temporal LoRA effort as a low-cost signal run, not a full research run: fixed Slow/Medium/Fast adapters on frozen Kronos-small, static weights `0.25/0.45/0.30`, 100 assets, 3 forward windows, required frozen/naive/stale controls, deferred shuffled-label control only if early lift appears, and a `$75` spend cap.
- Implement the signal-run scaffold as an additive Modal experiment: custom LoRA wrappers target Kronos `transformer`/`dep_layer` linears, training uses context-normalized tokenizer next-token cross entropy, ensembles combine raw adapter predictions before one shared calibration pass, and Modal collection returns LoRA adapter weights as base64 artifacts.
- Fix the first Temporal LoRA sampling issue before reruns: capped training sets are sampled evenly across the target-date-sorted train window rather than taking the earliest rows, so Slow adapters include each rolling window end instead of reusing only early-history examples.
