Checkpoint 1: Problem Definition
Goal
Lock the exact research question and experiment.
Outputs
A short research plan document with the problem, research question, core method, and prediction task.
Problem: financial models need continual adaptation, but naive fine-tuning causes forgetting.
Research question: Can time-scale-specialized LoRA experts reduce forgetting while improving future financial forecasting?
Core method: Slow LoRA + Medium LoRA + Fast LoRA + regime-aware router.
Prediction task: Input past 120 trading days → predict future 5-day return.
Success Criteria
You can explain the whole paper in one paragraph.

Checkpoint 2: Dataset Spec (april 7, 2025 onwards)
Goal
Decide what data you are using and what each training example looks like.
Outputs
Assets: top 100–500 liquid U.S. stocks.
Time range: 2015–2024 daily data.
Raw features: open, high, low, close, adjusted close, volume.
Derived features: daily return, log return, 5-day return, 20-day volatility, 60-day volatility, volume z-score, moving average distance, drawdown, market index return.
Target: future 5-day return.
Success Criteria
You know exactly what goes into the model and what it predicts.

Checkpoint 3: Data Pipeline
Goal
Turn raw stock data into training samples.
Outputs
Working code that produces X: [num_samples, 120, num_features].
router_features: [num_samples, num_router_features].
y: [num_samples, 1].
metadata: asset_id, date, split_id.
Train/validation/test splits, rolling-window split definitions, data normalization method, and sample count table.
Success Criteria
You can load a batch and verify past 120 days → future 5-day return without data leakage.

Checkpoint 4: Baseline Forecasting Model
Goal
Get a simple model working before adding LoRA.
Outputs
Basic time-series forecasting model: Input [batch, 120, features] → output future 5-day return.
Recommended first model: small Transformer encoder or simple decoder-only time-series Transformer.
First baseline results for linear baseline and small Transformer.
Success Criteria
The pipeline trains end-to-end and produces non-random predictions.

Checkpoint 5: LoRA Integration
Goal
Add LoRA adaptation to the base model.
Outputs
Working implementation of frozen base model + LoRA adapters.
Baselines: frozen base model, full fine-tuning, single LoRA.
Result table with MSE, directional accuracy, RankIC, and trainable parameters.
Success Criteria
Single LoRA trains correctly and uses far fewer trainable parameters than full fine-tuning.

Checkpoint 6: Continual Learning Setup
Goal
Create the rolling adaptation benchmark.
Outputs
Continual learning loop: initial train 2015–2019, adapt 2020 Q1, test 2020 Q2, then roll forward.
At every step, save model checkpoint, future test performance, old-window performance, and forgetting score.
Result table by step with adapt window, test window, future RankIC, and forgetting.
Success Criteria
You can measure whether adapting to new data hurts older-window performance.

Checkpoint 7: Replay and Distillation Baselines
Goal
Add stronger forgetting baselines.
Outputs
Replay LoRA: train on new data plus a small buffer of old data.
Distillation LoRA: train on new data while preserving old model outputs on old samples.
Comparison table with future RankIC, direction accuracy, forgetting, and training cost.
Success Criteria
You have credible baselines that reviewers would expect.

Checkpoint 8: Temporal LoRA Experts
Goal
Build the core method without the router first.
Outputs
Three separate LoRA experts: Slow, Medium, and Fast.
Slow LoRA: trained on broad historical data; captures stable long-term structure.
Medium LoRA: trained on recent months; captures current market regime.
Fast LoRA: trained on recent days/weeks; captures short-term changes.
Static combination baseline: average adapters or average predictions.
Success Criteria
The time-scale LoRAs train separately and can be combined or ensembled.

Checkpoint 9: Regime-Aware Router
Goal
Add the actual routing mechanism.
Outputs
Router input: recent volatility, trend strength, drawdown, volume spike, market return, horizon.
Router output: softmax weights over Slow, Medium, and Fast LoRA.
Final method: router-weighted combination of Slow, Medium, and Fast adapters.
Main table comparing frozen base, full fine-tune, single LoRA, replay LoRA, static LoRA ensemble, and Temporal Mixture-of-LoRAs.
Success Criteria
Your method beats single LoRA on forgetting and is competitive or better on future forecasting.

Checkpoint 10: Ablation Study
Goal
Prove which pieces of the method matter.
Outputs
Ablation table: full method, no router/equal weights, Slow+Medium only, Slow+Fast only, Medium+Fast only, router without regime features, no replay loss, no distillation loss.
Success Criteria
You can say which component improves performance and which component reduces forgetting.

Checkpoint 11: Analysis Figures
Goal
Create the visuals that make the paper convincing.
Outputs
Figure 1: method diagram.
Figure 2: rolling future performance over time.
Figure 3: forgetting score over adaptation steps.
Figure 4: router weights across market regimes.
Optional Figure 5: performance split by volatility regime.
Success Criteria
A reader can understand the method and main result from the figures alone.

Checkpoint 12: Paper Draft
Goal
Write the first full paper.
Outputs
Complete draft with title, abstract, introduction, related work, method, experiments, results, analysis, limitations, conclusion, and references.
Abstract: problem → method → benchmark → main finding.
Introduction: markets shift; continual adaptation is needed; fine-tuning forgets; single LoRA is cheap but limited; propose time-scale LoRA experts with routing.
Method: base model, Slow/Medium/Fast LoRAs, router, training objective, continual-learning benchmark.
Experiments: dataset, rolling evaluation, baselines, metrics, implementation details.
Results: main table, forgetting table, ablations, router analysis.
Limitations: daily OHLCV only, no order book data, no text/news data, limited trading simulation.
Success Criteria
The draft is complete enough that someone else can read it and understand the contribution.

Checkpoint 13: Paper Strengthening
Goal
Make the paper more credible.
Outputs
Improve or add more baselines, more rolling windows, more assets, stronger forgetting metric, transaction-cost-adjusted trading simulation, regime-specific analysis, and clear limitations.
Potential extra experiment: Event LoRA for earnings/gap/high-volatility periods.
Only add Event LoRA if the 3-LoRA version already works.
Success Criteria
The paper has enough evidence to support the main claim.

Checkpoint 14: Submission Package
Goal
Prepare the final version for preprint/workshop submission.
Outputs
Final PDF, clean GitHub repo, README, dataset instructions, experiment scripts, saved config files, main results CSV, figures, and appendix.
Suggested repo structure: /data, /models, /lora, /router, /experiments, /evaluation, /figures, /configs.
Success Criteria
Someone can reproduce the main experiment from your repo.

