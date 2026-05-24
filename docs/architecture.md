# Architecture

This repository packages the final research direction developed during the competition work. The project did **not** start from the current 2-level stack. It evolved through several model families, feature systems, and validation protocols. This document records the final architecture and the reasoning behind it.

## 1. Problem framing

Competition setup:
- multiple related time series identified by `code`, `sub_code`, `sub_category`
- four forecast horizons: `1`, `3`, `10`, `25`
- numeric target `y_target`
- weighted evaluation metric:
  - `weighted_rmse_score(y_true, y_pred, weight)`

Important structural properties discovered during research:
- strong non-stationarity across `ts_index`
- different behavior by horizon, so per-horizon modeling was consistently better than one shared head
- exact train/test series overlap is limited, but `code` and `sub_category` overlap is strong
- many rows are near zero, but a small heavy-tail subset dominates the metric
- most raw features are shared across horizons for the same `(code, sub_code, sub_category, ts_index)` key

These findings ruled out naive local forecasting pipelines and pushed the work toward:
- per-horizon global models
- time-aware validation
- context features from hierarchy and cross-section
- conservative stacking and calibration rather than large end-to-end deep sequence models

## 2. Final exported architecture

The export contains two paths:

### 2.1 Primary path: 2-level stacked architecture

Entry point:
- `scripts/run_two_level_submission.py`

Core module:
- `src/models/two_level_stack.py`

Structure:

#### Level 1: pattern learners
For each horizon independently:
- LightGBM ensemble
- XGBoost ensemble

Both models are trained on engineered tabular time-series features.

Main level-1 feature blocks:
- target encodings from past-only train data:
  - `sub_category_enc`
  - `sub_code_enc`
- interaction features:
  - `feature_al - feature_am`
  - `feature_al / abs(feature_am)`
  - `feature_cg - feature_by`
- lag features on top raw signals
- rolling means and rolling standard deviations
- exponential weighted moving averages
- cross-sectional z-scores by `ts_index`
- cyclic time features:
  - `sin(2π ts_index / 100)`
  - `cos(2π ts_index / 100)`

Temporal segmentation:
- base train window
- meta train window
- final evaluation window

This segmentation supports leakage-safe calibration:
- level 1 learns on the base segment
- level 2 learns from level-1 outputs on the meta segment
- final model selection is measured on the held-out eval segment

#### Level 2: calibrated meta-predictor
Per horizon:
- ridge regression calibrator / blender

Inputs:
- `pred_lgb`
- `pred_xgb`
- mean of both predictions
- prediction difference and absolute difference
- prediction min / max
- `ts_index`
- encoded context features propagated from level 1

Selection logic:
- compare four candidates on the eval segment:
  - LightGBM
  - XGBoost
  - simple mean
  - ridge meta-model
- keep the best-scoring candidate per horizon

Final inference:
- refit level-1 models on full train data
- apply the selected per-horizon prediction rule
- merge horizon outputs into `submission.csv`

### 2.2 Secondary path: stable baseline LightGBM research pipeline

Entry points:
- `scripts/run_baselines.py`
- `scripts/run_cv.py`
- `scripts/run_training.py`
- `scripts/make_submission.py`

This path is the cleaner, lower-risk baseline package. It is kept because:
- it is easier to reproduce
- it supports walk-forward validation and ablation cleanly
- LightGBM was the most consistently reliable family across research stages

## 3. Validation architecture

The project used several validation schemes over time. Final lessons:

### Rejected
- random split inside train
- per-series random split
- overly small smoke CV as the only decision signal

These were too optimistic or too unstable relative to the leaderboard setting.

### Retained
- blocked tail holdout
- walk-forward / rolling time validation
- explicit base/meta/eval segmentation for stacking

The exported 2-level stack therefore uses strict temporal segmentation by design.

## 4. Why this architecture exists

The final architecture is a compromise forced by the experiments:

- Deep tabular DNNs largely collapsed to near-zero predictions and produced `0.0` weighted scores on many horizons.
- Pure LightGBM baselines were the strongest stable single-family models, especially on medium and long horizons.
- XGBoost sometimes added complementary signal, but was weaker and less stable than LightGBM alone.
- Recent-prior and cross-horizon research suggested that light stacking and calibration could help, but full panel stacks were only incremental rather than transformational.

That is why the final exported “important” architecture is:
- still tree-based
- still per horizon
- but with a second calibration layer that decides whether to trust LightGBM, XGBoost, their mean, or a learned ridge blend

## 5. Package layout

Shared package layout:

1. `src/data`
   - schema handling
   - temporal split logic
   - smoke-data helpers
2. `src/features`
   - baseline / research feature engineering
3. `src/models`
   - LightGBM helpers
   - 2-level stacked model
4. `src/metrics`
   - competition weighted score
5. `src/training`
   - CV and ablation support for the baseline path
6. `src/inference`
   - full-train prediction and submission generation

## 6. Practical interpretation

The repository should be read as a research artifact with two layers:

- **production-like baseline path**: stable, simpler, and easier to reproduce
- **final competition architecture**: the 2-level stack, which captures the strongest late-stage modeling idea that remained practical enough to export

For the full research history and why many other directions were rejected, see:
- `docs/experiments.md`
- `docs/ablations.md`
