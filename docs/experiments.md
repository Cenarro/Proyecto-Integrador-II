# Experiments

This document summarizes the main experiment tracks explored during the competition. The goal is not to list every command ever run, but to preserve the major branches, what was learned from them, and the most important numbers.

## 1. Dataset and metric context

Dataset:
- `train.parquet`: `5,337,414` rows
- `test.parquet`: `1,447,107` rows
- 86 anonymized numeric features: `feature_a ... feature_ch`

Primary metric:
- `weighted_rmse_score`

The metric is unforgiving:
- many models with reasonable RMSE still scored `0.0`
- heavy-tail rows matter disproportionately
- models that regressed toward zero often failed completely

## 2. Early baseline stage

### 2.1 Simple baselines
Tried:
- zero predictor
- horizon mean
- grouped means
- recent grouped means

Purpose:
- establish temporal sanity checks
- detect whether hierarchy-only averages carry signal

Outcome:
- useful only as leakage checks
- not competitive

### 2.2 Legacy per-horizon baseline
Earlier stable baseline family:
- per-horizon boosted trees
- simpler feature set

Recorded result from the legacy pipeline:

| Metric Family | Final / Aggregate |
|---|---:|
| Legacy per-horizon baseline aggregate | `0.14881963` |

Per-horizon snapshot from the handoff:

| Horizon | Score |
|---:|---:|
| 1 | `0.04012746` |
| 3 | `0.07800420` |
| 10 | `0.09869897` |
| 25 | `0.18727240` |

This established two recurring facts:
- `h25` was consistently the easiest horizon
- `h1` was consistently the hardest

## 3. Research feature-engineering pipeline

The next phase moved from “simple baseline” to “feature-centric global per-horizon modeling”.

Main feature blocks investigated:
- lag / rolling / EWM features
- momentum terms
- hierarchy-relative features
- cross-sectional normalization and ranks
- missing indicators
- group-aware imputation

Validation emphasis:
- walk-forward / blocked time splits
- no future leakage through shift logic only

### 3.1 LightGBM research runner
This became the main baseline research tool.

Key lesson:
- LightGBM was the most consistently useful family
- GPU fallback to CPU mattered operationally

### 3.2 XGBoost research runner
This was added to test whether XGBoost could outperform LightGBM on the same feature blocks.

Key lesson:
- XGBoost occasionally helped, but it was generally weaker and less stable
- it was still useful as an ensemble ingredient

## 4. Feature-block ablations

One of the clearest ablations came from the fast full-data XGBoost study.

Results:

| Experiment | Lag | Hierarchy | Cross | Missing | Final Metric | h1 | h3 | h10 | h25 |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| `lag_only` | on | off | off | off | `0.13848234` | `0.01294437` | `0.02599295` | `0.06840064` | `0.18505960` |
| `full_stack` | on | on | on | on | `0.09250967` | `0.01500790` | `0.01846197` | `0.07251668` | `0.11601627` |
| `lag_hierarchy` | on | on | off | off | `0.08308245` | `0.01378548` | `0.02261776` | `0.06623358` | `0.10350351` |
| `base_only` | off | off | off | off | `0.00000000` | `0.01412623` | `0.02718282` | `0.05903848` | `0.00000000` |

Conclusion:
- temporal lag structure carried most of the useful signal
- the larger handcrafted stack often added runtime and complexity without improving score

## 5. Deep learning branch

Several DNN directions were tested:
- dense tabular MLPs
- per-horizon DNNs
- feature-treatment preprocessing for DNNs
- non-temporal DNN variants

Observed behavior:
- training loss decreased
- RMSE sometimes looked acceptable
- weighted competition score frequently remained `0.0`

Representative result:

| Horizon | Split | Weighted Score | RMSE |
|---:|---|---:|---:|
| 1 | group_holdout | `0.00000000` | `9.86018109` |
| 3 | group_holdout | `0.00000000` | `20.09728810` |
| 10 | group_holdout | `0.00000000` | `32.80554548` |
| 25 | group_holdout | `0.00000000` | `54.45104640` |

Interpretation:
- DNNs were fitting the central mass but not the heavy rows that drive the metric
- low RMSE alone was misleading

This branch was not competitive enough to remain central.

## 6. Feature-treatment preprocessing branch

A full feature-treatment plan was created for all variables:
- missingness indicators before imputation
- temporal interpolation for low-missing columns
- KNN imputation for moderate-missing columns
- log1p / Yeo-Johnson / standard / robust scaling by feature type
- one-hot encoding for `feature_ch`
- `arcsinh(y_target)` considered for DNN training

This branch produced strong analysis value and clean preprocessing logic, but not a breakthrough model on its own.

Most important boosting result from this branch:

| Horizon | Best Model | Weighted Score | RMSE |
|---:|---|---:|---:|
| 1 | LightGBM | `0.143156` | `11.501254` |
| 3 | LightGBM | `0.212874` | `19.314128` |
| 10 | LightGBM | `0.327116` | `33.420446` |
| 25 | LightGBM | `0.411868` | `50.311154` |

Key lesson:
- feature treatment plus LightGBM was much more effective than feature treatment plus DNN

## 7. Late-stage panel / holdout research

As the competition progressed, the work shifted from standard CV to larger blocked tail holdouts intended to mimic leaderboard transfer better.

Validation protocol used in the strongest late-stage runs:
- blocked holdout, not random split
- large smoke setting:
  - `2,000,000` train rows
  - last `775` unique `ts_index` as holdout

### 7.1 Recent-prior models

Candidates:
- `baseline`
- `recent60`
- `recent180`
- `recent180_delta`

Stable results:

| Candidate | Final Holdout | h1 | h3 | h10 | h25 |
|---|---:|---:|---:|---:|---:|
| `baseline` | `0.118511` | `0.019449` | `0.044130` | `0.051123` | `0.157269` |
| `recent60` | `0.147057` | `0.031089` | `0.067076` | `0.123352` | `0.177475` |
| `recent180` | `0.139943` | `0.034810` | `0.068212` | `0.106128` | `0.172451` |
| `recent180_delta` | `0.137647` | `0.033131` | `0.068529` | `0.128952` | `0.159890` |

Interpretation:
- recent hierarchical target priors were real signal
- recency mattered
- but this still did not produce a large jump

### 7.2 Hard recent-history truncation

Candidates such as:
- `recentfit_900`
- `recentfit_1200`
- `recentfit_1800`
- `recentfit_2400`

Outcome:
- all clearly worse than the main recent-prior branch

Conclusion:
- emphasizing recency through priors helped
- throwing away too much history did not

### 7.3 Covariate-shift / domain weighting

Examples:
- `recent60_domain5`
- `recent60_domain3`
- `recent180_delta_domain5`

Outcome:
- did not beat the main recent-prior setup

Conclusion:
- the implemented domain adaptation weighting was not the missing edge

## 8. Cross-horizon structure branch

A key structural discovery was that for the same:
- `(code, sub_code, sub_category, ts_index)`

most raw features are identical across horizons, and only a small subset varies by horizon.

This led to three late-stage branches.

### 8.1 Cross-horizon delta features

Best result in that family:

| Candidate | Final Holdout |
|---|---:|
| `recent60_xhdelta` | `0.142421` |

Conclusion:
- some local benefit
- not a stable enough standalone breakthrough

### 8.2 Cross-horizon target stacking

Best single-model late-stage result:

| Candidate | Final Holdout | h1 | h3 | h10 | h25 |
|---|---:|---:|---:|---:|---:|
| `recent60_xhdelta_stack` | `0.147720` | `0.034627` | `0.062555` | `0.128275` | `0.176915` |

Interpretation:
- stacking longer-horizon signal back into short horizons helped
- but the gain was still incremental, not game-changing

### 8.3 Horizon-specific hybrid and coarse blend

Best horizon-specific hybrid:
- `h1 -> recent180`
- `h3 -> recent180_delta`
- `h10 -> recent180_delta`
- `h25 -> recent60`

Score:
- `0.148579`

Best coarse per-horizon blend:
- blend of `recent60`, `recent180`, `recent180_delta`

Score:
- `0.150177`

This was the strongest validated late-stage score, but still far from the hoped-for jump.

## 9. Panel stack branch

A more explicit LightGBM panel-stack pipeline was also implemented:
- blocked validation
- same-time panel aggregates
- cross-horizon stacking
- recency weighting
- post-processing via shrink / clipping

Bounded smoke results showed small improvements over the smoke baseline but not enough to justify a final high-confidence submission.

Important conclusion from that branch:
- the idea was valid
- the observed gains were too small

## 10. Final exported 2-level architecture

The final export therefore includes a practical “best-effort” stacked system:

- Level 1:
  - LightGBM per horizon
  - XGBoost per horizon
- Level 2:
  - ridge calibration / blending on level-1 outputs
- per-horizon selection among:
  - LightGBM
  - XGBoost
  - simple mean
  - ridge meta-model

This is not claimed to be the global best research branch numerically. It is the most coherent compact architecture to export:
- tree-based
- reproducible
- time-aware
- includes the late insight that a second calibration layer is useful

## 11. Main takeaways

Across all experiments, the most reliable lessons were:

1. Per-horizon modeling is necessary.
2. Strict time-aware validation matters more than raw CV convenience.
3. LightGBM is the strongest stable base learner.
4. DNNs can look good by RMSE and still fail completely on the competition metric.
5. Lag / rolling temporal structure matters more than large hand-built feature stacks.
6. Recent hierarchical target priors are real signal.
7. Cross-horizon structure helps, but only incrementally in the tested implementations.
8. No tested branch delivered the hoped-for jump to `0.3+`.
