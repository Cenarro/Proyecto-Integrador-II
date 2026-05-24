# Ablations

This document collects the main ablation and diagnostic studies that shaped the final system. The point is to capture what changed model quality and what turned out to be noise.

## 1. Validation ablations

Before feature or model ablations mattered, the project had to settle the validation question.

### Compared
- random split
- per-series random split
- blocked tail holdout
- walk-forward CV
- explicit base / meta / eval split for stacking

### Conclusion
- random splits were too optimistic
- blocked time-aware validation better matched transfer behavior
- stacking required a dedicated meta window to stay leakage-safe

This is why the export emphasizes:
- temporal holdouts
- walk-forward logic
- 2-level base/meta/eval segmentation

## 2. Feature-block ablations

The cleanest quantitative ablation came from the fast XGBoost study.

| Experiment | Lag | Hierarchy | Cross | Missing | Final Metric | Main Finding |
|---|---|---|---|---|---:|---|
| `base_only` | off | off | off | off | `0.00000000` | Raw tabular input alone was not enough |
| `lag_only` | on | off | off | off | `0.13848234` | Temporal lags and rolling statistics were the dominant signal source |
| `lag_hierarchy` | on | on | off | off | `0.08308245` | Hierarchy features did not help under this setup |
| `full_stack` | on | on | on | on | `0.09250967` | More features increased complexity without improving the metric |

### Interpretation
- the largest gain came from temporal lag structure
- hierarchy/cross/missing blocks were not universally bad, but they were not the core driver
- “more engineered features” was not equivalent to “better score”

## 3. Model-family ablations

### 3.1 LightGBM vs XGBoost

Observed pattern:
- LightGBM was the strongest consistent model family
- XGBoost occasionally added diversity but rarely beat LightGBM on its own

Example from the feature-treatment boosting branch:

| Horizon | LightGBM | XGBoost |
|---:|---:|---:|
| 1 | `0.143156` | `0.000000` |
| 3 | `0.212874` | `0.000000` |
| 10 | `0.327116` | `0.000000` |
| 25 | `0.411868` | `0.000000` |

### 3.2 Deep tabular models vs tree models

DNNs were repeatedly tested and repeatedly underperformed on the actual competition metric.

Pattern:
- train loss dropped
- RMSE could look reasonable
- weighted score frequently stayed at `0.0`

Conclusion:
- tree models were much more robust for this dataset
- DNNs were not discarded because of aesthetics; they were discarded because the metric did not reward them

## 4. Preprocessing ablations

The feature-treatment branch tested a systematic preprocessing policy:
- missing indicators before imputation
- temporal interpolation for low-missing columns
- KNN imputation for moderate-missing columns
- Yeo-Johnson / log1p / robust scaling by distribution type
- one-hot encoding for `feature_ch`
- `arcsinh` target transform considered for DNNs

### Findings
- the preprocessing policy was valuable for data hygiene and reproducibility
- it did **not** rescue the DNN branch
- it worked much better with LightGBM than with neural nets

Practical conclusion:
- preprocessing quality matters
- model family still mattered more

## 5. Recent-data ablations

Late in the competition, the research focused on recency because train/test drift was obvious.

### Compared
- baseline
- recent hierarchical priors over 60 steps
- recent hierarchical priors over 180 steps
- recent-prior deltas
- hard history truncation

### Stable results

| Candidate | Final Holdout |
|---|---:|
| `baseline` | `0.118511` |
| `recent60` | `0.147057` |
| `recent180` | `0.139943` |
| `recent180_delta` | `0.137647` |

### Hard recent-history cutoffs

| Candidate | Final Holdout |
|---|---:|
| `recentfit_900` | `0.122148` |
| `recentfit_1200` | `0.115701` |
| `recentfit_1800` | `0.113287` |
| `recentfit_2400` | `0.124838` |

### Interpretation
- recency was important
- but “use only recent rows” was not enough
- recency helped most when encoded as structured priors rather than hard truncation

## 6. Domain-weighting ablation

Covariate-shift weighting was tested to push train closer to test.

Representative results:

| Candidate | Final Holdout |
|---|---:|
| `recent60_domain5` | `0.144158` |
| `recent60_domain3` | `0.139534` |
| `recent180_delta_domain5` | `0.131809` |

Compared with:
- `recent60 = 0.147057`

Conclusion:
- the domain-weighting implementation did not beat the simpler recent-prior model

## 7. Cross-horizon ablations

The research uncovered that most raw features are identical across horizons at the same entity-time key. That motivated several cross-horizon ablations.

### 7.1 Cross-horizon delta features

| Candidate | Final Holdout |
|---|---:|
| `recent60_xhdelta` | `0.142421` |
| same-run `recent60` baseline | `0.136114` |

Conclusion:
- some within-run improvement
- weak stability across seeds

### 7.2 Multi-seed stability test

Seed-level summary:
- seed `42`: `recent60 = 0.123902`, `recent60_xhdelta = 0.124067`
- seed `314`: `recent60 = 0.119526`, `recent60_xhdelta = 0.122141`
- seed `2718`: `recent60 = 0.108293`, `recent60_xhdelta = 0.107207`

Seed-averaged:
- `recent60 = 0.127898`
- `recent60_xhdelta = 0.127930`
- 50/50 blend = `0.129283`

Conclusion:
- the cross-horizon delta block was not a robust standalone leap

### 7.3 Cross-horizon target stacking

| Candidate | Final Holdout |
|---|---:|
| `recent60_stack` | `0.142343` |
| `recent60_xhdelta_stack` | `0.147720` |

Conclusion:
- stacking longer-horizon information back into short horizons helped
- but the gain was still incremental

## 8. Blending ablations

Two useful late-stage blend studies were run.

### Horizon-specific hybrid
- `h1 -> recent180`
- `h3 -> recent180_delta`
- `h10 -> recent180_delta`
- `h25 -> recent60`

Score:
- `0.148579`

### Coarse per-horizon blend

Best coarse blend:
- `h1 = 0.25 recent60 + 0.50 recent180 + 0.25 recent180_delta`
- `h3 = 0.25 recent60 + 0.25 recent180 + 0.50 recent180_delta`
- `h10 = 0.25 recent60 + 0.00 recent180 + 0.75 recent180_delta`
- `h25 = 0.75 recent60 + 0.25 recent180 + 0.00 recent180_delta`

Score:
- `0.150177`

Conclusion:
- modest blending helped
- but the improvement was still not large enough to change the overall competition story

## 9. Panel-stack ablation

A later LightGBM panel-stack pipeline tested:
- same-time panel aggregates
- recency weighting
- cross-horizon meta-features
- post-hoc shrink / clipping

Smoke findings:
- panel aggregates helped somewhat
- recency weighting was not consistently positive
- cross-horizon stack improved smoke metrics slightly
- the full gain was still too small to justify high confidence

Conclusion:
- valid idea
- insufficient magnitude

## 10. What the ablations proved

The combined ablation record is consistent:

1. Temporal lag structure is the dominant raw feature source.
2. LightGBM is the most stable base learner.
3. DNNs fail on the competition metric even when RMSE looks superficially acceptable.
4. Recent hierarchical target priors are real signal.
5. Cross-horizon structure helps, but the tested implementations only gave incremental gains.
6. Larger feature stacks and more complicated weighting schemes did not automatically improve the score.

That is why the exported repository centers on:
- strong tree-based level-1 learners
- conservative level-2 calibration
- strict temporal segmentation

rather than a more exotic but less defensible architecture.
