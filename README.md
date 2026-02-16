# Kairos — Robust Multihorizon Time Series Forecasting (Proyecto Integrador II)

Kairos is a reproducible end-to-end forecasting pipeline designed for the Kaggle **Hedge Fund Time Series Forecasting** challenge. The project focuses on **multiseries, multihorizon forecasting** under **strict temporal constraints**, where predictions at time index `t` must only use data from indices `0..t` (no look-ahead, no leakage).

The system includes:
- A sequential data pipeline and temporal split generator
- Strong baselines (naive / statistical / tree-based)
- A global ML forecasting model for cross-series generalization
- Rigorous temporal validation and diagnostics per horizon and group
- Reproducibility tooling (pinned dependencies, fixed seeds, executable scripts)

---

## Project Context

Forecasting systems often fail in real deployment because evaluation allows implicit leakage or does not reflect strict causality. This competition enforces sequential processing and evaluates performance mainly via a **private leaderboard**, promoting generalization beyond the training sample.

**Task:** Predict continuous values for each combination of:
- `code`, `sub_code`, `sub_category`, `horizon`
with strict chronological processing by `ts_index`.

---

## Evaluation Metric

Performance is measured using a **weighted RMSE skill score**:

$$
\text{Score} =
\sqrt{
  1 - \min\!\left(
    \max\!\left(
      \frac{\sum_{i \in I} w_i (y_i - \hat{y}_i)^2}
           {\sum_{i \in I} w_i y_i^2},
      0
    \right),
    1
  \right)
}
$$


Where:
- `I` = evaluation subset
- `w_i` = observation weights
- `y_i` = true values
- `ŷ_i` = predictions

---

## Objectives

### General objective
Build a **leakage-free**, **reproducible** forecasting pipeline that generalizes out of sample across identifiers and horizons.

### Specific objectives
- Implement sequential ingestion + temporal splits with strict causality
- Establish baseline suite and benchmarking framework
- Train a global ML model conditioned on horizon/groups
- Implement temporal cross-validation + horizon-wise diagnostics
- Run ablation studies (features, models, ensembling)
- Deliver a fully reproducible package (pinned deps, seeds, scripts, protocols)

---

## Scope

### Included
- Feature engineering under strict temporal constraints
- Multihorizon training/inference with weighted metrics
- Validation + diagnostics + ablation studies
- Reproducibility package and Kaggle submission notebook

### Excluded
- Live trading execution and portfolio optimization
- External private datasets not allowed by Kaggle rules
- Any method violating sequential processing constraints

---

## Repository Structure (recommended)

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── data/                # ingestion, validation, split logic
│   ├── features/            # lag/rolling/horizon/group features
│   ├── models/              # baselines + ML models
│   ├── metrics/             # weighted skill score implementation
│   ├── training/            # train/validate loops, CV runner
│   └── inference/           # submission generation, sequential predict
├── scripts/
│   ├── run_baselines.py
│   ├── run_training.py
│   ├── run_cv.py
│   └── make_submission.py
├── notebooks/
│   ├── exploration.ipynb
│   └── kaggle_submission.ipynb
├── tests/
│   ├── test_no_leakage.py
│   ├── test_metric.py
│   └── test_splits.py
├── docs/
│   ├── architecture.md
│   ├── experiments.md
│   └── ablations.md
└── reports/
    ├── proposal.pdf
    └── final_report.pdf
