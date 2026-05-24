# Hedge Fund Time Series Forecasting

Reproducible repository for the Kaggle hedge fund multihorizon forecasting competition. The main pipeline is a 2-level horizon-wise stack backed by a simpler LightGBM baseline workflow.

## What Is Included

- 2-level horizon-wise architecture:
  - level 1: LightGBM and XGBoost per horizon
  - level 2: ridge calibrator/blender on top of level-1 predictions
- strict time-based train/meta/eval segmentation
- target encodings, interaction features, lag/rolling/EWM blocks, hierarchy-relative features, and cross-sectional normalization
- schema validation and temporal split logic
- LightGBM training with GPU fallback to CPU
- walk-forward cross-validation and ablation support
- submission generation from full-train refit

## Setup

```bash
pip install -r requirements.txt
```

The Kaggle parquet files are expected at:

```text
src/data/train.parquet
src/data/test.parquet
```

These parquet files are tracked with Git LFS because they exceed GitHub's normal file-size limit. Generated outputs are ignored by git.

## Quick Start

For a complete command-by-command guide, see [HOW_TO_RUN.md](HOW_TO_RUN.md).

Primary stacked submission:

```bash
python scripts/run_two_level_submission.py --output-dir outputs/predictions/2-level_submission
```

Baseline workflows:

```bash
python scripts/run_baselines.py
python scripts/run_cv.py --skip-final-fit
python scripts/run_training.py --mode full
python scripts/make_submission.py --validation-summary outputs/predictions/validation_summary_full.json
```

## Repository Layout

```text
.
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- docs/
|   |-- architecture.md
|   |-- experiments.md
|   `-- ablations.md
|-- notebooks/
|   |-- exploration.ipynb
|   `-- kaggle_submission.ipynb
|-- scripts/
|   |-- run_two_level_submission.py
|   |-- run_baselines.py
|   |-- run_cv.py
|   |-- run_training.py
|   `-- make_submission.py
`-- src/
    |-- data/
    |-- features/
    |-- inference/
    |-- metrics/
    |-- models/
    `-- training/
```