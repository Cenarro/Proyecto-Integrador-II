# How To Run

This repo expects the Kaggle parquet files in:

```text
src/data/train.parquet
src/data/test.parquet
```

The parquet files are tracked with Git LFS. After cloning the repo, install Git LFS and pull the data:

```bash
git lfs install
git lfs pull
```

## 1. Set Up Python

If you already have the `dl` conda environment:

```bash
conda activate dl
pip install -r requirements.txt
```

If you need a fresh environment:

```bash
conda create -n hedge-forecasting python=3.11 -y
conda activate hedge-forecasting
pip install -r requirements.txt
```

## 2. Check The Data Files

This reads only parquet metadata, not the full dataset:

```bash
python -c "import pyarrow.parquet as pq; [print(p, pq.ParquetFile(p).metadata.num_rows, 'rows') for p in ['src/data/train.parquet', 'src/data/test.parquet']]"
```

Expected files:

```text
src/data/train.parquet
src/data/test.parquet
```

## 3. Verify The Code Imports

```bash
python -m compileall -q src scripts
python scripts/run_two_level_submission.py --help
python scripts/run_cv.py --help
```

## 4. Run A Simple Baseline

This is the quickest real run. It creates `outputs/predictions/baseline_summary.json`.

```bash
python scripts/run_baselines.py
```

## 5. Run Cross-Validation

Validation only, without the final full refit:

```bash
python scripts/run_cv.py --skip-final-fit
```

Full validation plus final refit and submission:

```bash
python scripts/run_cv.py
```

Outputs are written to:

```text
outputs/predictions/
```

## 6. Create The Main 2-Level Submission

This is the primary stacked architecture:

```bash
python scripts/run_two_level_submission.py --output-dir outputs/predictions/2-level_submission
```

The expected submission file is:

```text
outputs/predictions/2-level_submission/submission.csv
```

## 7. CPU-Only Runs

If GPU support is not available, add `--no-use-gpu`:

```bash
python scripts/run_two_level_submission.py --no-use-gpu --output-dir outputs/predictions/2-level_submission
python scripts/run_cv.py --no-use-gpu --skip-final-fit
python scripts/run_training.py --no-use-gpu
```

## 8. Rebuild A Submission From CV Results

After `run_cv.py` creates `outputs/predictions/validation_summary_full.json`, run:

```bash
python scripts/make_submission.py --validation-summary outputs/predictions/validation_summary_full.json
```

The submission is written to:

```text
outputs/predictions/submission.csv
```

## Notes

- All scripts default to `src/data/train.parquet` and `src/data/test.parquet`.
- Generated files under `outputs/` are ignored by git.
- The full training commands can take a long time on CPU.
- Use `--help` on any script to inspect available options.

## Notebooks

The `notebooks/` folder contains two runnable notebooks:

```text
notebooks/exploration.ipynb
notebooks/kaggle_submission.ipynb
```

Use `exploration.ipynb` for schema checks, metadata, horizon summaries, missing-value sampling, and simple split inspection. Use `kaggle_submission.ipynb` to print or launch the submission commands from inside Jupyter. Long-running notebook cells are disabled by default through `RUN_* = False` flags.
