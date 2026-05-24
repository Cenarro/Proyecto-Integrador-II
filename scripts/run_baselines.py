from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.splits import load_data, validate_schema, build_time_split
from src.metrics.skill import weighted_rmse_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run simple temporal baselines.')
    p.add_argument('--train-path', type=Path, default=Path('src/data/train.parquet'))
    p.add_argument('--test-path', type=Path, default=Path('src/data/test.parquet'))
    p.add_argument('--val-ratio', type=float, default=0.15)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/predictions'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_data(args.train_path, args.test_path)
    validate_schema(train_df, test_df)
    train_mask, val_mask, split_ts = build_time_split(train_df, val_ratio=args.val_ratio)
    fit_df = train_df.filter(pl.Series(train_mask))
    val_df = train_df.filter(pl.Series(val_mask))

    horizon_mean = fit_df.group_by('horizon').agg(pl.col('y_target').mean().alias('pred'))
    val_pred = val_df.join(horizon_mean, on='horizon', how='left').with_columns(pl.col('pred').fill_null(0.0))
    overall = weighted_rmse_score(
        y_target=val_pred.get_column('y_target').to_numpy(),
        y_pred=val_pred.get_column('pred').to_numpy(),
        w=val_pred.get_column('weight').to_numpy(),
    )

    summary = {
        'baseline': 'horizon_mean',
        'split_ts': int(split_ts),
        'final_metric': float(overall),
        'rows_fit': int(fit_df.height),
        'rows_val': int(val_df.height),
    }
    (args.output_dir / 'baseline_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
