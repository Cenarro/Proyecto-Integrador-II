"""Run leakage-safe temporal cross-validation for a global tree model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.competition_pipeline import (
    build_model,
    detect_schema,
    ensure_dir,
    fit_model,
    fit_preprocessor,
    load_table,
    predict_model,
    regression_metrics,
    save_json,
    split_from_window,
    summarize_by_horizon,
    temporal_cv_windows,
    transform_with_preprocessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=ROOT / "train.parquet")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "cv_runs"
    )
    parser.add_argument(
        "--model", choices=("lgbm", "xgb", "catboost"), default="lgbm"
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--val-size", type=int, default=180)
    parser.add_argument("--min-train-size", type=int, default=2200)
    parser.add_argument("--num-boost-round", type=int, default=600)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    df = load_table(args.train_path)
    schema = detect_schema(df, require_target=True)
    assert schema.target_col is not None

    windows = temporal_cv_windows(
        df[schema.ts_col].dropna().unique(),
        n_splits=args.n_splits,
        val_size=args.val_size,
        min_train_size=args.min_train_size,
    )

    rows: list[dict[str, float | int | str]] = []
    fold_iter = tqdm(windows, desc="cv folds", disable=not args.show_progress)
    for window in fold_iter:
        train_df, val_df = split_from_window(df, schema.ts_col, window)
        for horizon in sorted(df[schema.horizon_col].dropna().unique()):
            horizon = int(horizon)
            train_h = train_df.loc[train_df[schema.horizon_col] == horizon].copy()
            val_h = val_df.loc[val_df[schema.horizon_col] == horizon].copy()
            if train_h.empty or val_h.empty:
                continue

            prepared = fit_preprocessor(train_h, schema)
            X_train = prepared.frame[prepared.feature_cols]
            X_val = transform_with_preprocessor(val_h, schema, prepared)
            y_train = train_h[schema.target_col].to_numpy(dtype="float32")
            y_val = val_h[schema.target_col].to_numpy(dtype="float32")
            w_train = (
                train_h[schema.weight_col].to_numpy(dtype="float32")
                if schema.weight_col is not None
                else None
            )
            w_val = (
                val_h[schema.weight_col].to_numpy(dtype="float32")
                if schema.weight_col is not None
                else None
            )

            model = build_model(
                args.model,
                seed=args.seed + window.fold * 100 + horizon,
                num_boost_round=args.num_boost_round,
                use_gpu=args.use_gpu,
                early_stopping_rounds=args.early_stopping_rounds,
            )
            model = fit_model(
                model,
                args.model,
                X_train,
                y_train,
                w_train,
                X_val=X_val,
                y_val=y_val,
                w_val=w_val,
                categorical_cols=prepared.categorical_cols,
                early_stopping_rounds=args.early_stopping_rounds,
            )
            preds = predict_model(model, X_val)
            metrics = regression_metrics(y_val, preds, w_val)
            rows.append(
                {
                    "model": args.model,
                    "fold": window.fold,
                    "horizon": horizon,
                    "train_rows": int(len(train_h)),
                    "val_rows": int(len(val_h)),
                    "train_ts_max": window.train_ts_max,
                    "val_ts_min": window.val_ts_min,
                    "val_ts_max": window.val_ts_max,
                    **metrics,
                }
            )

    result_df = pd.DataFrame(rows).sort_values(["fold", "horizon"]).reset_index(drop=True)
    summary_df = summarize_by_horizon(rows)
    overall_df = (
        result_df.groupby("model", as_index=False)[
            ["weighted_score", "rmse", "mape", "ratio_sse_sst", "corr", "std_pred"]
        ]
        .mean(numeric_only=True)
    )

    result_df.to_csv(output_dir / "cv_results.csv", index=False)
    summary_df.to_csv(output_dir / "cv_summary_by_horizon.csv", index=False)
    overall_df.to_csv(output_dir / "cv_summary_overall.csv", index=False)
    save_json(
        output_dir / "run_config.json",
        {
            "train_path": str(args.train_path),
            "model": args.model,
            "n_splits": args.n_splits,
            "val_size": args.val_size,
            "min_train_size": args.min_train_size,
            "num_boost_round": args.num_boost_round,
            "early_stopping_rounds": args.early_stopping_rounds,
            "use_gpu": args.use_gpu,
            "schema": schema.__dict__,
            "folds": [window.__dict__ for window in windows],
        },
    )

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
