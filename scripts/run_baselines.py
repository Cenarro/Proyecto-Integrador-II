"""Run strict temporal baselines for the hedge fund competition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.competition_pipeline import (
    DataSchema,
    detect_schema,
    ensure_dir,
    load_table,
    regression_metrics,
    save_json,
    summarize_by_horizon,
    temporal_holdout_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=ROOT / "train.parquet")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "baselines"
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--recent-window", type=int, default=180)
    return parser.parse_args()


def _weights(df: pd.DataFrame, schema: DataSchema) -> np.ndarray:
    if schema.weight_col is None:
        return np.ones(len(df), dtype=np.float32)
    return df[schema.weight_col].to_numpy(dtype=np.float32)


def _horizon_mean_predictor(train_df: pd.DataFrame, val_df: pd.DataFrame, schema: DataSchema) -> np.ndarray:
    means = train_df.groupby(schema.horizon_col)[schema.target_col].mean().to_dict()
    return val_df[schema.horizon_col].map(means).fillna(0.0).to_numpy(dtype=np.float64)


def _group_mean_predictor(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    schema: DataSchema,
    *,
    recent_only: bool,
    recent_window: int,
) -> np.ndarray:
    assert schema.target_col is not None
    group_cols = [schema.horizon_col]
    if "code" in train_df.columns:
        group_cols.append("code")
    if "sub_category" in train_df.columns:
        group_cols.append("sub_category")
    source = train_df
    if recent_only:
        cutoff = int(train_df[schema.ts_col].max()) - recent_window + 1
        source = train_df.loc[train_df[schema.ts_col] >= cutoff].copy()
        if source.empty:
            source = train_df
    grouped = source.groupby(group_cols)[schema.target_col].mean().rename("group_mean")
    horizon_mean = train_df.groupby(schema.horizon_col)[schema.target_col].mean()
    pred = val_df.merge(grouped, on=group_cols, how="left")
    pred["prediction"] = pred["group_mean"]
    pred["prediction"] = pred["prediction"].fillna(pred[schema.horizon_col].map(horizon_mean))
    return pred["prediction"].fillna(0.0).to_numpy(dtype=np.float64)


def _evaluate_candidate(
    name: str,
    preds: np.ndarray,
    val_df: pd.DataFrame,
    schema: DataSchema,
) -> list[dict[str, float | int | str]]:
    assert schema.target_col is not None
    scored = val_df[[schema.horizon_col, schema.target_col]].copy()
    if schema.weight_col is not None:
        scored[schema.weight_col] = val_df[schema.weight_col].to_numpy()
    scored["prediction"] = preds
    rows: list[dict[str, float | int | str]] = []
    for horizon, horizon_df in scored.groupby(schema.horizon_col, sort=True):
        metrics = regression_metrics(
            horizon_df[schema.target_col].to_numpy(dtype=np.float64),
            horizon_df["prediction"].to_numpy(dtype=np.float64),
            _weights(horizon_df, schema),
        )
        rows.append(
            {
                "baseline": name,
                "horizon": int(horizon),
                "rows": int(len(horizon_df)),
                **metrics,
            }
        )
    return rows


def main() -> None:
    global args
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    train_df = load_table(args.train_path)
    schema = detect_schema(train_df, require_target=True)
    train_split, val_split = temporal_holdout_split(
        train_df, schema.ts_col, args.val_fraction
    )

    candidates = {
        "zero": np.zeros(len(val_split), dtype=np.float64),
        "horizon_mean": _horizon_mean_predictor(train_split, val_split, schema),
        "group_mean": _group_mean_predictor(
            train_split, val_split, schema, recent_only=False, recent_window=args.recent_window
        ),
        "recent_group_mean": _group_mean_predictor(
            train_split, val_split, schema, recent_only=True, recent_window=args.recent_window
        ),
    }

    rows: list[dict[str, float | int | str]] = []
    for name, preds in candidates.items():
        rows.extend(_evaluate_candidate(name, preds, val_split, schema))

    per_horizon = pd.DataFrame(rows).sort_values(["baseline", "horizon"]).reset_index(
        drop=True
    )
    overall = (
        per_horizon.groupby("baseline", as_index=False)[
            ["weighted_score", "rmse", "mape", "ratio_sse_sst", "corr", "std_pred"]
        ]
        .mean(numeric_only=True)
        .sort_values("weighted_score", ascending=False)
    )

    per_horizon.to_csv(output_dir / "baseline_per_horizon.csv", index=False)
    overall.to_csv(output_dir / "baseline_summary.csv", index=False)
    save_json(
        output_dir / "run_config.json",
        {
            "train_path": str(args.train_path),
            "val_fraction": args.val_fraction,
            "recent_window": args.recent_window,
            "detected_schema": schema.__dict__,
            "train_rows": int(len(train_split)),
            "val_rows": int(len(val_split)),
        },
    )

    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
