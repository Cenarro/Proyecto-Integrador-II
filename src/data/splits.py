from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

REQUIRED_TRAIN_COLUMNS = {
    "id",
    "code",
    "sub_code",
    "sub_category",
    "horizon",
    "ts_index",
    "weight",
    "y_target",
}

REQUIRED_TEST_COLUMNS = REQUIRED_TRAIN_COLUMNS - {"y_target", "weight"}
KEY_COLUMNS = ["code", "sub_code", "sub_category", "horizon", "ts_index"]


def load_data(train_path: str | Path, test_path: str | Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    train_df = pl.read_parquet(train_path)
    test_df = pl.read_parquet(test_path)
    return train_df, test_df


def _has_duplicates(df: pl.DataFrame, cols: list[str]) -> bool:
    return df.select(cols).unique().height != df.height


def validate_schema(train_df: pl.DataFrame, test_df: pl.DataFrame) -> None:
    missing_train = REQUIRED_TRAIN_COLUMNS - set(train_df.columns)
    missing_test = REQUIRED_TEST_COLUMNS - set(test_df.columns)

    if missing_train:
        raise ValueError(f"Missing required train columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"Missing required test columns: {sorted(missing_test)}")
    if "y_target" in test_df.columns:
        raise ValueError("test dataframe must not contain y_target")

    if train_df.select(pl.col("id").is_duplicated().any()).item():
        raise ValueError("Duplicate ids found in train")
    if test_df.select(pl.col("id").is_duplicated().any()).item():
        raise ValueError("Duplicate ids found in test")
    if _has_duplicates(train_df, KEY_COLUMNS):
        raise ValueError("Duplicate composite keys found in train")
    if _has_duplicates(test_df, KEY_COLUMNS):
        raise ValueError("Duplicate composite keys found in test")

    valid_horizons = {1, 3, 10, 25}
    train_horizons = set(int(x) for x in train_df.get_column("horizon").unique().to_list())
    test_horizons = set(int(x) for x in test_df.get_column("horizon").unique().to_list())

    if not train_horizons.issubset(valid_horizons):
        raise ValueError(f"Unexpected train horizons: {sorted(train_horizons - valid_horizons)}")
    if not test_horizons.issubset(valid_horizons):
        raise ValueError(f"Unexpected test horizons: {sorted(test_horizons - valid_horizons)}")

    train_max_ts = int(train_df.select(pl.max("ts_index")).item())
    test_min_ts = int(test_df.select(pl.min("ts_index")).item())
    if train_max_ts >= test_min_ts:
        raise ValueError("Temporal boundary violated: train max ts_index must be < test min ts_index")


def build_time_split(
    train_df: pl.DataFrame,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    unique_ts = np.sort(train_df.get_column("ts_index").unique().to_numpy())
    if len(unique_ts) < 3:
        raise ValueError("Need at least 3 distinct ts_index values for a split")

    split_idx = int(np.floor((1.0 - val_ratio) * len(unique_ts)))
    split_idx = max(1, min(split_idx, len(unique_ts) - 1))
    val_start_ts = int(unique_ts[split_idx])

    ts_vals = train_df.get_column("ts_index").to_numpy()
    train_mask = ts_vals < val_start_ts
    val_mask = ~train_mask

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError("Invalid split produced empty train or validation set")

    return train_mask, val_mask, val_start_ts


def build_walk_forward_folds(
    ts_index_values: np.ndarray,
    n_folds: int = 3,
    val_size: int = 180,
    min_train_size: int = 800,
) -> list[dict[str, int]]:
    """Build deterministic walk-forward folds on unique sorted ts_index values."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if val_size < 1:
        raise ValueError("val_size must be >= 1")
    if min_train_size < 1:
        raise ValueError("min_train_size must be >= 1")

    unique_ts = np.sort(np.unique(np.asarray(ts_index_values)))
    n_ts = len(unique_ts)
    if n_ts <= min_train_size + val_size:
        raise ValueError("Not enough unique ts_index values for requested walk-forward setup")

    total_val = n_folds * val_size
    if total_val >= n_ts:
        raise ValueError("n_folds * val_size must be smaller than number of unique ts_index values")

    start_idx = n_ts - total_val
    if start_idx < min_train_size:
        start_idx = min_train_size

    folds: list[dict[str, int]] = []
    for i in range(n_folds):
        val_start_idx = start_idx + i * val_size
        val_end_idx = min(val_start_idx + val_size - 1, n_ts - 1)
        if val_start_idx >= n_ts:
            break

        val_start_ts = int(unique_ts[val_start_idx])
        val_end_ts = int(unique_ts[val_end_idx])
        train_end_ts = int(unique_ts[val_start_idx - 1])

        folds.append(
            {
                "fold": i + 1,
                "train_end_ts": train_end_ts,
                "val_start_ts": val_start_ts,
                "val_end_ts": val_end_ts,
            }
        )

    if not folds:
        raise ValueError("No valid folds were generated")
    return folds


def stratified_sample_by_horizon(
    df: pl.DataFrame,
    total_rows: int,
    random_seed: int = 42,
) -> pl.DataFrame:
    if total_rows <= 0:
        raise ValueError("total_rows must be > 0")
    if df.height <= total_rows:
        return df.clone()

    horizons = sorted(int(x) for x in df.get_column("horizon").unique().to_list())
    base = total_rows // len(horizons)
    rem = total_rows % len(horizons)

    sampled_parts: list[pl.DataFrame] = []
    for i, horizon in enumerate(horizons):
        group_df = df.filter(pl.col("horizon") == horizon)
        n_rows = base + (1 if i < rem else 0)
        n_rows = min(n_rows, group_df.height)
        sampled_parts.append(group_df.sample(n=n_rows, seed=random_seed + i, shuffle=True))

    sampled = pl.concat(sampled_parts, how="vertical_relaxed")
    if sampled.height < total_rows:
        extra_needed = total_rows - sampled.height
        remaining = df.join(sampled.select("id"), on="id", how="anti")
        extra = remaining.sample(n=extra_needed, seed=random_seed + 91, shuffle=True)
        sampled = pl.concat([sampled, extra], how="vertical_relaxed")

    sampled = sampled.sample(fraction=1.0, seed=random_seed, shuffle=True)
    return sampled
