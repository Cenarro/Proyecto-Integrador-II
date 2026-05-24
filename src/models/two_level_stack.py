from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from xgboost import XGBRegressor

from src.metrics.skill import weighted_rmse_score

VALID_HORIZONS = (1, 3, 10, 25)
TOP_FEATURES = ["feature_al", "feature_am", "feature_cg", "feature_by", "feature_s"]
LAGS = (1, 3, 5, 10, 25)
ROLL_WINDOWS = (5, 10)


@dataclass
class TwoLevelConfig:
    train_path: Path = Path("src/data/train.parquet")
    test_path: Path = Path("src/data/test.parquet")
    output_dir: Path = Path("outputs/predictions/2-level_submission")
    level1_threshold: int = 3500
    meta_threshold: int = 3550
    inner_val_size: int = 120
    lgb_seeds: tuple[int, ...] = (42, 2024, 12345, 99, 420)
    xgb_seeds: tuple[int, ...] = (42, 2024, 12345, 99, 420)
    use_gpu: bool = True
    show_progress: bool = True
    max_train_rows_per_horizon: int = 0
    max_test_rows_per_horizon: int = 0
    lgb_n_estimators: int = 4200
    lgb_early_stopping_rounds: int = 200
    xgb_n_estimators: int = 2800
    xgb_early_stopping_rounds: int = 200
    xgb_min_rounds: int = 50


@dataclass
class HorizonResult:
    horizon: int
    selected_model: str
    lgb_score: float
    xgb_score: float
    mean_score: float
    meta_score: float
    rows_meta: int
    rows_eval: int
    best_iterations_lgb: list[int]
    best_iterations_xgb: list[int]


def load_horizon_frames(
    train_path: Path,
    test_path: Path,
    horizon: int,
    *,
    max_train_rows: int = 0,
    max_test_rows: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cols = ["id", "code", "sub_code", "sub_category", "horizon", "ts_index"]
    train_schema = pl.scan_parquet(str(train_path)).collect_schema().names()
    feature_cols = sorted(col for col in train_schema if col.startswith("feature_"))

    train_cols = [*base_cols, "weight", "y_target", *feature_cols]
    test_cols = [*base_cols, *feature_cols]

    train_df = (
        pl.scan_parquet(str(train_path))
        .filter(pl.col("horizon") == horizon)
        .select(train_cols)
        .sort(["code", "sub_code", "sub_category", "horizon", "ts_index"])
        .collect()
        .to_pandas()
    )
    test_df = (
        pl.scan_parquet(str(test_path))
        .filter(pl.col("horizon") == horizon)
        .select(test_cols)
        .sort(["code", "sub_code", "sub_category", "horizon", "ts_index"])
        .collect()
        .to_pandas()
    )

    if max_train_rows > 0 and len(train_df) > max_train_rows:
        train_df = train_df.tail(max_train_rows).copy()
    if max_test_rows > 0 and len(test_df) > max_test_rows:
        test_df = test_df.head(max_test_rows).copy()

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_encoding_stats(train_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "sub_category": train_df.groupby("sub_category")["y_target"].mean().to_dict(),
        "sub_code": train_df.groupby("sub_code")["y_target"].mean().to_dict(),
        "global_mean": float(train_df["y_target"].mean()),
    }


def _safe_float32(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(np.float32)


def build_context_features(df: pd.DataFrame, enc_stats: dict[str, Any] | None) -> pd.DataFrame:
    x = df.copy()
    group_cols = ["code", "sub_code", "sub_category", "horizon"]

    if enc_stats is not None:
        x["sub_category_enc"] = x["sub_category"].map(enc_stats["sub_category"]).fillna(
            enc_stats["global_mean"]
        )
        x["sub_code_enc"] = x["sub_code"].map(enc_stats["sub_code"]).fillna(
            enc_stats["global_mean"]
        )

    if {"feature_al", "feature_am"}.issubset(x.columns):
        x["d_al_am"] = x["feature_al"] - x["feature_am"]
        x["r_al_am"] = x["feature_al"] / (x["feature_am"].abs() + 1e-7)
    if {"feature_cg", "feature_by"}.issubset(x.columns):
        x["d_cg_by"] = x["feature_cg"] - x["feature_by"]

    for col in TOP_FEATURES:
        if col not in x.columns:
            continue
        ts_mean = x.groupby("ts_index")[col].transform("mean")
        ts_std = x.groupby("ts_index")[col].transform("std").replace(0.0, np.nan)
        x[f"{col}_cs_z"] = ((x[col] - ts_mean) / (ts_std + 1e-6)).astype(np.float32)

        group = x.groupby(group_cols, sort=False)[col]
        for lag in LAGS:
            x[f"{col}_lag{lag}"] = group.shift(lag).astype(np.float32)
        x[f"{col}_diff1"] = group.diff(1).astype(np.float32)
        for window in ROLL_WINDOWS:
            x[f"{col}_roll{window}"] = group.transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            ).astype(np.float32)
            x[f"{col}_rollstd{window}"] = group.transform(
                lambda s: s.rolling(window, min_periods=1).std()
            ).astype(np.float32)
        x[f"{col}_ewm5"] = group.transform(
            lambda s: s.ewm(span=5, adjust=False).mean()
        ).astype(np.float32)

    x["t_cycle_sin"] = np.sin(2 * np.pi * x["ts_index"] / 100.0).astype(np.float32)
    x["t_cycle_cos"] = np.cos(2 * np.pi * x["ts_index"] / 100.0).astype(np.float32)

    feature_cols = [c for c in x.columns if c.startswith("feature_")]
    for col in feature_cols:
        x[col] = _safe_float32(x[col])

    extra_numeric = [
        col
        for col in x.columns
        if col.endswith("_enc")
        or col.startswith("d_")
        or col.startswith("r_")
        or col.endswith("_diff1")
        or "_lag" in col
        or "_roll" in col
        or "_rollstd" in col
        or "_ewm" in col
        or col.endswith("_cs_z")
        or col.startswith("t_cycle_")
    ]
    for col in extra_numeric:
        x[col] = _safe_float32(x[col])

    for col in ["code", "sub_code", "sub_category"]:
        x[col] = x[col].astype("category")

    return x


def make_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "id",
        "code",
        "sub_code",
        "sub_category",
        "horizon",
        "ts_index",
        "weight",
        "y_target",
        "__split__",
    }
    return [c for c in df.columns if c not in exclude]


def split_segments(
    train_df: pd.DataFrame, level1_threshold: int, meta_threshold: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_df = train_df.loc[train_df["ts_index"] <= level1_threshold].copy()
    meta_df = train_df.loc[
        (train_df["ts_index"] > level1_threshold) & (train_df["ts_index"] <= meta_threshold)
    ].copy()
    eval_df = train_df.loc[train_df["ts_index"] > meta_threshold].copy()
    if base_df.empty or meta_df.empty or eval_df.empty:
        raise ValueError(
            "Invalid thresholds: need non-empty base/meta/eval segments. "
            f"Got base={len(base_df)} meta={len(meta_df)} eval={len(eval_df)}"
        )
    return base_df, meta_df, eval_df


def split_inner_base(base_df: pd.DataFrame, inner_val_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_ts = np.sort(base_df["ts_index"].unique())
    if len(unique_ts) <= inner_val_size:
        raise ValueError(
            f"inner_val_size={inner_val_size} is too large for base segment with "
            f"{len(unique_ts)} unique timestamps"
        )
    cutoff_ts = unique_ts[-inner_val_size]
    inner_train = base_df.loc[base_df["ts_index"] < cutoff_ts].copy()
    inner_val = base_df.loc[base_df["ts_index"] >= cutoff_ts].copy()
    return inner_train, inner_val


def _to_lgb_categorical(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["code", "sub_code", "sub_category"]:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def _to_xgb_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["code", "sub_code", "sub_category"]:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def _train_lgb_seed(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    w_val: np.ndarray,
    *,
    seed: int,
    use_gpu: bool,
    n_estimators: int,
    early_stopping_rounds: int,
) -> tuple[lgb.LGBMRegressor, int]:
    params: dict[str, Any] = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.015,
        "n_estimators": n_estimators,
        "num_leaves": 80,
        "min_child_samples": 200,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 10.0,
        "verbosity": -1,
        "random_state": seed,
    }
    if use_gpu:
        params["device"] = "gpu"
    model = lgb.LGBMRegressor(**params)
    try:
        model.fit(
            _to_lgb_categorical(X_train),
            y_train,
            sample_weight=w_train,
            eval_set=[(_to_lgb_categorical(X_val), y_val)],
            eval_sample_weight=[w_val],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
    except Exception:
        params.pop("device", None)
        model = lgb.LGBMRegressor(**params)
        model.fit(
            _to_lgb_categorical(X_train),
            y_train,
            sample_weight=w_train,
            eval_set=[(_to_lgb_categorical(X_val), y_val)],
            eval_sample_weight=[w_val],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
    best_iter = int(getattr(model, "best_iteration_", params["n_estimators"]) or params["n_estimators"])
    return model, best_iter


def _refit_lgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    *,
    seed: int,
    rounds: int,
    use_gpu: bool,
) -> lgb.LGBMRegressor:
    params: dict[str, Any] = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.015,
        "n_estimators": rounds,
        "num_leaves": 80,
        "min_child_samples": 200,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 10.0,
        "verbosity": -1,
        "random_state": seed,
    }
    if use_gpu:
        params["device"] = "gpu"
    model = lgb.LGBMRegressor(**params)
    try:
        model.fit(_to_lgb_categorical(X_train), y_train, sample_weight=w_train)
    except Exception:
        params.pop("device", None)
        model = lgb.LGBMRegressor(**params)
        model.fit(_to_lgb_categorical(X_train), y_train, sample_weight=w_train)
    return model


def _train_xgb_seed(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    seed: int,
    use_gpu: bool,
    n_estimators: int,
    early_stopping_rounds: int,
    min_rounds: int,
) -> tuple[XGBRegressor, int]:
    params: dict[str, Any] = {
        "objective": "reg:squarederror",
        "learning_rate": 0.03,
        "n_estimators": n_estimators,
        "max_depth": 8,
        "min_child_weight": 24.0,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_lambda": 10.0,
        "reg_alpha": 0.2,
        "tree_method": "hist",
        "random_state": seed,
        "enable_categorical": True,
        "verbosity": 0,
        "early_stopping_rounds": early_stopping_rounds,
    }
    if use_gpu:
        params["device"] = "cuda"
    model = XGBRegressor(**params)
    try:
        model.fit(
            _to_xgb_frame(X_train),
            y_train,
            sample_weight=w_train,
            eval_set=[(_to_xgb_frame(X_val), y_val)],
            verbose=False,
        )
    except Exception:
        params.pop("device", None)
        model = XGBRegressor(**params)
        model.fit(
            _to_xgb_frame(X_train),
            y_train,
            sample_weight=w_train,
            eval_set=[(_to_xgb_frame(X_val), y_val)],
            verbose=False,
        )
    best_iter = int(getattr(model, "best_iteration", params["n_estimators"] - 1))
    best_rounds = max(min_rounds, best_iter + 1)
    return model, best_rounds


def _refit_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    *,
    seed: int,
    rounds: int,
    use_gpu: bool,
) -> XGBRegressor:
    params: dict[str, Any] = {
        "objective": "reg:squarederror",
        "learning_rate": 0.03,
        "n_estimators": rounds,
        "max_depth": 8,
        "min_child_weight": 24.0,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_lambda": 10.0,
        "reg_alpha": 0.2,
        "tree_method": "hist",
        "random_state": seed,
        "enable_categorical": True,
        "verbosity": 0,
    }
    if use_gpu:
        params["device"] = "cuda"
    model = XGBRegressor(**params)
    try:
        model.fit(_to_xgb_frame(X_train), y_train, sample_weight=w_train, verbose=False)
    except Exception:
        params.pop("device", None)
        model = XGBRegressor(**params)
        model.fit(_to_xgb_frame(X_train), y_train, sample_weight=w_train, verbose=False)
    return model


def train_level1_predictions(
    base_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    config: TwoLevelConfig,
) -> dict[str, Any]:
    inner_train, inner_val = split_inner_base(base_df, config.inner_val_size)

    X_inner_train = inner_train[feature_cols]
    y_inner_train = inner_train["y_target"].to_numpy(dtype=np.float32)
    w_inner_train = inner_train["weight"].to_numpy(dtype=np.float32)
    X_inner_val = inner_val[feature_cols]
    y_inner_val = inner_val["y_target"].to_numpy(dtype=np.float32)
    w_inner_val = inner_val["weight"].to_numpy(dtype=np.float32)

    X_base = base_df[feature_cols]
    y_base = base_df["y_target"].to_numpy(dtype=np.float32)
    w_base = base_df["weight"].to_numpy(dtype=np.float32)

    X_meta = meta_df[feature_cols]
    X_eval = eval_df[feature_cols]
    X_test = test_df[feature_cols]

    lgb_rounds: list[int] = []
    xgb_rounds: list[int] = []
    lgb_meta_preds: list[np.ndarray] = []
    lgb_eval_preds: list[np.ndarray] = []
    lgb_test_preds: list[np.ndarray] = []
    xgb_meta_preds: list[np.ndarray] = []
    xgb_eval_preds: list[np.ndarray] = []
    xgb_test_preds: list[np.ndarray] = []

    for seed in tqdm(config.lgb_seeds, desc="L1 LightGBM", disable=not config.show_progress, leave=False):
        _, best_iter = _train_lgb_seed(
            X_inner_train,
            y_inner_train,
            w_inner_train,
            X_inner_val,
            y_inner_val,
            w_inner_val,
            seed=seed,
            use_gpu=config.use_gpu,
            n_estimators=config.lgb_n_estimators,
            early_stopping_rounds=config.lgb_early_stopping_rounds,
        )
        lgb_rounds.append(best_iter)
        model = _refit_lgb(X_base, y_base, w_base, seed=seed, rounds=best_iter, use_gpu=config.use_gpu)
        lgb_meta_preds.append(model.predict(_to_lgb_categorical(X_meta)).astype(np.float64))
        lgb_eval_preds.append(model.predict(_to_lgb_categorical(X_eval)).astype(np.float64))
        lgb_test_preds.append(model.predict(_to_lgb_categorical(X_test)).astype(np.float64))

    for seed in tqdm(config.xgb_seeds, desc="L1 XGBoost", disable=not config.show_progress, leave=False):
        _, best_iter = _train_xgb_seed(
            X_inner_train,
            y_inner_train,
            w_inner_train,
            X_inner_val,
            y_inner_val,
            seed=seed,
            use_gpu=config.use_gpu,
            n_estimators=config.xgb_n_estimators,
            early_stopping_rounds=config.xgb_early_stopping_rounds,
            min_rounds=config.xgb_min_rounds,
        )
        xgb_rounds.append(best_iter)
        model = _refit_xgb(X_base, y_base, w_base, seed=seed, rounds=best_iter, use_gpu=config.use_gpu)
        xgb_meta_preds.append(model.predict(_to_xgb_frame(X_meta)).astype(np.float64))
        xgb_eval_preds.append(model.predict(_to_xgb_frame(X_eval)).astype(np.float64))
        xgb_test_preds.append(model.predict(_to_xgb_frame(X_test)).astype(np.float64))

    return {
        "lgb_meta": np.mean(np.column_stack(lgb_meta_preds), axis=1),
        "lgb_eval": np.mean(np.column_stack(lgb_eval_preds), axis=1),
        "lgb_test": np.mean(np.column_stack(lgb_test_preds), axis=1),
        "xgb_meta": np.mean(np.column_stack(xgb_meta_preds), axis=1),
        "xgb_eval": np.mean(np.column_stack(xgb_eval_preds), axis=1),
        "xgb_test": np.mean(np.column_stack(xgb_test_preds), axis=1),
        "lgb_rounds": lgb_rounds,
        "xgb_rounds": xgb_rounds,
    }


def build_level2_frame(
    lgb_pred: np.ndarray,
    xgb_pred: np.ndarray,
    ref_df: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pred_lgb": lgb_pred,
            "pred_xgb": xgb_pred,
            "pred_mean": (lgb_pred + xgb_pred) / 2.0,
            "pred_diff": lgb_pred - xgb_pred,
            "pred_abs_diff": np.abs(lgb_pred - xgb_pred),
            "pred_min": np.minimum(lgb_pred, xgb_pred),
            "pred_max": np.maximum(lgb_pred, xgb_pred),
            "ts_index": ref_df["ts_index"].to_numpy(dtype=np.float64),
            "sub_category_enc": ref_df.get("sub_category_enc", pd.Series(0.0, index=ref_df.index)).to_numpy(dtype=np.float64),
            "sub_code_enc": ref_df.get("sub_code_enc", pd.Series(0.0, index=ref_df.index)).to_numpy(dtype=np.float64),
            "t_cycle_sin": ref_df.get("t_cycle_sin", pd.Series(0.0, index=ref_df.index)).to_numpy(dtype=np.float64),
            "t_cycle_cos": ref_df.get("t_cycle_cos", pd.Series(0.0, index=ref_df.index)).to_numpy(dtype=np.float64),
        }
    )


def fit_meta_model(X_meta: pd.DataFrame, y_meta: np.ndarray, w_meta: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=2.0)),
        ]
    )
    model.fit(X_meta, y_meta, ridge__sample_weight=w_meta)
    return model


def compute_candidate_scores(
    y_true: np.ndarray,
    w_true: np.ndarray,
    lgb_pred: np.ndarray,
    xgb_pred: np.ndarray,
    meta_pred: np.ndarray,
) -> dict[str, float]:
    mean_pred = (lgb_pred + xgb_pred) / 2.0
    return {
        "lgb": float(weighted_rmse_score(y_true, lgb_pred, w_true)),
        "xgb": float(weighted_rmse_score(y_true, xgb_pred, w_true)),
        "mean": float(weighted_rmse_score(y_true, mean_pred, w_true)),
        "meta_ridge": float(weighted_rmse_score(y_true, meta_pred, w_true)),
    }


def refit_full_level1_test_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    lgb_rounds: list[int],
    xgb_rounds: list[int],
    config: TwoLevelConfig,
) -> dict[str, np.ndarray]:
    X_train = train_df[feature_cols]
    y_train = train_df["y_target"].to_numpy(dtype=np.float32)
    w_train = train_df["weight"].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols]

    lgb_preds: list[np.ndarray] = []
    xgb_preds: list[np.ndarray] = []
    for seed, rounds in zip(config.lgb_seeds, lgb_rounds):
        model = _refit_lgb(X_train, y_train, w_train, seed=seed, rounds=rounds, use_gpu=config.use_gpu)
        lgb_preds.append(model.predict(_to_lgb_categorical(X_test)).astype(np.float64))
    for seed, rounds in zip(config.xgb_seeds, xgb_rounds):
        model = _refit_xgb(X_train, y_train, w_train, seed=seed, rounds=rounds, use_gpu=config.use_gpu)
        xgb_preds.append(model.predict(_to_xgb_frame(X_test)).astype(np.float64))

    return {
        "lgb": np.mean(np.column_stack(lgb_preds), axis=1),
        "xgb": np.mean(np.column_stack(xgb_preds), axis=1),
    }


def run_two_level_pipeline(config: TwoLevelConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    submission_parts: list[pd.DataFrame] = []
    horizon_rows: list[dict[str, Any]] = []
    eval_y_all: list[np.ndarray] = []
    eval_w_all: list[np.ndarray] = []
    eval_pred_all: list[np.ndarray] = []

    for horizon in tqdm(VALID_HORIZONS, desc="Horizons", disable=not config.show_progress):
        train_df, test_df = load_horizon_frames(
            config.train_path,
            config.test_path,
            horizon,
            max_train_rows=config.max_train_rows_per_horizon,
            max_test_rows=config.max_test_rows_per_horizon,
        )
        base_df_raw, _, _ = split_segments(train_df, config.level1_threshold, config.meta_threshold)
        enc_stats = compute_encoding_stats(base_df_raw)

        train_marked = train_df.copy()
        train_marked["__split__"] = "train"
        test_marked = test_df.copy()
        test_marked["__split__"] = "test"
        combined = pd.concat([train_marked, test_marked], ignore_index=True)
        combined = build_context_features(combined, enc_stats)

        feat_train = combined.loc[combined["__split__"] == "train"].copy().reset_index(drop=True)
        feat_test = combined.loc[combined["__split__"] == "test"].copy().reset_index(drop=True)

        base_df, meta_df, eval_df = split_segments(
            feat_train, config.level1_threshold, config.meta_threshold
        )
        feature_cols = make_feature_columns(feat_train)

        level1 = train_level1_predictions(
            base_df=base_df,
            meta_df=meta_df,
            eval_df=eval_df,
            test_df=feat_test,
            feature_cols=feature_cols,
            config=config,
        )

        meta_X = build_level2_frame(level1["lgb_meta"], level1["xgb_meta"], meta_df)
        eval_X = build_level2_frame(level1["lgb_eval"], level1["xgb_eval"], eval_df)
        meta_y = meta_df["y_target"].to_numpy(dtype=np.float64)
        meta_w = meta_df["weight"].to_numpy(dtype=np.float64)
        eval_y = eval_df["y_target"].to_numpy(dtype=np.float64)
        eval_w = eval_df["weight"].to_numpy(dtype=np.float64)

        meta_model = fit_meta_model(meta_X, meta_y, meta_w)
        meta_eval_pred = meta_model.predict(eval_X).astype(np.float64)
        scores = compute_candidate_scores(
            eval_y,
            eval_w,
            level1["lgb_eval"],
            level1["xgb_eval"],
            meta_eval_pred,
        )
        selected_model = max(scores, key=scores.get)

        full_test_level1 = refit_full_level1_test_predictions(
            train_df=feat_train,
            test_df=feat_test,
            feature_cols=feature_cols,
            lgb_rounds=level1["lgb_rounds"],
            xgb_rounds=level1["xgb_rounds"],
            config=config,
        )
        full_test_meta_X = build_level2_frame(full_test_level1["lgb"], full_test_level1["xgb"], feat_test)
        full_test_meta_pred = meta_model.predict(full_test_meta_X).astype(np.float64)

        if selected_model == "lgb":
            final_test_pred = full_test_level1["lgb"]
            eval_selected = level1["lgb_eval"]
        elif selected_model == "xgb":
            final_test_pred = full_test_level1["xgb"]
            eval_selected = level1["xgb_eval"]
        elif selected_model == "mean":
            final_test_pred = (full_test_level1["lgb"] + full_test_level1["xgb"]) / 2.0
            eval_selected = (level1["lgb_eval"] + level1["xgb_eval"]) / 2.0
        else:
            final_test_pred = full_test_meta_pred
            eval_selected = meta_eval_pred

        submission_parts.append(
            pd.DataFrame({"id": feat_test["id"].to_numpy(), "prediction": final_test_pred})
        )
        eval_y_all.append(eval_y)
        eval_w_all.append(eval_w)
        eval_pred_all.append(eval_selected)

        horizon_rows.append(
            asdict(
                HorizonResult(
                    horizon=horizon,
                    selected_model=selected_model,
                    lgb_score=scores["lgb"],
                    xgb_score=scores["xgb"],
                    mean_score=scores["mean"],
                    meta_score=scores["meta_ridge"],
                    rows_meta=int(len(meta_df)),
                    rows_eval=int(len(eval_df)),
                    best_iterations_lgb=[int(x) for x in level1["lgb_rounds"]],
                    best_iterations_xgb=[int(x) for x in level1["xgb_rounds"]],
                )
            )
        )

    submission = pd.concat(submission_parts, ignore_index=True).sort_values("id").reset_index(drop=True)
    submission_path = config.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    eval_y = np.concatenate(eval_y_all)
    eval_w = np.concatenate(eval_w_all)
    eval_pred = np.concatenate(eval_pred_all)
    final_eval_score = float(weighted_rmse_score(eval_y, eval_pred, eval_w))

    results_df = pd.DataFrame(horizon_rows).sort_values("horizon").reset_index(drop=True)
    results_df.to_csv(config.output_dir / "validation_by_horizon.csv", index=False)

    summary = {
        "final_eval_score": final_eval_score,
        "submission_path": str(submission_path),
        "config": {
            **asdict(config),
            "train_path": str(config.train_path),
            "test_path": str(config.test_path),
            "output_dir": str(config.output_dir),
        },
        "horizon_results": horizon_rows,
    }
    (config.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return {
        "submission_path": submission_path,
        "results_df": results_df,
        "final_eval_score": final_eval_score,
        "summary": summary,
    }
