from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl


def _categorical_columns(X: pl.DataFrame) -> list[str]:
    return [col for col in X.columns if X.schema.get(col) == pl.Utf8]


def _to_pandas_for_lgbm(X: pl.DataFrame) -> pd.DataFrame:
    pdf = X.to_pandas(use_pyarrow_extension_array=False)
    for col in _categorical_columns(X):
        pdf[col] = pdf[col].astype("category")
    return pdf


def _build_params(base_params: dict[str, Any], device: str, random_seed: int) -> dict[str, Any]:
    params = dict(base_params)
    params["device"] = device
    params["seed"] = random_seed
    params["feature_fraction_seed"] = random_seed
    params["bagging_seed"] = random_seed
    params["data_random_seed"] = random_seed
    return params


def train_lgbm(
    X_train: pl.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: pl.DataFrame | None,
    y_val: np.ndarray | None,
    w_val: np.ndarray | None,
    base_params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
    use_gpu: bool,
    force_gpu: bool,
    random_seed: int,
) -> tuple[lgb.Booster, dict[str, Any]]:
    cat_cols = _categorical_columns(X_train)
    X_train_pd = _to_pandas_for_lgbm(X_train)
    train_set = lgb.Dataset(
        X_train_pd,
        label=y_train,
        weight=w_train,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks: list[Any] = []

    if X_val is not None and y_val is not None and w_val is not None:
        X_val_pd = _to_pandas_for_lgbm(X_val)
        valid_set = lgb.Dataset(
            X_val_pd,
            label=y_val,
            weight=w_val,
            categorical_feature=cat_cols,
            free_raw_data=False,
        )
        valid_sets.append(valid_set)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

    callbacks.append(lgb.log_evaluation(period=100))

    if force_gpu and not use_gpu:
        raise ValueError("force_gpu=True requires use_gpu=True")

    devices = ["gpu"] if force_gpu else (["gpu", "cpu"] if use_gpu else ["cpu"])
    last_error: Exception | None = None

    for device in devices:
        params = _build_params(base_params=base_params, device=device, random_seed=random_seed)
        try:
            booster = lgb.train(
                params=params,
                train_set=train_set,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )
            info = {
                "device": device,
                "best_iteration": int(booster.best_iteration or num_boost_round),
            }
            return booster, info
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc

    if last_error is None:
        raise RuntimeError("Unknown training error")
    if force_gpu:
        raise RuntimeError(
            f"LightGBM GPU training failed with --force-gpu enabled: {last_error}"
        ) from last_error
    raise RuntimeError(f"LightGBM training failed on all devices: {last_error}") from last_error


def predict_lgbm(model: lgb.Booster, X: pl.DataFrame) -> np.ndarray:
    num_iter = model.best_iteration if model.best_iteration else model.current_iteration()
    X_pd = _to_pandas_for_lgbm(X)
    return model.predict(X_pd, num_iteration=num_iter)


def choose_num_boost_round(best_iterations: list[int], default_rounds: int) -> int:
    if not best_iterations:
        return default_rounds
    rounded = int(np.median(best_iterations))
    return max(50, rounded)
