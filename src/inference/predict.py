from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from src.features.research import (
    ResearchFeatureConfig,
    engineer_research_features,
    prepare_features,
    prepare_research_matrix,
    select_research_feature_config,
)
from src.models.lightgbm import predict_lgbm, train_lgbm


def fit_global_full(
    train_df: pl.DataFrame,
    lgbm_params: dict[str, Any],
    num_boost_round: int,
    use_gpu: bool,
    force_gpu: bool,
    random_seed: int,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    X_train, stats = prepare_features(train_df, mode="fit", include_horizon_feature=True)

    model, info = train_lgbm(
        X_train=X_train,
        y_train=train_df.get_column("y_target").to_numpy(),
        w_train=train_df.get_column("weight").to_numpy(),
        X_val=None,
        y_val=None,
        w_val=None,
        base_params=lgbm_params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=0,
        use_gpu=use_gpu,
        force_gpu=force_gpu,
        random_seed=random_seed,
    )
    return model, stats, info


def fit_per_horizon_full(
    train_df: pl.DataFrame,
    lgbm_params: dict[str, Any],
    best_iterations: dict[str, int],
    use_gpu: bool,
    force_gpu: bool,
    random_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    models: dict[str, Any] = {}
    stats_by_h: dict[str, Any] = {}
    info_by_h: dict[str, Any] = {}

    for horizon in sorted(int(x) for x in train_df.get_column("horizon").unique().to_list()):
        horizon_key = str(int(horizon))
        part = train_df.filter(pl.col("horizon") == horizon)
        X_train, stats = prepare_features(part, mode="fit", include_horizon_feature=False)

        rounds = int(best_iterations.get(horizon_key, 500))
        model, info = train_lgbm(
            X_train=X_train,
            y_train=part.get_column("y_target").to_numpy(),
            w_train=part.get_column("weight").to_numpy(),
            X_val=None,
            y_val=None,
            w_val=None,
            base_params=lgbm_params,
            num_boost_round=rounds,
            early_stopping_rounds=0,
            use_gpu=use_gpu,
            force_gpu=force_gpu,
            random_seed=random_seed + int(horizon),
        )

        models[horizon_key] = model
        stats_by_h[horizon_key] = stats
        info_by_h[horizon_key] = info

    return models, stats_by_h, info_by_h


def predict_test_global(
    model: Any,
    stats: dict[str, Any],
    test_df: pl.DataFrame,
) -> np.ndarray:
    X_test, _ = prepare_features(test_df, mode="transform", stats=stats)
    return predict_lgbm(model, X_test)


def predict_test_per_horizon(
    models: dict[str, Any],
    stats_by_h: dict[str, Any],
    test_df: pl.DataFrame,
) -> np.ndarray:
    preds = np.zeros(test_df.height, dtype=np.float64)
    missing_horizons: list[int] = []

    test_indexed = test_df.with_row_index("__rowid__")

    for horizon in sorted(int(x) for x in test_df.get_column("horizon").unique().to_list()):
        horizon_key = str(int(horizon))
        if horizon_key not in models:
            missing_horizons.append(int(horizon))
            continue

        test_part = test_indexed.filter(pl.col("horizon") == horizon)
        X_test, _ = prepare_features(test_part, mode="transform", stats=stats_by_h[horizon_key])
        pred_part = predict_lgbm(models[horizon_key], X_test)
        rows = test_part.get_column("__rowid__").to_numpy()
        preds[rows] = pred_part

    if missing_horizons:
        raise RuntimeError(f"Missing models for horizons: {sorted(missing_horizons)}")

    return preds


def _objective_overrides(objective: str) -> dict[str, Any]:
    if objective == "regression":
        return {"objective": "regression", "metric": "rmse"}
    if objective == "huber":
        return {"objective": "huber", "metric": "rmse", "alpha": 0.9}
    if objective == "regression_l1":
        return {"objective": "regression_l1", "metric": "rmse"}
    raise ValueError(f"Unsupported objective: {objective}")


def fit_predict_per_horizon_research_ensemble(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    base_lgbm_params: dict[str, Any],
    objectives: list[str],
    seeds: list[int],
    best_iterations: dict[str, dict[str, int]],
    feature_config_by_horizon: dict[str, dict[str, Any]] | None,
    use_gpu: bool,
    force_gpu: bool,
    random_seed: int,
    max_lag_cols: int,
    max_cross_cols: int,
    missing_indicator_threshold: float,
    lags: list[int],
    rolling_windows: list[int],
    ewm_spans: list[int],
    use_lag_block: bool,
    use_hierarchy_block: bool,
    use_cross_section_block: bool,
    use_missing_indicators: bool,
    show_progress: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    preds = np.zeros(test_df.height, dtype=np.float64)
    train_info: dict[str, Any] = {}

    test_indexed = test_df.with_row_index("__rowid__")

    horizons = sorted(int(x) for x in test_df.get_column("horizon").unique().to_list())
    for horizon in tqdm(horizons, desc="Final-fit horizons", disable=not show_progress):
        horizon_key = str(int(horizon))
        h_train = train_df.filter(pl.col("horizon") == horizon)
        h_test = test_indexed.filter(pl.col("horizon") == horizon)

        if h_train.is_empty() or h_test.is_empty():
            continue

        if feature_config_by_horizon and horizon_key in feature_config_by_horizon:
            feature_cfg = ResearchFeatureConfig.from_dict(feature_config_by_horizon[horizon_key])
        else:
            feature_cfg = select_research_feature_config(
                h_train,
                max_lag_cols=max_lag_cols,
                max_cross_cols=max_cross_cols,
                missing_indicator_threshold=missing_indicator_threshold,
                lags=lags,
                rolling_windows=rolling_windows,
                ewm_spans=ewm_spans,
                use_lag_block=use_lag_block,
                use_hierarchy_block=use_hierarchy_block,
                use_cross_section_block=use_cross_section_block,
                use_missing_indicators=use_missing_indicators,
            )

        train_feat = engineer_research_features(h_train, config=feature_cfg)

        # Train has weight/y_target while test does not; diagonal concat aligns schemas safely.
        combo = pl.concat([h_train, h_test.drop("__rowid__")], how="diagonal_relaxed")
        combo_feat = engineer_research_features(combo, config=feature_cfg)
        test_ids = set(h_test.get_column("id").to_list())
        test_feat = combo_feat.filter(pl.col("id").is_in(test_ids))

        X_train, matrix_stats = prepare_research_matrix(train_feat, mode="fit")
        X_test, _ = prepare_research_matrix(test_feat, mode="transform", stats=matrix_stats)

        y_train = train_feat.get_column("y_target").to_numpy()
        w_train = train_feat.get_column("weight").to_numpy()

        model_preds: list[np.ndarray] = []
        model_rows: list[dict[str, Any]] = []
        for objective in objectives:
            obj_overrides = _objective_overrides(objective)
            for seed in tqdm(
                seeds,
                desc=f"h={horizon} {objective}",
                disable=not show_progress,
                leave=False,
            ):
                tag = f"{objective}_s{seed}"
                rounds = int(best_iterations.get(horizon_key, {}).get(tag, 250))

                params = dict(base_lgbm_params)
                params.update(obj_overrides)

                model, info = train_lgbm(
                    X_train=X_train,
                    y_train=y_train,
                    w_train=w_train,
                    X_val=None,
                    y_val=None,
                    w_val=None,
                    base_params=params,
                    num_boost_round=rounds,
                    early_stopping_rounds=0,
                    use_gpu=use_gpu,
                    force_gpu=force_gpu,
                    random_seed=random_seed + int(horizon) + int(seed),
                )

                model_preds.append(predict_lgbm(model, X_test))
                model_rows.append(
                    {
                        "tag": tag,
                        "rounds": rounds,
                        "device": str(info.get("device", "unknown")),
                        "best_iteration": int(info.get("best_iteration", rounds)),
                    }
                )

        if not model_preds:
            raise RuntimeError(f"No models trained for horizon {horizon}")

        ensemble_pred = np.mean(np.column_stack(model_preds), axis=1)
        pred_df = pl.DataFrame({"id": test_feat.get_column("id"), "__pred__": ensemble_pred})
        aligned = h_test.join(pred_df, on="id", how="left").get_column("__pred__")
        if aligned.null_count() > 0:
            raise RuntimeError(f"Missing predictions when aligning horizon {horizon}")

        rows = h_test.get_column("__rowid__").to_numpy()
        preds[rows] = aligned.to_numpy().astype(np.float64)

        train_info[horizon_key] = {
            "models": model_rows,
            "n_train_rows": int(h_train.height),
            "n_test_rows": int(h_test.height),
            "n_features": int(X_train.width),
            "feature_config": feature_cfg.to_dict(),
        }

    return preds, train_info


def write_submission(
    test_df: pl.DataFrame,
    predictions: np.ndarray,
    output_path: str | Path,
) -> pl.DataFrame:
    if len(predictions) != test_df.height:
        raise ValueError("Prediction length must match test_df")

    submission = pl.DataFrame(
        {
            "id": test_df.get_column("id"),
            "prediction": predictions.astype(np.float64),
        }
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(output_path)
    return submission
