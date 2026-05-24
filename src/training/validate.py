from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from src.data.splits import build_time_split, build_walk_forward_folds
from src.features.research import (
    engineer_research_features,
    prepare_features,
    prepare_research_matrix,
    select_research_feature_config,
)
from src.metrics.skill import weighted_rmse_score
from src.models.lightgbm import predict_lgbm, train_lgbm


@dataclass
class ValidationOutcome:
    layout: str
    score_overall: float
    score_by_horizon: dict[int, float]
    split_ts: int
    best_iterations: dict[str, int]
    device_by_model: dict[str, str]
    oof_frame: pl.DataFrame
    models: dict[str, Any]
    feature_stats: dict[str, Any]


@dataclass
class ResearchCVOutcome:
    final_metric: float
    score_by_horizon: dict[int, float]
    cv_cache: pl.DataFrame
    per_fold_metrics: list[dict[str, Any]]
    best_iterations: dict[str, dict[str, int]]
    device_by_tag: dict[str, dict[str, str]]
    feature_config_by_horizon: dict[str, dict[str, Any]]
    model_tags: list[str]


def _score_by_horizon(frame: pl.DataFrame) -> dict[int, float]:
    out: dict[int, float] = {}
    for horizon in sorted(int(x) for x in frame.get_column("horizon").unique().to_list()):
        part = frame.filter(pl.col("horizon") == horizon)
        out[int(horizon)] = weighted_rmse_score(
            y_target=part.get_column("y_target").to_numpy(),
            y_pred=part.get_column("prediction").to_numpy(),
            w=part.get_column("weight").to_numpy(),
        )
    return out


def _objective_overrides(objective: str) -> dict[str, Any]:
    if objective == "regression":
        return {"objective": "regression", "metric": "rmse"}
    if objective == "huber":
        return {"objective": "huber", "metric": "rmse", "alpha": 0.9}
    if objective == "regression_l1":
        return {"objective": "regression_l1", "metric": "rmse"}
    raise ValueError(f"Unsupported objective: {objective}")


def _dominant_device(device_list: list[str]) -> str:
    if not device_list:
        return "unknown"
    values, counts = np.unique(np.array(device_list), return_counts=True)
    return str(values[int(np.argmax(counts))])


def run_global_validation(
    train_df: pl.DataFrame,
    val_ratio: float,
    lgbm_params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
    use_gpu: bool,
    force_gpu: bool,
    random_seed: int,
) -> ValidationOutcome:
    _, _, split_ts = build_time_split(train_df, val_ratio=val_ratio)

    fit_df = train_df.filter(pl.col("ts_index") < split_ts)
    val_df = train_df.filter(pl.col("ts_index") >= split_ts)

    X_fit, stats = prepare_features(fit_df, mode="fit", include_horizon_feature=True)
    X_val, _ = prepare_features(val_df, mode="transform", stats=stats)

    model, info = train_lgbm(
        X_train=X_fit,
        y_train=fit_df.get_column("y_target").to_numpy(),
        w_train=fit_df.get_column("weight").to_numpy(),
        X_val=X_val,
        y_val=val_df.get_column("y_target").to_numpy(),
        w_val=val_df.get_column("weight").to_numpy(),
        base_params=lgbm_params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        use_gpu=use_gpu,
        force_gpu=force_gpu,
        random_seed=random_seed,
    )

    preds = predict_lgbm(model, X_val)
    oof = val_df.select(["id", "horizon", "ts_index", "y_target", "weight"]).with_columns(
        pl.Series("prediction", preds)
    )

    overall = weighted_rmse_score(
        y_target=oof.get_column("y_target").to_numpy(),
        y_pred=oof.get_column("prediction").to_numpy(),
        w=oof.get_column("weight").to_numpy(),
    )

    return ValidationOutcome(
        layout="global",
        score_overall=overall,
        score_by_horizon=_score_by_horizon(oof),
        split_ts=split_ts,
        best_iterations={"global": int(info["best_iteration"])},
        device_by_model={"global": str(info["device"])},
        oof_frame=oof,
        models={"global": model},
        feature_stats={"global": stats},
    )


def run_per_horizon_validation(
    train_df: pl.DataFrame,
    val_ratio: float,
    lgbm_params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
    use_gpu: bool,
    force_gpu: bool,
    random_seed: int,
) -> ValidationOutcome:
    _, _, split_ts = build_time_split(train_df, val_ratio=val_ratio)

    oof_parts: list[pl.DataFrame] = []
    models: dict[str, Any] = {}
    stats_by_h: dict[str, Any] = {}
    best_iterations: dict[str, int] = {}
    devices: dict[str, str] = {}

    for horizon in sorted(int(x) for x in train_df.get_column("horizon").unique().to_list()):
        horizon_key = str(int(horizon))

        fit_df = train_df.filter((pl.col("horizon") == horizon) & (pl.col("ts_index") < split_ts))
        val_df = train_df.filter((pl.col("horizon") == horizon) & (pl.col("ts_index") >= split_ts))

        if fit_df.is_empty() or val_df.is_empty():
            continue

        X_fit, stats = prepare_features(
            fit_df,
            mode="fit",
            include_horizon_feature=False,
        )
        X_val, _ = prepare_features(val_df, mode="transform", stats=stats)

        model, info = train_lgbm(
            X_train=X_fit,
            y_train=fit_df.get_column("y_target").to_numpy(),
            w_train=fit_df.get_column("weight").to_numpy(),
            X_val=X_val,
            y_val=val_df.get_column("y_target").to_numpy(),
            w_val=val_df.get_column("weight").to_numpy(),
            base_params=lgbm_params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            use_gpu=use_gpu,
            force_gpu=force_gpu,
            random_seed=random_seed + int(horizon),
        )

        preds = predict_lgbm(model, X_val)
        oof_part = val_df.select(["id", "horizon", "ts_index", "y_target", "weight"]).with_columns(
            pl.Series("prediction", preds)
        )

        oof_parts.append(oof_part)
        models[horizon_key] = model
        stats_by_h[horizon_key] = stats
        best_iterations[horizon_key] = int(info["best_iteration"])
        devices[horizon_key] = str(info["device"])

    if not oof_parts:
        raise RuntimeError("Per-horizon validation produced no folds")

    oof = pl.concat(oof_parts, how="vertical_relaxed")
    overall = weighted_rmse_score(
        y_target=oof.get_column("y_target").to_numpy(),
        y_pred=oof.get_column("prediction").to_numpy(),
        w=oof.get_column("weight").to_numpy(),
    )

    return ValidationOutcome(
        layout="per_horizon",
        score_overall=overall,
        score_by_horizon=_score_by_horizon(oof),
        split_ts=split_ts,
        best_iterations=best_iterations,
        device_by_model=devices,
        oof_frame=oof,
        models=models,
        feature_stats=stats_by_h,
    )


def run_per_horizon_walk_forward_cv(
    train_df: pl.DataFrame,
    base_lgbm_params: dict[str, Any],
    objectives: list[str],
    seeds: list[int],
    n_folds: int,
    val_size: int,
    min_train_size: int,
    num_boost_round: int,
    early_stopping_rounds: int,
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
) -> ResearchCVOutcome:
    model_tags = [f"{obj}_s{seed}" for obj in objectives for seed in seeds]
    cv_parts: list[pl.DataFrame] = []
    per_fold_metrics: list[dict[str, Any]] = []
    feature_config_by_horizon: dict[str, dict[str, Any]] = {}

    best_iterations_raw: dict[str, dict[str, list[int]]] = {}
    device_raw: dict[str, dict[str, list[str]]] = {}

    horizons = sorted(int(x) for x in train_df.get_column("horizon").unique().to_list())
    for horizon in tqdm(horizons, desc="CV horizons", disable=not show_progress):
        horizon_key = str(int(horizon))
        horizon_df = train_df.filter(pl.col("horizon") == horizon)

        folds = build_walk_forward_folds(
            ts_index_values=horizon_df.get_column("ts_index").to_numpy(),
            n_folds=n_folds,
            val_size=val_size,
            min_train_size=min_train_size,
        )

        best_iterations_raw[horizon_key] = {tag: [] for tag in model_tags}
        device_raw[horizon_key] = {tag: [] for tag in model_tags}

        for fold_spec in tqdm(
            folds,
            desc=f"h={horizon} folds",
            disable=not show_progress,
            leave=False,
        ):
            fold = int(fold_spec["fold"])
            val_start_ts = int(fold_spec["val_start_ts"])
            val_end_ts = int(fold_spec["val_end_ts"])

            fit_df = horizon_df.filter(pl.col("ts_index") < val_start_ts)
            val_df = horizon_df.filter((pl.col("ts_index") >= val_start_ts) & (pl.col("ts_index") <= val_end_ts))

            if fit_df.is_empty() or val_df.is_empty():
                continue

            feature_cfg = select_research_feature_config(
                fit_df,
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
            feature_config_by_horizon[horizon_key] = feature_cfg.to_dict()

            combo = pl.concat([fit_df, val_df], how="vertical_relaxed")
            combo_feat = engineer_research_features(combo, config=feature_cfg)

            fit_feat = combo_feat.filter(pl.col("ts_index") < val_start_ts)
            val_feat = combo_feat.filter((pl.col("ts_index") >= val_start_ts) & (pl.col("ts_index") <= val_end_ts))

            X_fit, matrix_stats = prepare_research_matrix(fit_feat, mode="fit")
            X_val, _ = prepare_research_matrix(val_feat, mode="transform", stats=matrix_stats)

            y_fit = fit_feat.get_column("y_target").to_numpy()
            w_fit = fit_feat.get_column("weight").to_numpy()
            y_val = val_feat.get_column("y_target").to_numpy()
            w_val = val_feat.get_column("weight").to_numpy()

            fold_preds: list[np.ndarray] = []
            for objective in objectives:
                obj_overrides = _objective_overrides(objective)
                for seed in tqdm(
                    seeds,
                    desc=f"h={horizon} fold={fold} {objective}",
                    disable=not show_progress,
                    leave=False,
                ):
                    tag = f"{objective}_s{seed}"
                    params = dict(base_lgbm_params)
                    params.update(obj_overrides)

                    model, info = train_lgbm(
                        X_train=X_fit,
                        y_train=y_fit,
                        w_train=w_fit,
                        X_val=X_val,
                        y_val=y_val,
                        w_val=w_val,
                        base_params=params,
                        num_boost_round=num_boost_round,
                        early_stopping_rounds=early_stopping_rounds,
                        use_gpu=use_gpu,
                        force_gpu=force_gpu,
                        random_seed=random_seed + int(horizon) + int(seed),
                    )

                    fold_preds.append(predict_lgbm(model, X_val))
                    best_iterations_raw[horizon_key][tag].append(int(info["best_iteration"]))
                    device_raw[horizon_key][tag].append(str(info["device"]))

            if not fold_preds:
                continue

            ensemble_pred = np.mean(np.column_stack(fold_preds), axis=1)
            fold_cache = pl.DataFrame(
                {
                    "id": val_feat.get_column("id").to_list(),
                    "horizon": val_feat.get_column("horizon").to_list(),
                    "ts_index": val_feat.get_column("ts_index").to_list(),
                    "fold": [fold] * val_feat.height,
                    "y": y_val,
                    "pred": ensemble_pred,
                    "wt": w_val,
                }
            )
            cv_parts.append(fold_cache)

            fold_metric = weighted_rmse_score(y_target=y_val, y_pred=ensemble_pred, w=w_val)
            per_fold_metrics.append(
                {
                    "horizon": int(horizon),
                    "fold": fold,
                    "val_start_ts": val_start_ts,
                    "val_end_ts": val_end_ts,
                    "metric": float(fold_metric),
                    "rows": int(fold_cache.height),
                }
            )

    if not cv_parts:
        raise RuntimeError("No CV predictions were produced")

    cv_cache = pl.concat(cv_parts, how="vertical_relaxed")
    final_metric = weighted_rmse_score(
        y_target=cv_cache.get_column("y").to_numpy(),
        y_pred=cv_cache.get_column("pred").to_numpy(),
        w=cv_cache.get_column("wt").to_numpy(),
    )

    score_by_horizon: dict[int, float] = {}
    for horizon in sorted(int(x) for x in cv_cache.get_column("horizon").unique().to_list()):
        part = cv_cache.filter(pl.col("horizon") == horizon)
        score_by_horizon[int(horizon)] = weighted_rmse_score(
            y_target=part.get_column("y").to_numpy(),
            y_pred=part.get_column("pred").to_numpy(),
            w=part.get_column("wt").to_numpy(),
        )

    best_iterations: dict[str, dict[str, int]] = {}
    device_by_tag: dict[str, dict[str, str]] = {}

    for horizon_key, per_tag in best_iterations_raw.items():
        best_iterations[horizon_key] = {}
        device_by_tag[horizon_key] = {}
        for tag, values in per_tag.items():
            if values:
                best_iterations[horizon_key][tag] = int(np.median(values))
            else:
                best_iterations[horizon_key][tag] = int(max(50, num_boost_round // 4))
            device_by_tag[horizon_key][tag] = _dominant_device(device_raw[horizon_key][tag])

    return ResearchCVOutcome(
        final_metric=float(final_metric),
        score_by_horizon=score_by_horizon,
        cv_cache=cv_cache,
        per_fold_metrics=per_fold_metrics,
        best_iterations=best_iterations,
        device_by_tag=device_by_tag,
        feature_config_by_horizon=feature_config_by_horizon,
        model_tags=model_tags,
    )


def run_feature_ablation(
    train_df: pl.DataFrame,
    base_lgbm_params: dict[str, Any],
    objectives: list[str],
    seeds: list[int],
    n_folds: int,
    val_size: int,
    min_train_size: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    use_gpu: bool,
    force_gpu: bool,
    random_seed: int,
    max_lag_cols: int,
    max_cross_cols: int,
    missing_indicator_threshold: float,
    lags: list[int],
    rolling_windows: list[int],
    ewm_spans: list[int],
    show_progress: bool = True,
) -> dict[str, Any]:
    blocks = [
        (
            "base_only",
            {
                "use_lag_block": False,
                "use_hierarchy_block": False,
                "use_cross_section_block": False,
                "use_missing_indicators": False,
            },
        ),
        (
            "base_plus_lag",
            {
                "use_lag_block": True,
                "use_hierarchy_block": False,
                "use_cross_section_block": False,
                "use_missing_indicators": False,
            },
        ),
        (
            "base_lag_hierarchy",
            {
                "use_lag_block": True,
                "use_hierarchy_block": True,
                "use_cross_section_block": False,
                "use_missing_indicators": False,
            },
        ),
        (
            "full_stack",
            {
                "use_lag_block": True,
                "use_hierarchy_block": True,
                "use_cross_section_block": True,
                "use_missing_indicators": True,
            },
        ),
    ]

    results: dict[str, Any] = {}
    for name, flags in tqdm(blocks, desc="Ablation blocks", disable=not show_progress):
        outcome = run_per_horizon_walk_forward_cv(
            train_df=train_df,
            base_lgbm_params=base_lgbm_params,
            objectives=objectives,
            seeds=seeds,
            n_folds=n_folds,
            val_size=val_size,
            min_train_size=min_train_size,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            use_gpu=use_gpu,
            force_gpu=force_gpu,
            random_seed=random_seed,
            max_lag_cols=max_lag_cols,
            max_cross_cols=max_cross_cols,
            missing_indicator_threshold=missing_indicator_threshold,
            lags=lags,
            rolling_windows=rolling_windows,
            ewm_spans=ewm_spans,
            use_lag_block=flags["use_lag_block"],
            use_hierarchy_block=flags["use_hierarchy_block"],
            use_cross_section_block=flags["use_cross_section_block"],
            use_missing_indicators=flags["use_missing_indicators"],
            show_progress=show_progress,
        )
        results[name] = {
            "final_metric": outcome.final_metric,
            "score_by_horizon": {str(k): v for k, v in outcome.score_by_horizon.items()},
        }

    return results
