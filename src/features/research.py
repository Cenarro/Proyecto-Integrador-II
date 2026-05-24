from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl

META_EXCLUDE_COLS = {"id", "y_target", "weight"}
SERIES_GROUP_COLS = ["code", "sub_code", "sub_category", "horizon"]


@dataclass
class FeatureArtifacts:
    numeric_cols: list[str]
    categorical_cols: list[str]
    fill_values: dict[str, float]
    category_values: dict[str, list]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureArtifacts":
        return cls(
            numeric_cols=list(payload["numeric_cols"]),
            categorical_cols=list(payload["categorical_cols"]),
            fill_values={k: float(v) for k, v in payload["fill_values"].items()},
            category_values={k: list(v) for k, v in payload["category_values"].items()},
        )


@dataclass
class ResearchFeatureConfig:
    raw_feature_cols: list[str]
    lag_feature_cols: list[str]
    cross_feature_cols: list[str]
    missing_indicator_raw_cols: list[str]
    lags: list[int]
    rolling_windows: list[int]
    ewm_spans: list[int]
    use_lag_block: bool = True
    use_hierarchy_block: bool = True
    use_cross_section_block: bool = True
    use_missing_indicators: bool = True
    use_temporal_imputation: bool = False
    use_stationarity_block: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchFeatureConfig":
        return cls(**payload)


@dataclass
class ResearchMatrixArtifacts:
    feature_order: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    category_values: dict[str, list]
    global_fill_values: dict[str, float]
    subcat_fill_values: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchMatrixArtifacts":
        return cls(
            feature_order=list(payload["feature_order"]),
            numeric_cols=list(payload["numeric_cols"]),
            categorical_cols=list(payload["categorical_cols"]),
            category_values={k: list(v) for k, v in payload["category_values"].items()},
            global_fill_values={k: float(v) for k, v in payload["global_fill_values"].items()},
            subcat_fill_values={
                col: {str(k): float(v) for k, v in per_cat.items()}
                for col, per_cat in payload["subcat_fill_values"].items()
            },
        )


def _median_map(df: pl.DataFrame, cols: list[str]) -> dict[str, float]:
    med = df.select([pl.col(c).median().alias(c) for c in cols]).to_dicts()[0]
    out: dict[str, float] = {}
    for k, v in med.items():
        if v is None:
            out[k] = 0.0
        else:
            out[k] = float(v)
    return out


def infer_numeric_feature_columns(df: pl.DataFrame) -> list[str]:
    cols = sorted(c for c in df.columns if c.startswith("feature_"))
    if "ts_index" in df.columns:
        cols.append("ts_index")
    return cols


def fit_feature_artifacts(
    train_df: pl.DataFrame,
    include_horizon_feature: bool,
) -> FeatureArtifacts:
    numeric_cols = infer_numeric_feature_columns(train_df)
    categorical_cols = ["code", "sub_code", "sub_category"]
    if include_horizon_feature:
        categorical_cols.append("horizon")

    fill_values = _median_map(train_df, numeric_cols)

    category_values = {
        col: sorted(
            train_df.select(pl.col(col).cast(pl.Utf8).drop_nulls().unique()).to_series().to_list()
        )
        for col in categorical_cols
    }

    return FeatureArtifacts(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        fill_values=fill_values,
        category_values=category_values,
    )


def transform_features(df: pl.DataFrame, artifacts: FeatureArtifacts) -> pl.DataFrame:
    cols = artifacts.numeric_cols + artifacts.categorical_cols
    X = df.select(cols)

    X = X.with_columns([pl.col(c).fill_null(artifacts.fill_values[c]).alias(c) for c in artifacts.numeric_cols])
    X = X.with_columns([pl.col(c).cast(pl.Utf8).alias(c) for c in artifacts.categorical_cols])
    return X


def prepare_features(
    df: pl.DataFrame,
    mode: str,
    stats: dict[str, Any] | None = None,
    include_horizon_feature: bool = True,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    mode = mode.lower()
    if mode == "fit":
        artifacts = fit_feature_artifacts(df, include_horizon_feature=include_horizon_feature)
        X = transform_features(df, artifacts)
        return X, artifacts.to_dict()

    if mode == "transform":
        if stats is None:
            raise ValueError("stats must be provided in transform mode")
        artifacts = FeatureArtifacts.from_dict(stats)
        X = transform_features(df, artifacts)
        return X, stats

    raise ValueError("mode must be one of {'fit', 'transform'}")


def infer_raw_feature_columns(df: pl.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("feature_"))


def select_research_feature_config(
    fit_df: pl.DataFrame,
    max_lag_cols: int = 10,
    max_cross_cols: int = 6,
    missing_indicator_threshold: float = 0.2,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    ewm_spans: list[int] | None = None,
    use_lag_block: bool = True,
    use_hierarchy_block: bool = True,
    use_cross_section_block: bool = True,
    use_missing_indicators: bool = True,
    use_temporal_imputation: bool = False,
    use_stationarity_block: bool = False,
) -> ResearchFeatureConfig:
    raw_cols = infer_raw_feature_columns(fit_df)
    if not raw_cols:
        raise ValueError("No feature_* columns found")

    miss_map = fit_df.select([(pl.col(c).is_null().mean()).alias(c) for c in raw_cols]).to_dicts()[0]
    var_map = fit_df.select([(pl.col(c).var()).alias(c) for c in raw_cols]).to_dicts()[0]

    stability = {
        c: float(var_map.get(c, 0.0) or 0.0) * max(0.0, 1.0 - float(miss_map.get(c, 0.0) or 0.0))
        for c in raw_cols
    }
    ranked = sorted(raw_cols, key=lambda c: stability[c], reverse=True)

    lag_cols = ranked[: max(1, min(max_lag_cols, len(ranked)))]
    cross_cols = lag_cols[: max(1, min(max_cross_cols, len(lag_cols)))]

    missing_cols = [c for c in raw_cols if float(miss_map.get(c, 0.0) or 0.0) >= missing_indicator_threshold]

    return ResearchFeatureConfig(
        raw_feature_cols=raw_cols,
        lag_feature_cols=lag_cols,
        cross_feature_cols=cross_cols,
        missing_indicator_raw_cols=missing_cols,
        lags=lags or [1, 2, 3, 5, 10, 20, 40],
        rolling_windows=rolling_windows or [5, 10, 20],
        ewm_spans=ewm_spans or [5, 10, 20],
        use_lag_block=use_lag_block,
        use_hierarchy_block=use_hierarchy_block,
        use_cross_section_block=use_cross_section_block,
        use_missing_indicators=use_missing_indicators,
        use_temporal_imputation=use_temporal_imputation,
        use_stationarity_block=use_stationarity_block,
    )


def engineer_research_features(
    df: pl.DataFrame,
    config: ResearchFeatureConfig,
) -> pl.DataFrame:
    df = df.sort(SERIES_GROUP_COLS + ["ts_index"])

    keep_cols = [
        c
        for c in ["id", "code", "sub_code", "sub_category", "horizon", "ts_index", "weight", "y_target"]
        if c in df.columns
    ]
    X = df.select(keep_cols + config.raw_feature_cols)

    if config.use_missing_indicators:
        X = X.with_columns(
            [pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_isna") for col in config.missing_indicator_raw_cols]
        )

    if config.use_temporal_imputation:
        # Fill internal gaps using local series history before global matrix-level fills.
        X = X.with_columns(
            [pl.col(col).forward_fill().over(SERIES_GROUP_COLS).alias(col) for col in config.raw_feature_cols]
        )
        X = X.with_columns(
            [pl.col(col).backward_fill().over(SERIES_GROUP_COLS).alias(col) for col in config.raw_feature_cols]
        )

    exprs: list[pl.Expr] = []

    if config.use_lag_block:
        for col in config.lag_feature_cols:
            lag1 = pl.col(col).shift(1).over(SERIES_GROUP_COLS)
            lag2 = pl.col(col).shift(2).over(SERIES_GROUP_COLS)

            exprs.extend([
                pl.col(col).shift(lag).over(SERIES_GROUP_COLS).alias(f"{col}_lag{lag}")
                for lag in config.lags
            ])

            if 1 in config.lags and 5 in config.lags:
                lag5 = pl.col(col).shift(5).over(SERIES_GROUP_COLS)
                exprs.append((lag1 - lag5).alias(f"{col}_mom_lag1_minus_lag5"))
                exprs.append((lag1 / (lag5.abs() + 1e-4)).alias(f"{col}_mom_lag1_div_lag5"))

            for window in config.rolling_windows:
                base = pl.col(col).shift(1).over(SERIES_GROUP_COLS)
                exprs.append(base.rolling_mean(window_size=window, min_samples=1).alias(f"{col}_rmean{window}"))
                exprs.append(base.rolling_std(window_size=window, min_samples=1).alias(f"{col}_rstd{window}"))

            for span in config.ewm_spans:
                exprs.append(
                    pl.col(col)
                    .shift(1)
                    .ewm_mean(span=span, adjust=False)
                    .over(SERIES_GROUP_COLS)
                    .alias(f"{col}_ewm{span}")
                )

            if config.use_stationarity_block:
                exprs.append((pl.col(col) - lag1).alias(f"{col}_diff1"))
                exprs.append((lag1 - lag2).alias(f"{col}_diff_prev"))
                exprs.append(((pl.col(col) - lag1) / (lag1.abs() + 1e-4)).alias(f"{col}_ret1"))

    for col in config.cross_feature_cols:
        base_series = pl.col(col)

        if config.use_hierarchy_block:
            subcat_mean = pl.col(col).mean().over(["horizon", "ts_index", "sub_category"])
            code_mean = pl.col(col).mean().over(["horizon", "ts_index", "code"])
            subcode_mean = pl.col(col).mean().over(["horizon", "ts_index", "sub_code"])

            exprs.append((base_series - subcat_mean).alias(f"{col}_rel_subcat_diff"))
            exprs.append((base_series / (subcat_mean.abs() + 1e-4)).alias(f"{col}_rel_subcat_ratio"))
            exprs.append((base_series - code_mean).alias(f"{col}_rel_code_diff"))
            exprs.append((base_series / (code_mean.abs() + 1e-4)).alias(f"{col}_rel_code_ratio"))
            exprs.append((base_series - subcode_mean).alias(f"{col}_rel_subcode_diff"))
            exprs.append((base_series / (subcode_mean.abs() + 1e-4)).alias(f"{col}_rel_subcode_ratio"))

        if config.use_cross_section_block:
            ts_group = ["horizon", "ts_index"]
            ts_mean = pl.col(col).mean().over(ts_group)
            ts_std = pl.col(col).std().over(ts_group)
            ts_rank_pct = pl.col(col).rank("average").over(ts_group) / pl.len().over(ts_group)

            exprs.append(((base_series - ts_mean) / (ts_std.fill_null(0.0) + 1e-4)).alias(f"{col}_ts_z"))
            exprs.append(ts_rank_pct.alias(f"{col}_ts_rank"))

    if exprs:
        X = X.with_columns(exprs)

    return X


def _build_subcat_fill_values(
    X: pl.DataFrame,
    numeric_cols: list[str],
) -> dict[str, dict[str, float]]:
    if "sub_category" not in X.columns:
        return {}

    subcat_values: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        grouped = X.group_by("sub_category").agg(pl.col(col).median().alias(col)).drop_nulls(col)
        subcat_values[col] = {
            str(row["sub_category"]): float(row[col])
            for row in grouped.to_dicts()
        }
    return subcat_values


def fit_research_matrix(feature_df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    X = feature_df
    drop_cols = [c for c in META_EXCLUDE_COLS if c in X.columns]
    if drop_cols:
        X = X.drop(drop_cols)

    categorical_cols = [c for c in ["code", "sub_code", "sub_category"] if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    global_fill = _median_map(X, numeric_cols)
    subcat_fill = _build_subcat_fill_values(X, numeric_cols) if "sub_category" in X.columns else {}

    fill_exprs: list[pl.Expr] = []
    if "sub_category" in X.columns:
        for col in numeric_cols:
            fill_exprs.append(
                pl.col(col)
                .fill_null(pl.col(col).median().over("sub_category"))
                .fill_null(global_fill[col])
                .alias(col)
            )
    else:
        fill_exprs.extend([pl.col(col).fill_null(global_fill[col]).alias(col) for col in numeric_cols])

    if fill_exprs:
        X = X.with_columns(fill_exprs)

    X = X.with_columns([pl.col(col).cast(pl.Float32).alias(col) for col in numeric_cols])

    category_values = {
        col: sorted(X.select(pl.col(col).cast(pl.Utf8).drop_nulls().unique()).to_series().to_list())
        for col in categorical_cols
    }
    if categorical_cols:
        X = X.with_columns([pl.col(col).cast(pl.Utf8).alias(col) for col in categorical_cols])

    feature_order = numeric_cols + categorical_cols
    artifacts = ResearchMatrixArtifacts(
        feature_order=feature_order,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        category_values=category_values,
        global_fill_values=global_fill,
        subcat_fill_values=subcat_fill,
    )
    return X.select(feature_order), artifacts.to_dict()


def transform_research_matrix(feature_df: pl.DataFrame, stats: dict[str, Any]) -> pl.DataFrame:
    artifacts = ResearchMatrixArtifacts.from_dict(stats)
    X = feature_df
    drop_cols = [c for c in META_EXCLUDE_COLS if c in X.columns]
    if drop_cols:
        X = X.drop(drop_cols)

    missing_cols = [c for c in artifacts.feature_order if c not in X.columns]
    if missing_cols:
        X = X.with_columns([pl.lit(None).alias(c) for c in missing_cols])

    fill_exprs: list[pl.Expr] = []
    if "sub_category" in X.columns:
        for col in artifacts.numeric_cols:
            mapping = artifacts.subcat_fill_values.get(col, {})
            subcat_fill_expr = pl.col("sub_category").cast(pl.Utf8).replace(mapping, default=None).cast(pl.Float64)
            fill_exprs.append(
                pl.col(col)
                .fill_null(subcat_fill_expr)
                .fill_null(artifacts.global_fill_values[col])
                .alias(col)
            )
    else:
        fill_exprs.extend(
            [pl.col(col).fill_null(artifacts.global_fill_values[col]).alias(col) for col in artifacts.numeric_cols]
        )

    if fill_exprs:
        X = X.with_columns(fill_exprs)

    X = X.with_columns([pl.col(col).cast(pl.Float32).alias(col) for col in artifacts.numeric_cols])

    if artifacts.categorical_cols:
        X = X.with_columns([pl.col(col).cast(pl.Utf8).alias(col) for col in artifacts.categorical_cols])

    return X.select(artifacts.feature_order)


def prepare_research_matrix(
    feature_df: pl.DataFrame,
    mode: str,
    stats: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    mode = mode.lower()
    if mode == "fit":
        X, artifacts = fit_research_matrix(feature_df)
        return X, artifacts
    if mode == "transform":
        if stats is None:
            raise ValueError("stats must be provided in transform mode")
        X = transform_research_matrix(feature_df, stats)
        return X, stats
    raise ValueError("mode must be one of {'fit', 'transform'}")
