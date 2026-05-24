from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.splits import load_data, validate_schema
from src.inference.predict import fit_predict_per_horizon_research_ensemble, write_submission
from src.metrics.skill import weighted_rmse_score
from src.training.config import LGBMConfig
from src.training.validate import run_feature_ablation, run_per_horizon_walk_forward_cv


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(',') if x.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(',') if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run walk-forward CV for the per-horizon LightGBM baseline.')
    p.add_argument('--train-path', type=Path, default=Path('src/data/train.parquet'))
    p.add_argument('--test-path', type=Path, default=Path('src/data/test.parquet'))
    p.add_argument('--predictions-dir', type=Path, default=Path('outputs/predictions'))
    p.add_argument('--cv-folds', type=int, default=3)
    p.add_argument('--cv-val-size', type=int, default=180)
    p.add_argument('--cv-min-train-size', type=int, default=900)
    p.add_argument('--objectives', type=str, default='regression,huber')
    p.add_argument('--ensemble-seeds', type=str, default='42,143')
    p.add_argument('--num-boost-round', type=int, default=1400)
    p.add_argument('--early-stopping-rounds', type=int, default=120)
    p.add_argument('--random-seed', type=int, default=42)
    p.add_argument('--use-gpu', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--force-gpu', action='store_true')
    p.add_argument('--progress', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max-lag-cols', type=int, default=10)
    p.add_argument('--max-cross-cols', type=int, default=6)
    p.add_argument('--missing-indicator-threshold', type=float, default=0.2)
    p.add_argument('--lags', type=str, default='1,2,3,5,10,20,40')
    p.add_argument('--rolling-windows', type=str, default='5,10,20')
    p.add_argument('--ewm-spans', type=str, default='5,10,20')
    p.add_argument('--disable-lag-block', action='store_true')
    p.add_argument('--disable-hierarchy-block', action='store_true')
    p.add_argument('--disable-cross-block', action='store_true')
    p.add_argument('--disable-missing-indicators', action='store_true')
    p.add_argument('--run-ablation', action='store_true')
    p.add_argument('--skip-final-fit', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_data(args.train_path, args.test_path)
    validate_schema(train_df, test_df)

    objectives = _parse_str_list(args.objectives)
    seeds = _parse_int_list(args.ensemble_seeds)
    lags = _parse_int_list(args.lags)
    rolling_windows = _parse_int_list(args.rolling_windows)
    ewm_spans = _parse_int_list(args.ewm_spans)

    lgbm_cfg = LGBMConfig()
    base_lgbm_params = dict(lgbm_cfg.params)
    base_lgbm_params.update({
        'learning_rate': 0.025,
        'num_leaves': 72,
        'min_child_samples': 260,
        'feature_fraction': 0.68,
        'bagging_fraction': 0.78,
        'bagging_freq': 5,
        'lambda_l1': 0.3,
        'lambda_l2': 18.0,
        'max_bin': 255,
        'verbosity': -1,
    })

    if args.run_ablation:
        ablation = run_feature_ablation(
            train_df=train_work,
            base_lgbm_params=base_lgbm_params,
            objectives=objectives,
            seeds=seeds,
            n_folds=args.cv_folds,
            val_size=args.cv_val_size,
            min_train_size=args.cv_min_train_size,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            use_gpu=args.use_gpu,
            force_gpu=args.force_gpu,
            random_seed=args.random_seed,
            max_lag_cols=args.max_lag_cols,
            max_cross_cols=args.max_cross_cols,
            missing_indicator_threshold=args.missing_indicator_threshold,
            lags=lags,
            rolling_windows=rolling_windows,
            ewm_spans=ewm_spans,
            show_progress=args.progress,
        )
        (args.predictions_dir / 'ablation_summary_full.json').write_text(json.dumps(ablation, indent=2), encoding='utf-8')

    outcome = run_per_horizon_walk_forward_cv(
        train_df=train_df,
        base_lgbm_params=base_lgbm_params,
        objectives=objectives,
        seeds=seeds,
        n_folds=args.cv_folds,
        val_size=args.cv_val_size,
        min_train_size=args.cv_min_train_size,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        use_gpu=args.use_gpu,
        force_gpu=args.force_gpu,
        random_seed=args.random_seed,
        max_lag_cols=args.max_lag_cols,
        max_cross_cols=args.max_cross_cols,
        missing_indicator_threshold=args.missing_indicator_threshold,
        lags=lags,
        rolling_windows=rolling_windows,
        ewm_spans=ewm_spans,
        use_lag_block=not args.disable_lag_block,
        use_hierarchy_block=not args.disable_hierarchy_block,
        use_cross_section_block=not args.disable_cross_block,
        use_missing_indicators=not args.disable_missing_indicators,
        show_progress=args.progress,
    )
    outcome.cv_cache.write_csv(args.predictions_dir / 'cv_cache_per_horizon_full.csv')
    summary = {
        'mode': 'full',
        'objectives': objectives,
        'seeds': seeds,
        'final_metric': outcome.final_metric,
        'score_by_horizon': {str(k): v for k, v in outcome.score_by_horizon.items()},
        'best_iterations': outcome.best_iterations,
        'device_by_tag': outcome.device_by_tag,
        'per_fold_metrics': outcome.per_fold_metrics,
        'feature_config_by_horizon': outcome.feature_config_by_horizon,
    }
    summary_path = args.predictions_dir / 'validation_summary_full.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"final_metric = weighted_rmse_score(cv_cache['y'], cv_cache['pred'], cv_cache['wt']) = {outcome.final_metric:.8f}")
    if args.skip_final_fit:
        return
    preds, fit_info = fit_predict_per_horizon_research_ensemble(
        train_df=train_df,
        test_df=test_df,
        base_lgbm_params=base_lgbm_params,
        objectives=objectives,
        seeds=seeds,
        best_iterations=outcome.best_iterations,
        feature_config_by_horizon=outcome.feature_config_by_horizon,
        use_gpu=args.use_gpu,
        force_gpu=args.force_gpu,
        random_seed=args.random_seed,
        max_lag_cols=args.max_lag_cols,
        max_cross_cols=args.max_cross_cols,
        missing_indicator_threshold=args.missing_indicator_threshold,
        lags=lags,
        rolling_windows=rolling_windows,
        ewm_spans=ewm_spans,
        use_lag_block=not args.disable_lag_block,
        use_hierarchy_block=not args.disable_hierarchy_block,
        use_cross_section_block=not args.disable_cross_block,
        use_missing_indicators=not args.disable_missing_indicators,
        show_progress=args.progress,
    )
    write_submission(test_df, preds, args.predictions_dir / 'submission_per_horizon_full.csv')
    (args.predictions_dir / 'final_fit_info_per_horizon_full.json').write_text(json.dumps(fit_info, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
