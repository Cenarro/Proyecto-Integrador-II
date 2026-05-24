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
from src.training.config import LGBMConfig


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(',') if x.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(',') if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run full training and generate submission from fixed hyperparameters.')
    p.add_argument('--train-path', type=Path, default=Path('src/data/train.parquet'))
    p.add_argument('--test-path', type=Path, default=Path('src/data/test.parquet'))
    p.add_argument('--predictions-dir', type=Path, default=Path('outputs/predictions'))
    p.add_argument('--objectives', type=str, default='regression,huber')
    p.add_argument('--ensemble-seeds', type=str, default='42,143')
    p.add_argument('--num-boost-round', type=int, default=350)
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
    best_iterations = {
        str(h): {f'{obj}_s{seed}': int(args.num_boost_round) for obj in objectives for seed in seeds}
        for h in [1, 3, 10, 25]
    }
    cfg = LGBMConfig()
    base_lgbm_params = dict(cfg.params)
    preds, fit_info = fit_predict_per_horizon_research_ensemble(
        train_df=train_df,
        test_df=test_df,
        base_lgbm_params=base_lgbm_params,
        objectives=objectives,
        seeds=seeds,
        best_iterations=best_iterations,
        feature_config_by_horizon=None,
        use_gpu=args.use_gpu,
        force_gpu=args.force_gpu,
        random_seed=args.random_seed,
        max_lag_cols=args.max_lag_cols,
        max_cross_cols=args.max_cross_cols,
        missing_indicator_threshold=args.missing_indicator_threshold,
        lags=lags,
        rolling_windows=rolling_windows,
        ewm_spans=ewm_spans,
        use_lag_block=True,
        use_hierarchy_block=True,
        use_cross_section_block=True,
        use_missing_indicators=True,
        show_progress=args.progress,
    )
    submission_path = args.predictions_dir / 'submission_full.csv'
    write_submission(test_df, preds, submission_path)
    (args.predictions_dir / 'final_fit_info_full.json').write_text(json.dumps(fit_info, indent=2), encoding='utf-8')
    print(f'submission_path={submission_path}')


if __name__ == '__main__':
    main()
