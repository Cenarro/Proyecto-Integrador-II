from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.two_level_stack import TwoLevelConfig, run_two_level_pipeline


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return tuple(int(v) for v in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a 2-level horizon-wise stack: "
            "level 1 = LightGBM + XGBoost, "
            "level 2 = ridge calibrator/blender, "
            "then write submission.csv."
        )
    )
    parser.add_argument("--train-path", type=Path, default=Path("src/data/train.parquet"))
    parser.add_argument("--test-path", type=Path, default=Path("src/data/test.parquet"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predictions/2-level_submission"),
    )
    parser.add_argument("--level1-threshold", type=int, default=3500)
    parser.add_argument("--meta-threshold", type=int, default=3550)
    parser.add_argument("--inner-val-size", type=int, default=120)
    parser.add_argument("--lgb-seeds", type=str, default="42,2024,12345,99,420")
    parser.add_argument("--xgb-seeds", type=str, default="42,2024,12345,99,420")
    parser.add_argument("--lgb-n-estimators", type=int, default=4200)
    parser.add_argument("--lgb-early-stopping-rounds", type=int, default=200)
    parser.add_argument("--xgb-n-estimators", type=int, default=2800)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=200)
    parser.add_argument("--xgb-min-rounds", type=int, default=50)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-train-rows-per-horizon",
        type=int,
        default=0,
        help="Optional development cap; 0 means full horizon data.",
    )
    parser.add_argument(
        "--max-test-rows-per-horizon",
        type=int,
        default=0,
        help="Optional development cap; 0 means full horizon data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TwoLevelConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        level1_threshold=args.level1_threshold,
        meta_threshold=args.meta_threshold,
        inner_val_size=args.inner_val_size,
        lgb_seeds=_parse_int_list(args.lgb_seeds),
        xgb_seeds=_parse_int_list(args.xgb_seeds),
        use_gpu=args.use_gpu,
        show_progress=args.show_progress,
        max_train_rows_per_horizon=args.max_train_rows_per_horizon,
        max_test_rows_per_horizon=args.max_test_rows_per_horizon,
        lgb_n_estimators=args.lgb_n_estimators,
        lgb_early_stopping_rounds=args.lgb_early_stopping_rounds,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_early_stopping_rounds=args.xgb_early_stopping_rounds,
        xgb_min_rounds=args.xgb_min_rounds,
    )

    outcome = run_two_level_pipeline(cfg)
    print(outcome["results_df"].to_string(index=False))
    print(f"\nfinal_eval_score={outcome['final_eval_score']:.8f}")
    print(f"submission_path={outcome['submission_path']}")


if __name__ == "__main__":
    main()
