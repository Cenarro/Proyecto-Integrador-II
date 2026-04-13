"""Train one model per horizon on the full training set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.competition_pipeline import (
    detect_schema,
    ensure_dir,
    fit_per_horizon_models,
    load_table,
    save_json,
    save_pickle,
    serialize_schema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=ROOT / "train.parquet")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "trained_models"
    )
    parser.add_argument(
        "--model", choices=("lgbm", "xgb", "catboost"), default="lgbm"
    )
    parser.add_argument("--num-boost-round", type=int, default=900)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    models_dir = ensure_dir(output_dir / "models")

    train_df = load_table(args.train_path)
    schema = detect_schema(train_df, require_target=True)

    artifacts = fit_per_horizon_models(
        train_df,
        schema,
        model_name=args.model,
        seed=args.seed,
        num_boost_round=args.num_boost_round,
        use_gpu=args.use_gpu,
        early_stopping_rounds=args.early_stopping_rounds,
        show_progress=args.show_progress,
    )

    model_manifest: dict[str, dict[str, object]] = {}
    for horizon, artifact in tqdm(
        sorted(artifacts.items()),
        desc="save horizons",
        disable=not args.show_progress,
    ):
        artifact_path = models_dir / f"h{horizon}_{args.model}.pkl"
        payload = {
            "model_name": args.model,
            "schema": serialize_schema(schema),
            "prepared": artifact["prepared"],
            "model": artifact["model"],
            "row_count": artifact["row_count"],
        }
        save_pickle(artifact_path, payload)
        model_manifest[str(horizon)] = {
            "artifact_path": str(artifact_path),
            "row_count": int(artifact["row_count"]),
        }

    save_json(
        output_dir / "manifest.json",
        {
            "train_path": str(args.train_path),
            "model": args.model,
            "num_boost_round": args.num_boost_round,
            "early_stopping_rounds": args.early_stopping_rounds,
            "seed": args.seed,
            "use_gpu": args.use_gpu,
            "schema": serialize_schema(schema),
            "artifacts": model_manifest,
        },
    )

    print(f"Saved {len(model_manifest)} horizon models to {models_dir}")


if __name__ == "__main__":
    main()
