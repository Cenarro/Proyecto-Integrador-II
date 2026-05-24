from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    train_path: Path = Path("src/data/train.parquet")
    test_path: Path = Path("src/data/test.parquet")
    figures_dir: Path = Path("outputs/figures")
    predictions_dir: Path = Path("outputs/predictions")


@dataclass
class SplitConfig:
    val_ratio: float = 0.15


@dataclass
class RuntimeConfig:
    random_seed: int = 42
    use_gpu: bool = True
    mode: str = "full"


@dataclass
class LGBMConfig:
    params: dict = field(
        default_factory=lambda: {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.03,
            "num_leaves": 96,
            "min_child_samples": 200,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.75,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 10.0,
            "max_bin": 255,
            "verbosity": -1,
        }
    )
    num_boost_round: int = 2_000
    early_stopping_rounds: int = 150
