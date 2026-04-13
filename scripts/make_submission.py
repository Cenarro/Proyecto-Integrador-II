"""Generate a competition submission from saved horizon models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.competition_pipeline import (
    DataSchema,
    detect_schema,
    build_prediction_frame,
    ensure_dir,
    load_pickle,
    load_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-path", type=Path, default=ROOT / "test.parquet")
    parser.add_argument(
        "--artifact-dir", type=Path, default=ROOT / "outputs" / "trained_models"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "outputs" / "submission" / "submission.csv",
    )
    return parser.parse_args()


def _load_manifest(artifact_dir: Path) -> dict[str, object]:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    manifest = _load_manifest(args.artifact_dir)
    schema = DataSchema(**manifest["schema"])

    artifacts = {}
    for horizon, info in manifest["artifacts"].items():
        payload = load_pickle(info["artifact_path"])
        artifacts[int(horizon)] = {
            "model": payload["model"],
            "prepared": payload["prepared"],
        }

    test_df = load_table(args.test_path)
    test_schema = detect_schema(test_df, require_target=False)
    if schema.id_col is None and test_schema.id_col is not None:
        schema = DataSchema(
            target_col=schema.target_col,
            weight_col=schema.weight_col,
            id_col=test_schema.id_col,
            horizon_col=schema.horizon_col,
            ts_col=schema.ts_col,
            numeric_feature_cols=schema.numeric_feature_cols,
            categorical_feature_cols=schema.categorical_feature_cols,
            feature_cols=schema.feature_cols,
        )
    submission = build_prediction_frame(test_df, schema, artifacts)

    ensure_dir(args.output_path.parent)
    submission.to_csv(args.output_path, index=False)
    print(f"submission_path={args.output_path}")
    print(f"submission_rows={len(submission)}")


if __name__ == "__main__":
    main()
