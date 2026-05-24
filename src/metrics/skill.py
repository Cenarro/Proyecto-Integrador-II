from __future__ import annotations

import numpy as np


def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))


def weighted_rmse_score(
    y_target: np.ndarray,
    y_pred: np.ndarray,
    w: np.ndarray,
) -> float:
    """Competition metric: sqrt(1 - clip01(weighted_mse / weighted_target_power))."""
    y_target = np.asarray(y_target, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    if y_target.shape != y_pred.shape or y_target.shape != w.shape:
        raise ValueError("y_target, y_pred, and w must have the same shape")

    denom = np.sum(w * np.square(y_target))
    if denom <= 0:
        return 0.0

    ratio = np.sum(w * np.square(y_target - y_pred)) / denom
    clipped = _clip01(float(ratio))
    val = 1.0 - clipped
    return float(np.sqrt(val))
