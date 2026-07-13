"""Evaluation metrics and result helpers."""

from __future__ import annotations

import numpy as np


def relative_to_usd(relative: np.ndarray, base: np.ndarray) -> np.ndarray:
    return (relative + 1.0) * base


def regression_metrics(pred_usd: np.ndarray, true_usd: np.ndarray) -> dict[str, float]:
    errors = pred_usd - true_usd
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors / true_usd)) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def direction_accuracy(pred_usd: np.ndarray, true_usd: np.ndarray, prev_usd: np.ndarray) -> float:
    pred_dir = pred_usd > prev_usd
    true_dir = true_usd > prev_usd
    return float(np.mean(pred_dir == true_dir) * 100)
