"""Data loading, EDA helpers, and windowing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET_COLUMN, TRAIN_FRAC, VAL_FRAC, TEST_FRAC


def load_stock_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def chronological_split_indices(n_rows: int) -> tuple[int, int]:
    train_end = int(n_rows * TRAIN_FRAC)
    val_end = int(n_rows * (TRAIN_FRAC + VAL_FRAC))
    return train_end, val_end


def build_windows(
    prices: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-window normalisation; returns X, y (relative), base (USD reference)."""
    x_list, y_list, base_list = [], [], []
    for i in range(len(prices) - window_size):
        window_prices = prices[i : i + window_size]
        last_price = window_prices[-1]
        x_list.append((window_prices / last_price) - 1.0)
        y_list.append((prices[i + window_size] / last_price) - 1.0)
        base_list.append(last_price)
    x = np.array(x_list, dtype="float32").reshape(-1, window_size, 1)
    y = np.array(y_list, dtype="float32")
    base = np.array(base_list, dtype="float64")
    return x, y, base


def assign_split_masks(
    n_samples: int,
    window_size: int,
    train_end: int,
    val_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask, val_mask, test_mask = [], [], []
    for i in range(n_samples):
        target_idx = window_size + i
        train_mask.append(target_idx < train_end)
        val_mask.append(train_end <= target_idx < val_end)
        test_mask.append(target_idx >= val_end)
    return np.array(train_mask), np.array(val_mask), np.array(test_mask)


def daily_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(prices) / prices[:-1]
