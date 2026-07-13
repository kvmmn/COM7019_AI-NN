"""Project-wide constants and naming helpers."""

from __future__ import annotations

# Trading-day windows used in finance ML (NOT calendar days).
# 1W ≈ one trading week, 1M ≈ one trading month (~21 days), 1Y = 252 trading days/year.
WINDOWS: dict[str, dict[str, int | str]] = {
    "1W": {"label": "1Week", "trading_days": 5},
    "1M": {"label": "1Month", "trading_days": 21},
    "1Y": {"label": "1Year", "trading_days": 252},
}

PRIMARY_WINDOW_KEY = "1M"
SEED = 25199053
TARGET_COLUMN = "Adj Close"
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2


def build_model_name(
    architecture: str,
    layer_desc: str,
    units_desc: str,
    dropout: float | None,
    window_key: str,
    learning_rate: float | None = None,
    finetuned: bool = False,
) -> str:
    """Human-readable model name — every hyperparameter visible in the string."""
    win = WINDOWS[window_key]
    parts = [
        architecture,
        layer_desc,
        units_desc,
        f"Dropout{dropout}" if dropout is not None else "NoDropout",
        f"Window{win['label']}_{win['trading_days']}TradingDays",
    ]
    if learning_rate is not None and learning_rate != LEARNING_RATE:
        parts.append(f"LR{learning_rate:g}")
    if finetuned:
        parts.append("FineTuned")
    return "_".join(parts)
