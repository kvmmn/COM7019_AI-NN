# Methodology

One-step-ahead forecast of **Adjusted Close** from daily OHLCV (single equity, 1980–2020).

## Protocol

1. Chronological 70 / 15 / 15 split (no shuffle)
2. Per-window normalisation (no global scaler leakage)
3. Persistence baseline before neural models
4. Validation-only model selection; test used once per run
5. Fixed seed `25199053`; identical loss, optimiser, and early stopping

## Windows

Trading days: **1W = 5**, **1M = 21** (primary), **1Y = 252**.

## Models

LSTM and GRU (required); CNN-LSTM hybrid (extension). Metrics: RMSE, MAE, MAPE in USD.
