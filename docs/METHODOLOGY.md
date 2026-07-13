# Methodology summary

## Problem

One-step-ahead forecasting of Adjusted Close from daily OHLCV for a single stock (1980–2020).

## Why financial data is hard to predict

- Daily returns are near a random walk (weak autocorrelation).
- Prices embed dividends, splits (Adj Close mitigates).
- Regime shifts (e.g. Mar 2020) break stationarity.
- Efficient-market view: exploitable patterns get arbitraged away.

## Protocol

1. Chronological 70/15/15 split (no shuffle).
2. Per-window normalisation (leakage-free vs global MinMax).
3. Persistence baseline before any NN.
4. Validation-only model selection; test touched once per run.
5. Fixed seed 25199053; identical optimiser/loss/early stopping.

## Windows

Trading days: 1W=5, 1M=21, 1Y=252. Primary experiments use 1M.

## Models

LSTM, GRU (required), CNN-LSTM hybrid (extension). Compared under identical protocol.

## Metrics

RMSE, MAE, MAPE (USD); direction accuracy (supplementary); regularisation heatmap.
