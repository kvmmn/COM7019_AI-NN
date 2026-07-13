"""Plotting helpers for EDA and model comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import seaborn as sns


def plot_candlestick_sample(
    df: pd.DataFrame,
    start_idx: int,
    n_days: int,
    output_path: str | Path,
    title: str,
) -> None:
    """Candlestick view of OHLCV — shows financial data nature clearly."""
    sample = df.iloc[start_idx : start_idx + n_days].copy()
    sample = sample.set_index("Date")
    ohlc = sample[["Open", "High", "Low", "Close", "Volume"]]
    style = mpf.make_mpf_style(base_mpf_style="charles", gridstyle=":", y_on_right=False)
    mpf.plot(
        ohlc,
        type="candle",
        style=style,
        title=title,
        volume=True,
        savefig=dict(fname=str(output_path), dpi=150, bbox_inches="tight"),
    )


def plot_correlation_matrix(df: pd.DataFrame, cols: list[str], output_path: str | Path) -> None:
    corr = df[cols].corr()
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
    )
    plt.title("Feature correlation matrix (OHLCV + Adj Close)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_model_comparison_bars(results_df: pd.DataFrame, output_path: str | Path) -> None:
    """Grouped bar chart: RMSE / MAE / MAPE across all trained models."""
    plot_df = results_df[results_df["Model"] != "Baseline (persistence)"].copy()
    plot_df = plot_df.sort_values("Test RMSE").head(12)
    metrics = ["Test RMSE", "Test MAE", "Test MAPE %"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colours = plt.cm.tab20(np.linspace(0, 1, len(plot_df)))
    x = np.arange(len(plot_df))
    for ax, metric in zip(axes, metrics):
        ax.bar(x, plot_df[metric], color=colours)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["Model"], rotation=65, ha="right", fontsize=7)
    fig.suptitle("Figure 4: Model comparison (test set metrics)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_regularization_matrix(results_df: pd.DataFrame, output_path: str | Path) -> None:
    """Heatmap of test RMSE across architecture family and regularisation setting."""
    rows = []
    for _, row in results_df.iterrows():
        name = row["Model"]
        if name == "Baseline (persistence)":
            continue
        if name.startswith("LSTM"):
            family = "LSTM"
        elif name.startswith("GRU"):
            family = "GRU"
        elif name.startswith("Hybrid"):
            family = "Hybrid CNN-LSTM"
        else:
            continue
        if "FineTuned" in name:
            reg = "Fine-tuned LR"
        elif "NoDropout" in name:
            reg = "No dropout"
        elif "Dropout0.2" in name or "Dropout0.2" in name:
            reg = "Dropout 0.2"
        elif "LR0.0001" in name:
            reg = "LR 0.0001"
        else:
            reg = "Default"
        rows.append({"Family": family, "Regularisation": reg, "Test RMSE": row["Test RMSE"]})
    if not rows:
        return
    pivot = pd.DataFrame(rows).pivot_table(
        index="Family", columns="Regularisation", values="Test RMSE", aggfunc="min"
    )
    plt.figure(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu_r", linewidths=0.5)
    plt.title("Regularisation matrix: test RMSE (USD, lower is better)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_window_comparison(window_results: dict[str, float], output_path: str | Path) -> None:
    labels = list(window_results.keys())
    values = list(window_results.values())
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["steelblue", "seagreen", "darkorange"])
    plt.ylabel("Test RMSE (USD)")
    plt.title("Window length comparison (best GRU, no dropout)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
