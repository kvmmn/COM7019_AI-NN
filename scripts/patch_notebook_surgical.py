#!/usr/bin/env python3
"""Surgical tutor-feedback patches on the GOOD notebook (preserve diagrams + comments)."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "_code" / "COM7019_25199053.ipynb"
BAD_ARCHIVE = ROOT / "_code" / "archive" / "COM7019_25199053_bad_rewrite.ipynb"


def src_join(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_src(cell: dict, text: str) -> None:
    lines = text.split("\n")
    cell["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])


def clear_outputs(nb: dict) -> None:
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None


def insert_cell(nb: dict, idx: int, cell: dict) -> None:
    nb["cells"].insert(idx, cell)


CANDLESTICK_CELL = {
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [],
}

CANDLESTICK_SOURCE = r'''# ============================================================
# STEP 3b: Exploratory plots — candlestick charts (financial EDA)
# ============================================================
# Line charts of Close alone hide the daily range (High/Low) and where
# the price opened vs closed. For OHLCV stock data the standard view is
# a candlestick chart: each day is one candle (body = Open→Close, wicks =
# High/Low). I use mplfinance, which is built for this and expects exactly
# our column names (Open, High, Low, Close, Volume) with Date as index.
#
# I plot two windows:
#   1) The last 60 trading days (recent behaviour, readable zoom)
#   2) The test-period start around early 2019 (includes the run-up and
#      the March 2020 crash — a stress-test slice for the report)
#
# mav=(5, 21) overlays 5- and 21-day simple moving averages (1 week and
# 1 month in TRADING days), matching the window sizes I use later.

import mplfinance as mpf

def plot_candlestick_slice(start_idx, n_days, filename, title):
    """Save a candlestick + volume chart for a date slice of df."""
    ohlcv = df.iloc[start_idx : start_idx + n_days].copy()
    ohlcv = ohlcv.set_index("Date")
    # mplfinance requires these exact column names (case-sensitive)
    ohlcv = ohlcv[["Open", "High", "Low", "Close", "Volume"]]
    style = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle=":", y_on_right=False)
    save_path = OUTPUT_FOLDER + "/figures/" + filename
    mpf.plot(
        ohlcv,
        type="candle",
        style=style,
        title=title,
        volume=True,
        mav=(5, 21),
        figsize=(12, 7),
        savefig=dict(fname=save_path, dpi=150, bbox_inches="tight"),
    )
    # Also display inline in the notebook (Colab report screenshots)
    mpf.plot(
        ohlcv,
        type="candle",
        style=style,
        title=title,
        volume=True,
        mav=(5, 21),
        figsize=(12, 7),
    )
    print("Saved:", save_path)

# Last 60 trading days of the full series
plot_candlestick_slice(
    len(df) - 60,
    60,
    "eda_candlestick_last60days.png",
    "EDA Figure A: Last 60 trading days (candlestick + 5/21-day MAs)",
)

# Test period begins near index val_end (set in STEP 4); use a 90-day slice
# from early 2019 so the chart shows pre-crash and crash candles clearly.
idx_2019 = int(df[df["Date"] >= "2019-01-01"].index[0])
plot_candlestick_slice(
    idx_2019,
    90,
    "eda_candlestick_2019_2020.png",
    "EDA Figure B: 2019–2020 (90 trading days, includes Mar 2020 volatility)",
)
'''

COMPARE_CELL_SOURCE = r'''# ============================================================
# STEP 20b: Graphical comparison of all models (test metrics)
# ============================================================
# The results table in STEP 20 is exact but dense. This bar-chart view
# makes it easier to see how close every model sits to the baseline and
# which configuration tweaks (dropout, window, LR) moved the needle.
# I plot the top 12 non-baseline runs by test RMSE.

import matplotlib.pyplot as plt
import numpy as np

plot_df = results_df[results_df["Model"] != "Baseline (persistence)"].copy()
plot_df = plot_df.sort_values("Test RMSE").head(12)
metrics = ["Test RMSE", "Test MAE", "Test MAPE %"]
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
x = np.arange(len(plot_df))
colours = plt.cm.tab20(np.linspace(0, 1, len(plot_df)))
for ax, metric in zip(axes, metrics):
    ax.bar(x, plot_df[metric], color=colours)
    ax.set_title(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Model"], rotation=70, ha="right", fontsize=6)
fig.suptitle("Figure 4: Model comparison (test set)")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/figures/fig4_model_comparison_bars.png", dpi=150)
plt.show()
'''

# Model name mapping: old -> new (Keras-safe, readable, no + or spaces)
NAME_MAP = {
    "LSTM_64_d02": "LSTM_1Layer_64Units_Dropout0p2_Win1Month_21d",
    "LSTM_64x32_d02": "LSTM_2Layer_64-32Units_Dropout0p2_Win1Month_21d",
    "GRU_64_d02": "GRU_1Layer_64Units_Dropout0p2_Win1Month_21d",
    "GRU_64x32_d02": "GRU_2Layer_64-32Units_Dropout0p2_Win1Month_21d",
    "LSTM_64_d0": "LSTM_1Layer_64Units_NoDropout_Win1Month_21d",
    "LSTM_64x32_d0": "LSTM_2Layer_64-32Units_NoDropout_Win1Month_21d",
    "GRU_64_d0": "GRU_1Layer_64Units_NoDropout_Win1Month_21d",
    "GRU_64x32_d0": "GRU_2Layer_64-32Units_NoDropout_Win1Month_21d",
    "GRU_64_d0_w30": "GRU_1Layer_64Units_NoDropout_Win1Week_5d",  # replaced by window ablation
    "GRU_64_d0_lr1e-4": "GRU_1Layer_64Units_NoDropout_LR0p0001_Win1Month_21d",
    "CNN32k3_LSTM64": "Hybrid_CNN32_LSTM64_Dropout0p2_Win1Month_21d",
    "CNN32k3_LSTM64_finetuned": "Hybrid_CNN32_LSTM64_Dropout0p2_Win1Month_21d_FineTuned",
}

WINDOW_ABLATION_SOURCE = r'''# ============================================================
# STEP 16: Window-length ablation — 1W / 1M / 1Y (trading days)
# ============================================================
# Finance ML commonly uses trading-day lookbacks, not calendar days:
#   1W = 5 days, 1M = 21 days, 1Y = 252 days (~52 weeks).
# Our main experiments above use 1M (21). Here I rebuild the windows
# for all three horizons using the same GRU architecture (64 units,
# no dropout — the winner's ablation setup) so the ONLY change is
# how much history each prediction sees.
#
# I do NOT expect large differences on one-step-ahead price forecasting
# (the literature often finds window length matters less than claimed
# for daily returns), but using the standard horizons keeps the study
# defensible to a finance audience.

WINDOW_ABLATION = {
    "1W_5d": 5,
    "1M_21d": 21,
    "1Y_252d": 252,
}

window_ablation_rmse = {}

for tag, wsize in WINDOW_ABLATION.items():
    # --- Rebuild windows (same logic as STEP 5, different length) ---
    Xw_list, yw_list, basew_list = [], [], []
    for i in range(len(prices) - wsize):
        window_prices = prices[i : i + wsize]
        last_price = window_prices[-1]
        Xw_list.append((window_prices / last_price) - 1.0)
        yw_list.append((prices[i + wsize] / last_price) - 1.0)
        basew_list.append(last_price)
    Xw = np.array(Xw_list, dtype="float32").reshape(-1, wsize, 1)
    yw = np.array(yw_list, dtype="float32")
    basew = np.array(basew_list, dtype="float64")

    tr_m, va_m, te_m = [], [], []
    for i in range(len(yw)):
        tidx = wsize + i
        tr_m.append(tidx < train_end)
        va_m.append(train_end <= tidx < val_end)
        te_m.append(tidx >= val_end)
    tr_m, va_m, te_m = np.array(tr_m), np.array(va_m), np.array(te_m)

    X_tr_w, y_tr_w = Xw[tr_m], yw[tr_m]
    X_va_w, y_va_w = Xw[va_m], yw[va_m]
    X_te_w, y_te_w, base_te_w = Xw[te_m], yw[te_m], basew[te_m]

    run_name = f"GRU_1Layer_64Units_NoDropout_Win{tag}"
    print(f"\n--- Window ablation: {tag} ({wsize} trading days) ---")
    print("  train/val/test samples:", len(y_tr_w), len(y_va_w), len(y_te_w))

    m = keras.Sequential(name=run_name)
    m.add(keras.layers.Input(shape=(wsize, 1)))
    m.add(keras.layers.GRU(64))
    m.add(keras.layers.Dense(1))
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])

    m, elapsed = train_model(m, run_name, X_tr_w, y_tr_w, X_va_w, y_va_w)
    rmse = evaluate_model(m, run_name, elapsed, X_te_w, y_te_w, base_te_w)
    window_ablation_rmse[tag] = rmse

# Bar chart: does horizon length change test RMSE?
plt.figure(figsize=(7, 4))
tags = list(window_ablation_rmse.keys())
vals = list(window_ablation_rmse.values())
plt.bar(tags, vals, color=["steelblue", "seagreen", "darkorange"])
plt.axhline(baseline_rmse, color="gray", linestyle="--", label=f"Baseline RMSE {baseline_rmse:.3f}")
plt.ylabel("Test RMSE (USD)")
plt.title("Figure 5: Window ablation (GRU, no dropout)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/figures/fig5_window_ablation.png", dpi=150)
plt.show()
'''


def patch_step1(text: str) -> str:
    if "mplfinance" not in text:
        text = text.replace(
            "from tensorflow import keras\n",
            "import mplfinance as mpf  # candlestick EDA (STEP 3b); standard for OHLCV data\n"
            "from tensorflow import keras\n",
        )
    return text


def patch_step3(text: str) -> str:
    extra = '''
# --- Financial data characteristics (why this problem is hard) ---
# Stock prices are noisy and nearly unpredictable at daily frequency:
# returns have weak autocorrelation, so "tomorrow ≈ today" is strong.
# Any model must be judged against that baseline, not against a perfect fit.
daily_returns = np.diff(prices) / prices[:-1]
print("\\nDaily return mean:", round(daily_returns.mean(), 6))
print("Daily return std :", round(daily_returns.std(), 4))
print("Lag-1 autocorrelation:", round(float(np.corrcoef(daily_returns[:-1], daily_returns[1:])[0, 1]), 4))
'''
    if "Lag-1 autocorrelation" not in text:
        text = text.rstrip() + extra
    return text


def patch_markdown_windows(text: str) -> str:
    return text.replace(
        "- I use a **60-day window** as the default (roughly one quarter of trading days). I also test a 30-day window later as an experiment.",
        "- I use **trading-day** windows (finance convention, not calendar days): **1W = 5**, **1M = 21**, **1Y = 252**. All main models use **1M (21 days)** as the primary lookback; STEP 16 ablates all three horizons on the same GRU.",
    )


def patch_step5(text: str) -> str:
    text = text.replace("WINDOW_SIZE = 60  # number of past days used as input", "WINDOW_SIZE = 21  # 1M = 21 trading days (primary window)")
    text = text.replace("One sample = 60 consecutive days", "One sample = 21 consecutive trading days (1 month)")
    text = text.replace("days 0..59", "days 0..20")
    text = text.replace("day 60", "day 21")
    return text


def patch_step9_comments(text: str) -> str:
    return text.replace("one window of 60 days", "one window of 21 trading days (1M)")


def patch_hybrid_comment(text: str) -> str:
    text = text.replace("Name decoded: CNN with 32 filters", "Architecture: CNN with 32 filters")
    text = text.replace("   (60 days, 1 feature)", "   (21 days, 1 feature)")
    text = text.replace("     → Conv1D        → (60 days, 32 features)", "     → Conv1D        → (21 days, 32 features)")
    text = text.replace("     → MaxPooling1D  → (30 days, 32 features)", "     → MaxPooling1D  → (10 days, 32 features)")
    return text


def apply_name_map(text: str) -> str:
    for old, new in sorted(NAME_MAP.items(), key=lambda x: -len(x[0])):
        text = text.replace(f'"{old}"', f'"{new}"')
        text = text.replace(f"'{old}'", f"'{new}'")
        text = text.replace(old, new)
    # Fix pick_best conditionals for LSTM no-dropout branch
    text = text.replace(
        'if best_lstm_name == "LSTM_1Layer_64Units_Dropout0p2_Win1Month_21d":',
        'if best_lstm_name == NAME_LSTM_1L:',
    )
    text = text.replace(
        'if best_gru_name == "GRU_1Layer_64Units_Dropout0p2_Win1Month_21d":',
        'if best_gru_name == NAME_GRU_1L:',
    )
    return text


def patch_step13_add_constants(text: str) -> str:
    header = '''# Readable Keras-safe model names (tutor feedback: names must explain the setup)
NAME_LSTM_1L = "LSTM_1Layer_64Units_Dropout0p2_Win1Month_21d"
NAME_LSTM_2L = "LSTM_2Layer_64-32Units_Dropout0p2_Win1Month_21d"
NAME_GRU_1L = "GRU_1Layer_64Units_Dropout0p2_Win1Month_21d"
NAME_GRU_2L = "GRU_2Layer_64-32Units_Dropout0p2_Win1Month_21d"

'''
    if "NAME_LSTM_1L" not in text:
        text = text.replace(
            "lstm_models = {",
            header + "lstm_models = {",
        )
    return text


def patch_models_to_plot(text: str) -> str:
    text = text.replace(
        'models_to_plot = ["LSTM_64x32_d02", "GRU_64_d02", "CNN32k3_LSTM64"]',
        'models_to_plot = [NAME_LSTM_2L, NAME_GRU_1L, "Hybrid_CNN32_LSTM64_Dropout0p2_Win1Month_21d"]',
    )
    text = text.replace('"CNN32k3_LSTM64"', '"Hybrid_CNN32_LSTM64_Dropout0p2_Win1Month_21d"')
    text = text.replace('"Extension:", "CNN32k3_LSTM64"', '"Extension:", "Hybrid_CNN32_LSTM64_Dropout0p2_Win1Month_21d"')
    return text


def patch_step14_15_conditionals(text: str) -> str:
    text = text.replace('if best_lstm_name == "LSTM_1Layer_64Units_Dropout0p2_Win1Month_21d":', "if best_lstm_name == NAME_LSTM_1L:")
    text = text.replace('if best_lstm_name == "LSTM_64_d02":', "if best_lstm_name == NAME_LSTM_1L:")
    text = text.replace('if best_gru_name == "GRU_1Layer_64Units_Dropout0p2_Win1Month_21d":', "if best_gru_name == NAME_GRU_1L:")
    text = text.replace('if best_gru_name == "GRU_64_d02":', "if best_gru_name == NAME_GRU_1L:")
    return text


def main() -> None:
    nb = json.loads(NB_PATH.read_text())
    new_cells = []

    for cell in nb["cells"]:
        src = src_join(cell)

        if cell["cell_type"] == "code" and "import shutil" in src:
            continue  # drop Colab-only zip cell

        if cell["cell_type"] == "code":
            if "STEP 1:" in src:
                src = patch_step1(src)
            elif "STEP 3:" in src and "STEP 3b" not in src:
                src = patch_step3(src)
            elif "STEP 5:" in src:
                src = patch_step5(src)
            elif "STEP 9:" in src or "STEP 10:" in src or "STEP 11:" in src or "STEP 12:" in src:
                src = apply_name_map(src)
                if "STEP 9:" in src:
                    src = patch_step9_comments(src)
            elif "STEP 13:" in src:
                src = patch_step13_add_constants(src)
                src = apply_name_map(src)
            elif "STEP 14:" in src or "STEP 15:" in src:
                src = apply_name_map(src)
                src = patch_step14_15_conditionals(src)
            elif "STEP 16:" in src:
                src = WINDOW_ABLATION_SOURCE
            elif "STEP 17:" in src or "STEP 19:" in src:
                src = apply_name_map(src)
            elif "STEP 18:" in src:
                src = apply_name_map(src)
                src = patch_hybrid_comment(src)
            elif "STEP 21:" in src or "STEP 22:" in src or "STEP 23:" in src:
                src = patch_models_to_plot(src)
            set_src(cell, src)
        elif cell["cell_type"] == "markdown":
            if "How I prepare the input windows" in src:
                src = patch_markdown_windows(src)
            elif "30-day window instead of 60" in src:
                src = src.replace(
                    "1. **30-day window instead of 60.** Does the model really need three months of history, or is six weeks enough? A shorter window gives the model less context per prediction, but each sample is smaller so training is faster.",
                    "1. **Window ablation (1W / 1M / 1Y).** I compare 5, 21, and 252 trading-day lookbacks using the same GRU, because finance studies often report these horizons.",
                )
            set_src(cell, src)

        new_cells.append(cell)

        if cell["cell_type"] == "code" and "# STEP 4:" in src_join(cell):
            c = deepcopy(CANDLESTICK_CELL)
            set_src(c, CANDLESTICK_SOURCE)
            new_cells.append(c)

        if cell["cell_type"] == "code" and "# STEP 20:" in src_join(cell):
            c = {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": []}
            set_src(c, COMPARE_CELL_SOURCE)
            new_cells.append(c)

    nb["cells"] = new_cells
    clear_outputs(nb)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("Patched", NB_PATH, "— cells:", len(nb["cells"]))


if __name__ == "__main__":
    main()
