#!/usr/bin/env python3
"""Surgical viz patch: output folder, candlesticks, full model comparison."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "_code" / "COM7019_25199053.ipynb"

CANDLESTICK_SOURCE = r'''# ============================================================
# STEP 3b: Exploratory plots — candlestick charts (financial EDA)
# ============================================================
# Line charts of Close alone hide the daily range (High/Low) and where
# the price opened vs closed. For OHLCV stock data the standard view is
# a candlestick chart: each day is one candle (body = Open→Close, wicks =
# High/Low).
#
# I use mplfinance (finance-standard OHLCV layout) and also export a
# Plotly version. Plotly removes weekend gaps on the x-axis so every
# trading day is visible without empty slots, which makes dense periods
# (e.g. March 2020) easier to read in the report.
#
# Technical fixes for clipping / "missing" wicks:
#   - scale_padding adds headroom above High and below Low
#   - panel_ratios gives the price panel more vertical space than volume
#   - returnfig + subplots_adjust so the title does not cover candles
#   - validate each slice: print min Low / max High so I can confirm range
#
# Three windows (all TRADING days):
#   A) Last 60 days — recent behaviour (includes COVID crash tail)
#   B) Oct 2019 → Apr 2020 (~130 days) — full run-up + crash context
#   C) Mar 2020 only (35 days) — zoom on extreme volatility
#
# mav=(5, 21) overlays 1W and 1M moving averages (same as model windows).

import mplfinance as mpf

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("plotly not installed — mplfinance charts only (pip install plotly kaleido)")


def _ohlcv_slice(start_idx, n_days):
    """Return mplfinance-ready OHLCV with Date index."""
    chunk = df.iloc[start_idx : start_idx + n_days].copy()
    chunk = chunk.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
    return chunk


def plot_candlestick_mpl(ohlcv, filename, title):
    """Candlestick + volume with padding so wicks are never clipped."""
    save_path = OUTPUT_FOLDER + "/figures/" + filename
    style = mpf.make_mpf_style(
        base_mpf_style="yahoo",
        gridstyle=":",
        y_on_right=False,
        rc={"font.size": 10},
    )
    kwargs = dict(
        type="candle",
        style=style,
        title=title,
        volume=True,
        mav=(5, 21),
        mavcolors=("steelblue", "darkorange"),
        figsize=(14, 9),
        panel_ratios=(4, 1),
        scale_padding=dict(left=0.12, right=0.12, top=0.45, bottom=0.18),
        tight_layout=True,
        returnfig=True,
    )
    fig, axes = mpf.plot(ohlcv, **kwargs)
    # Extra y-margin beyond mplfinance padding (guards against title overlap)
    price_ax = axes[0]
    lo, hi = float(ohlcv["Low"].min()), float(ohlcv["High"].max())
    pad = (hi - lo) * 0.06
    price_ax.set_ylim(lo - pad, hi + pad)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.96)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(
        "Saved:", save_path,
        "| range:", round(lo, 2), "→", round(hi, 2),
        "| days:", len(ohlcv),
    )


def plot_candlestick_plotly(ohlcv, filename, title):
    """Plotly candlestick: no weekend gaps, full wick visibility."""
    if not HAS_PLOTLY:
        return
    html_path = OUTPUT_FOLDER + "/figures/" + filename.replace(".png", ".html")
    png_path = OUTPUT_FOLDER + "/figures/" + filename
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.72, 0.28],
    )
    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv["Open"], high=ohlcv["High"],
            low=ohlcv["Low"], close=ohlcv["Close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            name="OHLC",
        ),
        row=1, col=1,
    )
    colours = [
        "#26a69a" if c >= o else "#ef5350"
        for o, c in zip(ohlcv["Open"], ohlcv["Close"])
    ]
    fig.add_trace(
        go.Bar(x=ohlcv.index, y=ohlcv["Volume"], marker_color=colours, name="Volume"),
        row=2, col=1,
    )
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=720,
        margin=dict(t=60, b=40, l=60, r=30),
        showlegend=False,
    )
    # Hide non-trading gaps (weekends) so candles are evenly spaced
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
        tickformat="%d %b %Y",
    )
    fig.write_html(html_path)
    try:
        fig.write_image(png_path, scale=2)
        print("Saved:", png_path, "(plotly)")
    except Exception as exc:
        print("Plotly PNG skipped (install kaleido):", exc)
    print("Saved:", html_path)


def plot_candlestick_slice(start_idx, n_days, filename, title):
    ohlcv = _ohlcv_slice(start_idx, n_days)
    plot_candlestick_mpl(ohlcv, filename, title)
    plot_candlestick_plotly(ohlcv, filename.replace(".png", "_plotly.png"), title)


# --- Figure A: last 60 trading days ---
plot_candlestick_slice(
    len(df) - 60,
    60,
    "eda_candlestick_last60days.png",
    "EDA Figure A: Last 60 trading days (candlestick + 5/21-day MAs)",
)

# --- Figure B: Oct 2019 through Apr 2020 (run-up + crash; NOT Jan–May 2019 only) ---
idx_oct2019 = int(df[df["Date"] >= "2019-10-01"].index[0])
n_context = min(130, len(df) - idx_oct2019)
plot_candlestick_slice(
    idx_oct2019,
    n_context,
    "eda_candlestick_oct2019_apr2020.png",
    "EDA Figure B: Oct 2019 – Apr 2020 (run-up + Mar 2020 crash)",
)

# --- Figure C: March 2020 zoom (35 trading days around the crash) ---
idx_mar2020 = int(df[df["Date"] >= "2020-02-24"].index[0])
plot_candlestick_slice(
    idx_mar2020,
    35,
    "eda_candlestick_mar2020_zoom.png",
    "EDA Figure C: Feb–Mar 2020 zoom (high-volatility candles)",
)
'''

COMPARISON_SOURCE = r'''# ============================================================
# STEP 20b: Graphical comparison — ALL models, multiple lenses
# ============================================================
# The results table is exact but every model scores within a few cents
# of the persistence baseline, so plain bar charts look "flat". I compare
# EVERY run (not a top-12 subset) using views that separate models when
# absolute numbers are nearly identical:
#
#   Panel 1 — Zoomed absolute RMSE / MAE / MAPE (y-axis starts near min,
#             not at zero, so small gaps become visible)
#   Panel 2 — Δ vs baseline: RMSE − baseline_RMSE (negative = beats baseline)
#   Panel 3 — Train time & parameter count (log scale; spans orders of mag.)
#   Panel 4 — Business lens: RMSE improvement per training minute and per
#             10k parameters (operational cost vs accuracy gain)
#   Panel 5 — Rank heatmap: normalised ranks across accuracy, speed, size
#
# Short labels keep the x-axis readable; full names stay in the CSV table.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE_NAME = "Baseline (persistence)"
baseline_row = results_df.loc[results_df["Model"] == BASELINE_NAME].iloc[0]
b_rmse = float(baseline_row["Test RMSE"])

cmp = results_df.copy()
cmp["Parameters"] = cmp["Parameters"].fillna(0).astype(float)
cmp["Train time (s)"] = cmp["Train time (s)"].fillna(0).astype(float)
cmp = cmp.sort_values("Test RMSE").reset_index(drop=True)

# Readable x labels (full name in CSV; shortened for plots)
def short_label(name):
    if name == BASELINE_NAME:
        return "Baseline"
    s = name.replace("_Win1Month_21d", "").replace("_Win1Week_5d", "_1W").replace("_Win1Year_252d", "_1Y")
    s = s.replace("LSTM_1Layer_64Units_Dropout0p2", "LSTM-1L")
    s = s.replace("LSTM_2Layer_64-32Units_Dropout0p2", "LSTM-2L")
    s = s.replace("GRU_1Layer_64Units_Dropout0p2", "GRU-1L")
    s = s.replace("GRU_2Layer_64-32Units_Dropout0p2", "GRU-2L")
    s = s.replace("Hybrid_CNN32_LSTM64_Dropout0p2", "CNN-LSTM")
    s = s.replace("NoDropout", "no-drop")
    s = s.replace("FineTuned", "fine-tune")
    s = s.replace("LR0p0001", "LR↓")
    if len(s) > 28:
        s = s[:25] + "…"
    return s

cmp["Label"] = cmp["Model"].map(short_label)
cmp["Δ RMSE vs baseline"] = cmp["Test RMSE"] - b_rmse
cmp["Beats baseline"] = cmp["Δ RMSE vs baseline"] < 0
cmp["RMSE gain USD"] = np.maximum(0.0, b_rmse - cmp["Test RMSE"])
cmp["Gain per train min"] = np.where(
    cmp["Train time (s)"] > 0,
    cmp["RMSE gain USD"] / (cmp["Train time (s)"] / 60.0),
    0.0,
)
cmp["Gain per 10k params"] = np.where(
    cmp["Parameters"] > 0,
    cmp["RMSE gain USD"] / (cmp["Parameters"] / 10000.0),
    0.0,
)

x = np.arange(len(cmp))
bar_colors = [
    "#888888" if m == BASELINE_NAME else ("#2ca02c" if b else "#d62728")
    for m, b in zip(cmp["Model"], cmp["Beats baseline"])
]

# --- Figure 4a: zoomed accuracy metrics (all models) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
for ax, col, ylab in zip(
    axes,
    ["Test RMSE", "Test MAE", "Test MAPE %"],
    ["RMSE (USD)", "MAE (USD)", "MAPE (%)"],
):
    vals = cmp[col].astype(float)
    margin = max((vals.max() - vals.min()) * 0.35, vals.max() * 0.002)
    ax.bar(x, vals, color=bar_colors, edgecolor="white", linewidth=0.4)
    ax.set_title(col + " (zoomed axis)")
    ax.set_ylabel(ylab)
    ax.set_xticks(x)
    ax.set_xticklabels(cmp["Label"], rotation=75, ha="right", fontsize=7)
    ax.set_ylim(vals.min() - margin, vals.max() + margin)
    ax.axhline(baseline_row[col], color="black", linestyle="--", linewidth=0.8, alpha=0.6)
fig.suptitle("Figure 4a: All models — accuracy (zoomed; dashed = baseline)", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/figures/fig4a_accuracy_zoomed.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 4b: delta vs baseline (makes tiny RMSE gaps visible) ---
fig, ax = plt.subplots(figsize=(14, 5))
delta = cmp["Δ RMSE vs baseline"]
colors_delta = ["#888888" if m == BASELINE_NAME else ("#2ca02c" if d < 0 else "#d62728")
                for m, d in zip(cmp["Model"], delta)]
ax.bar(x, delta, color=colors_delta, edgecolor="white", linewidth=0.4)
ax.axhline(0, color="black", linewidth=0.9)
ax.set_ylabel("Δ RMSE vs baseline (USD)\nnegative = beats baseline")
ax.set_xticks(x)
ax.set_xticklabels(cmp["Label"], rotation=75, ha="right", fontsize=7)
ax.set_title("Figure 4b: Improvement over persistence baseline (all models)")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/figures/fig4b_delta_vs_baseline.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 4c: log-scale cost & complexity (neural nets only) ---
nn = cmp[cmp["Model"] != BASELINE_NAME].copy()
xn = np.arange(len(nn))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, title in zip(
    axes,
    ["Train time (s)", "Parameters"],
    ["Training time (log scale)", "Parameter count (log scale)"],
):
    vals = nn[col].replace(0, np.nan).astype(float)
    ax.bar(xn, vals, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xticks(xn)
    ax.set_xticklabels(nn["Label"], rotation=75, ha="right", fontsize=7)
fig.suptitle("Figure 4c: Training cost & model size (all neural nets)", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/figures/fig4c_cost_logscale.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 4d: business efficiency (accuracy gain vs operational cost) ---
nn2 = nn[nn["RMSE gain USD"] > 0].copy()
if len(nn2) == 0:
    nn2 = nn.copy()  # none beat baseline — still plot zeros for transparency
xn2 = np.arange(len(nn2))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(xn2, nn2["Gain per train min"], color="seagreen", edgecolor="white", linewidth=0.4)
axes[0].set_title("USD RMSE gain per training minute")
axes[0].set_xticks(xn2)
axes[0].set_xticklabels(nn2["Label"], rotation=75, ha="right", fontsize=7)
axes[1].bar(xn2, nn2["Gain per 10k params"], color="darkorange", edgecolor="white", linewidth=0.4)
axes[1].set_title("USD RMSE gain per 10k parameters")
axes[1].set_xticks(xn2)
axes[1].set_xticklabels(nn2["Label"], rotation=75, ha="right", fontsize=7)
fig.suptitle("Figure 4d: Business lens — accuracy gain vs deployment cost", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/figures/fig4d_business_efficiency.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 4e: rank heatmap (1 = best) across multiple criteria ---
rank_df = cmp[cmp["Model"] != BASELINE_NAME].copy()
rank_cols = {
    "Test RMSE": "accuracy",
    "Train time (s)": "train speed",
    "Parameters": "compactness",
}
rank_mat = pd.DataFrame(index=rank_df["Label"])
for col, nice in rank_cols.items():
    rank_mat[nice] = rank_df[col].rank(method="min", ascending=True).values
# Average rank (lower = better overall trade-off)
rank_mat["avg rank"] = rank_mat.mean(axis=1)
rank_mat = rank_mat.sort_values("avg rank")
fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(rank_mat))))
im = ax.imshow(rank_mat.values, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=len(rank_mat))
ax.set_xticks(range(rank_mat.shape[1]))
ax.set_xticklabels(rank_mat.columns, rotation=30, ha="right")
ax.set_yticks(range(len(rank_mat)))
ax.set_yticklabels(rank_mat.index, fontsize=8)
for i in range(rank_mat.shape[0]):
    for j in range(rank_mat.shape[1]):
        ax.text(j, i, f"{rank_mat.values[i, j]:.0f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, ax=ax, label="rank (1 = best)")
ax.set_title("Figure 4e: Multi-criteria rank heatmap (all neural nets)")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + "/figures/fig4e_rank_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# Save enriched comparison table for the report
cmp.to_csv(OUTPUT_FOLDER + "/tables/results_comparison_enriched.csv", index=False)
print("Saved enriched comparison:", OUTPUT_FOLDER + "/tables/results_comparison_enriched.csv")
'''

LOSS_CURVES_SOURCE = r'''# ============================================================
# STEP 21: Plot of training loss curves for main models
# ============================================================
# These curves show HOW each model trained, not just the final
# score. What to look for:
#   - both curves falling together  → healthy learning
#   - train falling, validation rising → overfitting
#   - the gap between the curves    → how much the model memorises
# Early stopping cut each run where validation loss stopped
# improving, so the curves end at different epochs.
#
# Because MSE loss values are also very close across models, I plot
# BOTH the raw scale and a log-y view so small validation gaps show up.

models_to_plot = [NAME_LSTM_2L, NAME_GRU_1L, "Hybrid_CNN32_LSTM64_Dropout0p2_Win1Month_21d"]

for use_log, tag, fname in [
    (False, "linear y", "fig2_loss_curves.png"),
    (True, "log y (zoom)", "fig2_loss_curves_log.png"),
]:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, name in zip(axes, models_to_plot):
        history = training_histories[name]
        ax.plot(history["loss"], label="train", color="steelblue")
        ax.plot(history["val_loss"], label="validation", color="seagreen")
        if use_log:
            ax.set_yscale("log")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("MSE loss")
    fig.suptitle("Figure 2: Training and validation loss (" + tag + ")")
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER + "/figures/" + fname, dpi=150)
    plt.show()
'''


def set_src(cell: dict, text: str) -> None:
    lines = text.split("\n")
    cell["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))

        if 'OUTPUT_FOLDER = "outputs_student"' in src:
            src = src.replace('OUTPUT_FOLDER = "outputs_student"', 'OUTPUT_FOLDER = "output"')

        if "all_results.append({\n" in src and "Baseline (persistence)" in src:
            src = src.replace(
                '"Test MAPE %": baseline_mape,\n}',
                '"Test MAPE %": baseline_mape,\n'
                '    "Parameters": 0,\n'
                '    "Train time (s)": 0.0,\n}',
            )

        if "STEP 3b:" in src and "plot_candlestick_slice" in src:
            set_src(cell, CANDLESTICK_SOURCE)
            continue

        if "STEP 20b:" in src and "fig4_model_comparison_bars" in src:
            set_src(cell, COMPARISON_SOURCE)
            continue

        if "STEP 21:" in src and "models_to_plot" in src:
            set_src(cell, LOSS_CURVES_SOURCE)
            continue

        set_src(cell, src)

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Patched:", NB_PATH)


if __name__ == "__main__":
    main()
