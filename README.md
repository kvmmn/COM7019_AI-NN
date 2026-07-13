# COM7019 — Stock Price Forecasting (LSTM / GRU / CNN-LSTM)

**Student ID:** 25199053 · **Module:** COM7019 Artificial Intelligence and Neural Networks

Reproducible MSc assignment project comparing recurrent neural networks on historical stock OHLCV data. Includes a professional Python package, Docker/UV setup, and a Colab-ready Jupyter notebook.

## What this repo contains

| Path | Purpose |
|------|---------|
| `_code/COM7019_25199053.ipynb` | Main assignment notebook (Task 1 evidence) |
| `src/com7019/` | Reusable Python package (data, config, evaluation, plots) |
| `_brain/` | Report draft, results ledger, rubric evidence (working files) |
| `Stock_Price_Data_[3921].csv` | Authorised dataset (brief [3921]) |

## Quick start (local, UV)

```bash
# Install UV: https://docs.astral.sh/uv/
cd "COM7019 - Assignment"
uv sync
source .venv/bin/activate   # or: uv run ...

# Run notebook locally (CPU; full training is slow — prefer Colab)
jupyter notebook _code/COM7019_25199053.ipynb
```

## Quick start (Google Colab — recommended for GPU)

1. Upload `COM7019_25199053.ipynb` and `Stock_Price_Data_[3921].csv` to Colab.
2. Run all cells. Outputs save to `output/` (inside `_code/` when the notebook lives there).
3. Download `output/` and the executed notebook for the report appendix.

## Docker

```bash
docker compose build
docker compose run notebook   # executes notebook headlessly (CPU, slow)
docker compose run shell      # interactive dev shell
```

## Trading-day windows (finance standard)

Windows use **trading days**, not calendar days:

| Label | Trading days | Typical use |
|-------|-------------|-------------|
| 1W | 5 | Short-term momentum |
| 1M | 21 | ~one trading month (primary experiments) |
| 1Y | 252 | ~one trading year (annual context) |

Primary model comparison uses **1M (21 days)**. Window ablation compares all three.

## Model naming convention

Names encode every setting, e.g.:

`GRU_1Layer_64Units_NoDropout_Window1Month_21TradingDays`

Readable without opening the code.

## Outputs

After a full run, `output/` contains:

- `figures/` — EDA candlesticks, architecture diagrams, loss curves, comparison charts (4a–4e)
- `tables/results_all_runs.csv` — frozen metrics for the report
- `tables/results_comparison_enriched.csv` — Δ baseline, business efficiency columns
- `models/model_summaries.txt` — architecture evidence

## Development

```bash
uv run ruff check src/
python scripts/patch_notebook_viz_v2.py   # apply viz patches (do not use upgrade_notebook.py)
```

## Licence

MIT — academic assignment code. Dataset © assessment brief; do not redistribute outside submission context.

## Changelog

- **v0.3** — Full Colab re-run (13 Jul 2026): `output/` folder, candlestick EDA, all-model comparison (4a–4e), log/zoom charts, 1M primary window. Docs and ledger frozen. Repo: https://github.com/kvmmn/COM7019_AI-NN
- **v0.2** — Tutor feedback (Jul 2026): GitHub packaging, candlestick EDA, 1W/1M/1Y windows, descriptive model names, comparison plots, business reflection in report.
- **v0.1** — Initial Colab run with LSTM/GRU/hybrid comparison.
