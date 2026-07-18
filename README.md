# COM7019 — Stock Price Forecasting

MSc assignment (Arden University): compare **LSTM**, **GRU**, and a **CNN-LSTM** hybrid for one-step-ahead stock price prediction on historical OHLCV data.

Student ID: `25199053`

## Highlights

- Chronological train/validation/test split (no leakage)
- Persistence baseline before any neural model
- Dropout, early stopping, and fine-tuning experiments
- Trading-day windows: 1W (5), 1M (21), 1Y (252)
- Frozen metrics and figures from a full GPU run

**Headline result:** CNN-LSTM best test RMSE **3.2277** vs persistence baseline **3.2380**. Among the required architectures, **GRU** matches LSTM accuracy at lower parameter cost.

## Repository layout

```text
data/          Authorised dataset (brief [3921])
notebook/      Assignment notebook (run this)
results/       Figures, tables, and model summaries
src/com7019/   Small reusable helpers (config, data, evaluation)
docs/          Short methodology note
```

## Quick start

### Google Colab (recommended)

1. Upload `notebook/COM7019_25199053.ipynb` and `data/Stock_Price_Data_[3921].csv`.
2. Runtime → Run all.
3. Download the generated `output/` folder if you need local copies.

### Local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Place the CSV next to the notebook (matches Colab path)
cp data/Stock_Price_Data_[3921].csv notebook/
jupyter notebook notebook/COM7019_25199053.ipynb
```

Full training is slow on CPU; use Colab with GPU when possible.

### Docker

```bash
docker compose run notebook
```

## Results

See `results/tables/results_all_runs.csv` and `results/figures/`.

| Model | Test RMSE |
|-------|----------:|
| CNN-LSTM hybrid | 3.2277 |
| GRU (LR=1e-4) | 3.2339 |
| Persistence baseline | 3.2380 |

## Licence

MIT for the code in this repository. The CSV is provided for the COM7019 assessment; keep academic-use context in mind if you reuse it.
