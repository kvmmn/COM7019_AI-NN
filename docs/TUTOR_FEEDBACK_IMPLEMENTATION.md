# Tutor feedback implementation log (Jul 2026)

| # | Tutor request | Implementation | Status |
|---|---------------|----------------|--------|
| 1 | Professional GitHub repo (Docker, UV, requirements, reusable) | `pyproject.toml`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `src/com7019/`, `README.md` | **Done** — https://github.com/kvmmn/COM7019_AI-NN |
| 2 | Explain unpredictable nature of financial data | STEP 3 prose + report Sections 2 and 5 | **Done** |
| 3 | Richer EDA with candlestick charts | STEP 3b: mplfinance + Plotly HTML; three windows (last 60d, Oct19–Apr20, Mar20 zoom) | **Done** (Colab run 13 Jul) |
| 4 | Standard windows 1W(5), 1M(21), 1Y(252) trading days | Primary = 1M; STEP 16 ablates all three | **Done** |
| 5 | Readable model names | Keras-safe descriptive names (e.g. `LSTM_1Layer_64Units_Dropout0p2_Win1Month_21d`) | **Done** |
| 6 | Graphical model comparison | STEP 20b: Figures 4a–4e (zoomed metrics, Δ baseline, log cost, business efficiency, rank heatmap); Figure 5 window ablation | **Done** |
| 7 | Business / advisory reflection in Task 2 | Subsection in `DRAFT_REPORT.md` Section 5 | **Done** |
| 8 | Regularisation matrix (optional) | Skipped — no clear finance-EDA justification | **N/A** |

## Colab re-run (13 Jul 2026)

- Notebook: `_code/COM7019_25199053.ipynb` — 27/27 cells, 0 errors
- Outputs: `_code/output/`
- Ledger: `_brain/RESULTS_LEDGER.md` frozen
- Report: `_brain/DRAFT_REPORT.md` updated with all numbers

## Remaining before submission

1. Capture step screenshots into `_code/screenshots/` (optional if notebook outputs suffice)
2. Harvard references from `SOURCE_LEDGER.md`
3. Assemble final `COM7019_Portfolio_25199053.docx`
